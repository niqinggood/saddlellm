# 原生推理引擎：缓存与生成优化

2026-09-29 新增可选的 [统一 token 预算调度与分块 prefill](NATIVE_CHUNKED_PREFILL_20260929.md)，开启后统一流式/非流式文本请求；默认行为不变。向服务端并发及本地运行能力推进的阶段计划见 [原生引擎路线](NATIVE_ENGINE_ROADMAP.md)。

跨阶段的问题与改动清单、实验结果和下一步优先级，先看 [推理优化复盘与后续路线](INFERENCE_OPTIMIZATION_REVIEW_AND_ROADMAP.md)。
2026-09-29 新增 [实验性逐步合批执行器](NATIVE_BATCH_DECODE_20260929.md)，不替代本文的已有固定微批，也尚未接入默认服务。

原生模型 `SaddleForCausalLM.generate()` 默认使用请求独立的几何增长缓存。
`saddlellm serve MODEL` 无需新增参数，加载原生模型时自动使用这条路径。
Hugging Face 模型继续由其自身的 generate 实现管理缓存。

2.38 新增明确的 `--backend native`，可将已支持的标准 Qwen3 safetensors 直接加载到
Saddle 模型中使用这条原生路径；默认 local 对 HF 模型的行为仍不变。
范围、同权重校验与数值差异见 [Qwen3 原生后端](NATIVE_QWEN3_BACKEND.md)。

后续已补真实 28M 权重的 GPU 对照与新优化，见
[推理性能比较](INFERENCE_PERFORMANCE_COMPARISON.md)。后文 CPU 验证记录是此前一轮，
不应把不同轮次、不同设备状态下的绝对速度直接相减当作加速幅度。

2026-09-27 已进一步补窗口 KV 回收、可选长度桶和 token 预算调度，详见
[窗口 KV 与调度说明](INFERENCE_WINDOW_AND_SCHEDULING.md)。窗口层最多保留 W 个逻辑 key，
镜像环形存储占 2W 物理槽位；全局层不裁剪，不改变旧检查点权重或注意力范围。

## 一眼看懂执行链

```text
saddlellm serve / LocalGenerationEngine
  → 请求分词与预算校验
  → 单请求，或兼容请求的固定微批
  → NativeGenerationSession.prefill()：完整 prompt → KV + 最后位置 logits
  → NativeGenerationSession.step()：复用 logits / 增量前向 → 采样 → stop / EOS
  → 文本或 SSE、实际 token 统计
  → close()：释放本会话引用
```

| 要看什么 | 代码入口 |
| --- | --- |
| 请求入口、合批资格、流式分流、取消 | [InferenceServer.py](../saddlellm/runtime/InferenceServer.py)：LocalGenerationEngine、_generation_control、_BatchGenerationControl。 |
| 有限队列、等待合批和服务关闭 | [InferenceBatcher.py](../saddlellm/runtime/InferenceBatcher.py)：InferenceBatcher。 |
| prompt / decode 时序、采样、重复惩罚、EOS、释放 | [NativeGeneration.py](../saddlellm/models/NativeGeneration.py)：NativeGenerationSession。 |
| KV 扩容、有效切片、token / mask 写入 | [InferenceCache.py](../saddlellm/models/InferenceCache.py)：InferenceLayerCache、GenerationBuffers。 |
| 可选跨请求 prompt 快照、容量、淘汰、失效 | [NativePrefixCache.py](../saddlellm/models/NativePrefixCache.py)：NativePrefixCache。 |
| 模型前向、最后位置词表投影、generate 入口 | [SaddleModeling.py](../saddlellm/models/SaddleModeling.py)：forward、create_generation_session、generate。 |
| 推理加载时合并旧模型投影 | [InferenceOptimizer.py](../saddlellm/models/InferenceOptimizer.py)：optimize_native_inference。 |
| 单 token GQA 无重复 KV 的 SDPA 路径 | [OperatorBackends.py](../saddlellm/models/OperatorBackends.py)：_can_fold_gqa_decode、_folded_gqa_decode。 |
| token 级 stop 检查与兼容回退 | [GenerationStop.py](../saddlellm/runtime/GenerationStop.py)：StopStringRows。 |
| 流式文本的 stop 后缀缓存 | [StreamingText.py](../saddlellm/runtime/StreamingText.py)：StreamingTextBuffer。 |

核心思路是减少历史复制、冗余投影、重复控制工作，并正确管理请求生命周期。
没有更换模型权重、训练目标或注意力数学语义；不同矩阵形状与低精度仍可能带来数值差异。

## 旧模型无需重训的投影合并

`serve MODEL` 加载原生模型并进入 eval 后，将普通 Q/K/V 的三次 Linear 合并成一次，
将 SwiGLU gate/up 的两次 Linear 合并成一次。10 层模型每次前向少 30 次 Linear 调用。
只是把矩阵沿输出维拼在一起；不量化、不裁剪、不更改逻辑权重。
利用已有的 state_dict 保存/读取兼容钩子，检查点继续用原来的逻辑参数名。
不改写磁盘上的原模型，也不把运行时布局写入原来的架构配置。

```powershell
saddlellm serve MODEL --device cuda --dtype bfloat16
saddlellm serve MODEL --device cuda --dtype bfloat16 --no-fuse-projections
```

直接 import 的使用者可这样做：

```python
from saddlellm import load_causal_lm, optimize_native_inference

model = load_causal_lm("MODEL", device="cuda", dtype="bfloat16").eval()
report = optimize_native_inference(model)  # 原地处理；返回报告，不返回模型
print(report.to_dict())
tokens = model.generate(input_ids, max_new_tokens=32)
```

只在模型独占、开始服务之前执行，不与请求并发。操作可重复调用，已合并模块不再次复制。
MLA 的低秩投影不强行合并；带局部 hooks、adapter、量化或自定义 Linear 的模块跳过。
MoE 中符合条件的 SwiGLU 专家同样可以合并。一次只处理一组权重，不保留整个旧模型的副本，
但单组权重转换期间仍有新旧两份的瞬时内存开销。
此操作更换参数对象；不要对已有 optimizer、adapter、编译图执行，后续训练应重新加载模型。
普通 load_causal_lm 不自动调用它，因此原训练流程不需要变化。

默认固定的残差系数 1.0 同时走恒等路径，避免无意义乘法；可训练的残差系数不跳过，
非 1.0 系数、dropout、串行/并行残差语义保持不变。

## GQA 单 token 解码：不重复复制 K/V

当一个 KV 头对应多个查询头时，旧 SDPA 回退路径要把 K/V 重复展开到查询头数。
新路径在无 mask 的单 token 解码中，把同组查询头临时放到查询行维度：
每行仍查询相同历史 K/V，输出再恢复为原来的头布局。不建立重复 K/V 副本。

自动条件严格限定为 CUDA、无梯度、一个 query token、无 mask、is_causal=false、
合法的整数组头比例和内置 SDPA 处理器。完整 prefill、多 token 解码、PAD/SWA/文档 mask、
训练梯度和自定义处理器维持原路径。有可用的原生 Flash GQA 时仍优先使用它。
这不是新 CUDA 算子，也不增加用户参数；运算仍调用现有 SDPA。

数学目标相同，但不同内核/矩阵布局可能改变浮点舍入，所以同时检查注意力数值误差和
真实模型输出。小模型的整段收益目前约 1–3%，不能拿长历史注意力局部加速比替代它。
独立复现局部工作量可用：

```powershell
python tools/benchmark_gqa_decode.py --device cuda --dtype bfloat16 --history 4096
```

## 请求独立的增量缓存

缓存按需分配，容量不足时翻倍，且不超过输入长度加生成预算。
容量足够时只写入新增 token，不拼接和复制整段历史 KV。
注意力只读取已经写入的有效切片。GQA/MQA/MHA 和 MLA latent
缓存均可使用；MLA latent 的历史 K/V 投影计算仍然存在。
训练 forward 和已有 tuple/dict 缓存接口保持兼容。

Python 用户可以用旧路径对比输出或性能：

```python
output = model.generate(input_ids, max_new_tokens=128)
reference = model.generate(
    input_ids, max_new_tokens=128, cache_implementation="dynamic"
)
```

新缓存会保留少量未使用容量，以减少分配次数；扩容时新旧缓冲区
同时存在，因此仍可能提高瞬时显存。它尚不是分页 KV Cache，
该请求缓存本身不跨请求共享；现在可额外选择下面的 prompt 快照缓存。
窗口层已自动使用有界环形缓存；连续批处理和窗口感知跨请求前缀复用仍未实现。

运行可重复的缓存微基准：

```powershell
python tools/benchmark_inference_cache.py --device cpu
python tools/benchmark_inference_cache.py --device cuda
```

报告只测单层 KV 写入开销，不代表整个模型 tokens/s。

## 可选的跨请求前缀缓存

默认关闭，不增加普通请求的缓存查找开销。原生 dense GQA/MHA/MQA 模型可开启：

```powershell
saddlellm serve MODEL --prefix-cache-mb 64
```

`64` 指最多保留 64 MiB 的缓存 tensor，不是总显存上限。默认至少 16 个 prompt token，
最多 128 个条目；超限按 LRU 淘汰，单个条目超预算则不保存。缓存只保存在本进程内存，
不写磁盘；服务关闭时释放。GPU allocator 的保留和异步释放不属于这个逻辑预算。

开启后默认只命中完全相同的 token 提示词，复用其 prompt KV 和未加工的下一 token logits。
这不缓存最终答案，不跳过后续采样或 decode。temperature、重复惩罚、EOS、stop、RNG
仍是每个请求独立的状态。相同文字经模板、分词、截断后 token 不同，则不能完整命中。

希望复用“相同开头、不同结尾”的提示词，可明确选择实验路径：

```powershell
saddlellm serve MODEL --prefix-cache-mb 64 --prefix-cache-partial
```

命中后先复制有效 KV 到本请求的私有缓冲区，再一次计算所有未命中的 prompt token；
不会错误地只计算尾部一个 token。新 prompt 比旧快照短时，若没有对应结束位置的 logits，
重新计算最后一个 prompt token。各请求之后的 KV 写入互不影响。
这是 copy-on-hit，不是分页 KV、零复制共享或连续批处理。

当前仅支持单行、无 padding、geometric KV、`logits_to_keep=1` 的普通原生 dense 模型。
普通生成与原生流式共用；固定合批、dynamic/no-cache、MLA/MoE、HF、自定义模块、
forward hooks、adapter、量化和编译包装继续原路径。HF/custom 不接收该原生参数。
`/health.prefix_cache` 只报告命中、复用 token 数、保存、淘汰、失效及保留 tensor 字节，
不暴露 prompt/token/hash。`enabled` 表示已配置缓存，不代表每个请求都满足资格。

直接 import：

```python
from saddlellm import NativePrefixCache, load_causal_lm

model = load_causal_lm("MODEL", device="cuda", dtype="bfloat16").eval()
cache = NativePrefixCache(max_bytes=64 * 1024 * 1024)
# 两次调用可命中相同 input_ids；默认不部分复用。
tokens = model.generate(input_ids, max_new_tokens=32, prefix_cache=cache)
tokens = model.generate(input_ids, max_new_tokens=32, prefix_cache=cache)
print(cache.snapshot())
cache.clear()  # 丢弃快照，已恢复 KV 的请求不受影响
cache.close()  # 释放并禁用后续查找/保存
```

高级使用者可构造 `NativePrefixCache(..., reuse_partial_prefixes=True)`。
正常参数修改、load_state_dict、参数对象/设备/精度、配置和执行策略变化，在下次合格查找时
会检查并清空旧条目；不是模型修改瞬间主动释放旧显存。模型不能与请求并发修改。
`.data` 等不更新版本号的修改、外部自定义行为需要显式 `clear()`，后续训练应关闭缓存。
只在可信的单进程/租户范围共享；本服务尚无多租户身份隔离，不宣称具备安全跨租户缓存。

缓存发布发生在 prompt 成功前向之后、采样修改 logits 之前；保存的是独立紧凑快照，
不保留请求的大容量 KV backing storage。CUDA 跨 stream 用 event 等待和 record_stream
保护缓存复制；清空/关闭期间的旧发布 ticket 作废。复制或保存遇到 CUDA OOM 时清理
可选快照并回到普通生成，但普通生成本身仍可能因显存不足失败。

短 prompt、小模型、低命中率时，查找、私有复制和保存开销可能大于收益。部分复用改变
矩阵形状，可能产生浮点舍入差异；算法等价不是所有任务的位级一致或质量保证。
当前 28M 实测只有首步准备明显减少，整段收益很小且不稳定，部分命中更慢，所以不默认开启。
实测和失败边界见 [性能比较](INFERENCE_PERFORMANCE_COMPARISON.md#prefix-cache-results)。

```powershell
python tools/benchmark_prefix_cache.py --model MODEL --device cuda --dtype float32
```

工具单独比较不使用缓存、冷请求、完整命中和部分命中；缓存预热在计时外，冷请求包含保存。
它检查生成 token，并报告 model-only prefill 和整段生成，不是 HTTP 首 token 延迟。
评估实际收益还需固定模型、输入/输出长度、精度和硬件，测量
完整生成速度、首 token 延迟及显存峰值。

## 最后位置词表投影

原生 `generate()` 默认只计算最后一个位置的 logits。处理长输入时，
全部输入仍然经过 Transformer 并建立 KV Cache，但词表投影不再生成
每个输入位置的完整词表分数。普通 `forward()` 默认返回全部 logits，
训练和评估保持原有语义。

高级调用可以使用 `model(input_ids, logits_to_keep=1)` 获取最后位置 logits。
默认值 0 表示保留全部位置；传入 labels 时必须为 0，以防丢失训练监督。
输出的 hidden_states 不被截断。`generate(logits_to_keep=0)` 可用于投影对照。

## token 和 mask 缓冲区

每次原生生成预留输入长度加本次生成预算的 token/mask 空间。
后续步骤写入新列，不再逐步拼接并复制完整序列。注意力只能看到有效列，
提前完成的样本按原有规则写入 PAD 和 mask。输入 tensor 不会被改写。
返回结果会复制为独立的实际长度 tensor，避免短回答保留整个预留缓冲区。
此改动不回收已完成 batch 行的模型计算，连续批处理仍需后续调度实现。

## 整段生成基准

```powershell
python tools/benchmark_native_generation.py --device cpu
python tools/benchmark_native_generation.py --device cuda
```

这是离线随机初始化的小模型基准。参考路径使用完整 logits 和逐步拼接，
优化路径使用最后位置 logits 和预留缓冲区；两者使用相同权重和几何 KV Cache。
脚本检查输出 token 一致性，并报告整段生成耗时、词表投影位置数，
CUDA 下还报告增量峰值显存。结果不代表实际训练模型或线上并发的收益。
两项优化均使用现有 PyTorch 算子，无需安装额外自定义算子。

对于没有 padding 的输入，generate 只在入口检查一次显式传入的全一 mask，
后续不再把冗余全一 mask 传给 forward，避免逐 token 重复扫描及 GPU/CPU 同步。
当 batch 中有行提前完成时，会恢复传递有效 mask，屏蔽后续 PAD；有初始 padding 的
请求仍使用完整 mask。普通训练和评估的 forward 行为不变。

## 原生非流式请求微批处理

默认服务行为不变。需要把兼容请求合成一次生成时，只加一个参数：

```powershell
saddlellm serve outputs/release --batch-size 4
```

本阶段只合并原生 SaddleLLM 模型的非流式、贪心请求：客户端设置
`temperature=0`、`stream=false`。服务会把实际 token 长度、有效输出预算和
生成参数相同的请求放入同一批，避免不同长度输入带来的额外 padding 和位置语义变化。
这是默认行为；可选 `--batch-length-bucket-size 64` 可合并相近长度，左 padding 不计入 usage。
重复惩罚不为 1、已有 padding/额外编码字段或自定义生成等情况仍回退严格等长分组。
`--batch-token-budget` 限制包含 padding 的 prompt + 输出槽位；`--max-queued-tokens`
可限制待处理预留量。HTTP 短 prompt 优先且带最老请求防饥饿，详见上面的新指南。
不同请求可以使用不同 stop 字符串；每行独立停止。提前停止后的补齐 PAD
不计入该请求的 completion_tokens，EOS 本身仍算一个生成 token。

默认最多等待 5 ms 收集同批请求，待处理队列最多 64 个；队列满返回
HTTP 503 和 `Retry-After: 1`，客户端可退避重试。取消等待中的请求会从后续批次跳过；
取消正在计算的请求会设置该行的停止事件，不影响同批其他行。
关闭服务时会通知请求停止并等待正在执行的模型线程完成。

`--batch-size` 是一次批计算的最大请求数，`--max-concurrency` 控制同时执行的
模型调用数，不是两者乘积意义上的显存保证。建议先保留默认 `--max-concurrency 1`，
从 batch_size=2 或 4 开始测显存。一个 batch 仍有多行 KV Cache，不是多个请求免费共享一份缓存。
`/health` 的 `batching` 字段显示是否可用、排队数、执行中请求数、已完成批次数和最大实际批量。

流式请求、随机采样请求、Hugging Face/custom 模型保留逐请求路径。
浏览器默认流式且 temperature=0.2，因此不会自动进入本阶段的合并路径。
这不是连续批处理：一批开始生成后不能插入新请求，已完成行仍占有该批的计算和缓存。
输入长度或输出预算差异大时，合批比例可能很低。它也不实现分页 KV Cache 或 CUDA Graph，
固定合批不使用上面的单请求前缀缓存。

函数 import 使用同一条原生批计算路径，无需启动 HTTP 服务：

```python
from saddlellm.runtime.InferenceServer import (
    CompletionRequest, InferenceServerSettings, LocalGenerationEngine,
    load_inference_model,
)

settings = InferenceServerSettings(model_path="outputs/release", batch_size=4)
model, tokenizer = load_inference_model(settings)
engine = LocalGenerationEngine(model, tokenizer, settings)
results = engine.complete_batch([
    CompletionRequest(prompt="介绍一下你自己", temperature=0, max_tokens=64),
    CompletionRequest(prompt="解释一下注意力机制", temperature=0, max_tokens=64),
])
```

`complete_batch` 和 `chat_batch` 会自动分组并按输入顺序返回结果。
输入 token 长度不同的上述两个示例可能分为两个批次；无需用户自行分组。
非原生模型和采样请求自动逐个执行，`stream=true` 则显式报错。
设置 batch_size=1 保持单请求计算；同步 import API 不使用 HTTP 排队等待。

## 批量吞吐测量

```powershell
# 离线随机小模型，不加载实际训练权重
python tools/benchmark_native_batching.py --device cpu --batch-size 4

# GPU 空闲时，对实际原生 checkpoint 测量，不会下载模型
python tools/benchmark_native_batching.py --model outputs/release --device cuda --dtype bfloat16 --batch-size 4
```

脚本固定相同输入、权重、精度和 KV 缓存策略，比较多个独立 generate 与一次批量
generate，检查输出 token 是否一致，报告总耗时、总生成 tokens/s 和 CUDA 增量峰值显存。
使用的仍是合成 token 输入，且关闭 EOS 提前停止以固定工作量；不衡量模型回答质量，
不包含分词、HTTP 队列、stop 字符串解析或服务调度开销。
CPU 默认 float32，CUDA 默认优先 bfloat16；可用 `--dtype float32` 作精度对照。
低精度、不同矩阵形状可能产生数值差异，输出不一致时脚本会失败，不能把该次结果当作已通过正确性验证。
上线前还需使用真实提示词和混合长度请求测量服务吞吐、首 token 延迟、尾部延迟和显存。

## 可暂停的逐 token 会话

`generate()` 与原生流式推理现在共享 `NativeGenerationSession`，不再维护两份
采样、缓存和停止逻辑。普通用户继续使用 `model.generate(...)` 或 `saddlellm serve`，
不需要学习新接口；编写高级推理调度器时可以显式拆开 prompt 处理和逐 token 推进：

```python
with model.create_generation_session(
    input_ids, max_new_tokens=128, do_sample=False,
) as session:
    next_logits = session.prefill()  # 处理 prompt、建立 KV，但还未输出 token
    for step in session:
        token_ids = step.token_ids  # [batch_size, 1]
        # 在两次 step 之间可以暂停，调度器决定何时继续。
    result = session.sequences()    # 需要的结果在 close 前复制出来
```

首次 `step()` 消费 prefill 已计算的 logits，不重复执行整个 prompt。
后续每个 `step()` 处理上一个生成 token、更新 KV，再采样下一个 token。
若没有显式调用 prefill，首次 step 会自动处理 prompt。
当前序列已经包含刚采样的 token，而 KV 会在下一步前向时才写入它；因此生成结束时
KV 通常比最终序列短一个 token。这是自回归执行时序，不是缓存漏写。

会话保留此前的 greedy/sampling、top-k/top-p、重复惩罚、最小输出长度、
逐行停止、geometric/dynamic 缓存和无缓存模式。每次 step 的推理上下文会在返回前退出，
不会把梯度开关或 inference_mode 状态留给调用者。共享 RoPE 常量缓存不会保存为
只能推理使用的 tensor，因此同一模型推理后再进入训练、执行 backward 的路径也已验证。

默认采样继续使用 PyTorch 全局随机数状态，保持原有行为。
原生 `generate(..., seed=123)` 或 `create_generation_session(..., seed=123)`
可改为独立请求 RNG：另一个请求采样或全局随机数状态改变，不影响该会话的随机序列。
这不保证跨硬件、不同数值精度或不同算法版本完全一致。
也可传入设备匹配的 `torch.Generator`，但不能同时指定 seed；服务 HTTP 请求暂未暴露 seed 参数。

`close()` 幂等地释放本会话持有的 KV、logits、token/mask 缓冲区、分数和模型引用，
不清除其他请求的缓存，不调用 `torch.cuda.empty_cache()`。PyTorch 分配器可能继续保留显存，
外部保存的结果、分数或缓存引用也不会被强制回收。with 退出、generate 正常结束或
执行失败时都会走清理路径。失败会话不能继续推进；并发调用 step/close 会明确拒绝，
避免半写入缓存被复用。该接口仅用于推理，不是训练 forward 的替代接口。

这是 token 级控制基础，不是连续批处理完成版：还不能往运行中的会话插入新请求、
移除已完成行或合并不同长度的已有 KV。服务的微批处理仍使用上一节的固定批次规则。

## 原生流式执行与取消

普通原生模型的流式请求直接拉取 session.step，再使用已有 TextStreamer 的分词解码规则
生成文本片段，不为每个请求额外启动后台生成线程。HTTP 仍通过线程池执行阻塞的模型
步骤，因此它不是“零线程”或异步 CUDA 引擎。Hugging Face/custom 模型及重写 generate
的原生模型保留原来的后台生成路径，避免覆盖模型自定义的生成语义。
`/health` 的 `streaming_backend` 为 `native_step` 或 `generate_worker`，便于确认实际路径。

stop 文本处理只保留最长 stop 长度减一的待确认后缀，支持跨文本片段匹配，
不会随着整段回答增长而反复保存和扫描完整已输出文本。分词解码器内部仍可能保留
token 历史；该优化仅针对文本 stop 缓冲，不宣称整个解码过程是常量空间。
遇到 EOS、stop 或客户端取消时，会保留准确的已生成 token 统计。
EOS 恰好出现在 max_tokens 的最后一位时，finish_reason 也为 stop，而不是 length。

HTTP 断连可能发生在 prompt 处理过程中。GPU 前向不能在任意算子中途安全终止，
服务会先通知取消，等待正在执行的 next(iterator) 完成，再关闭会话和释放模型调用许可。
清理使用屏蔽取消的作用域，避免返回的客户端连接已经结束，但模型仍在后台使用资源。
暂停/取消的会话不会继续偷偷生成剩余预算。

现有批量基准额外报告 `session_phases`：prefill_ms、first_sample_ms 和平均 decode_step_ms。
阶段计时逐次同步，只用于诊断；不混入前面的吞吐测试。
这些计时不包含会话创建、分词、HTTP 排队、文本流式解码和结果复制，不能直接当作
服务 TTFT 或线上每 token 延迟。`session_outputs_equal` 同时检查拆分执行与完整 generate 的输出一致。

## 采样与停止判断的热点优化

原生重复惩罚仍保持原有含义：出现过的 token 只惩罚一次，正分数除以
repetition_penalty，负分数乘以该值。现在每个请求只在第一次采样时对提示词
去重一次，随后把新 token 写入预留的紧凑索引缓冲区，通过批量 gather/scatter
处理分数。不再每一步逐行对完整历史执行 unique，也不逐步拼接索引缓冲区。
批量内短索引行使用该行已出现的 token 补齐，不会提前惩罚未出现的 PAD。
默认 repetition_penalty=1 不分配该缓冲区；close 会释放本会话持有的索引。
此实现仍读取紧凑索引，不宣称重复惩罚是常数时间或融合算子。

没有 EOS、没有 stopping_criteria 的固定预算原生生成，不再每一步读取
finished.sum().item()。有 EOS 或自定义停止条件时仍按原来的逐步判定执行，
其他前向和流式文本传输也可能发生同步，因此这不是整个生成过程“零同步”。

文本 stop 检查由请求独立的 StopStringRows 统一处理：

- 通用路径一次搬运生成 token，并且每行只解码一次，不按 stop 字符串重复解码。
- 普通原生生成满足固定行顺序、只追加 token 的契约时，自动检测是否可以启用
  ByteLevel 后缀路径。只有标准 HF fast decode、单一 ByteLevel decoder、未启用
  clean_up_tokenization_spaces 且 stop 不含 U+FFFD 时才使用该路径。
- 后缀路径只传输新增 token，保留每行不超过 4×最长 stop 字符数+4 个有效 token。
  特殊、空和未知 token 不计入这个窗口，避免 stop 跨越被跳过 token 时漏匹配。
  新增的整个 token/chunk 都会先检查，不能因为新 token 很长而丢掉其中的 stop。
- Unicode 替换字符停止词、其他 decoder、清理空格、自定义 decode/generate 和 HF
  生成继续完整解码，保留原有语义。流式和非流式、批量逐行停止共用这条路径。

命令行和请求参数不变，不需要用户选择 decoder 优化模式，也不新增重型依赖。
ByteLevel 窗口按 token 数有界；单个 token 的文本长度仍可能很大，不能称为
整个解码过程固定字节内存。该优化不改变采样随机数、模型前向或训练算法。

可以单独测量控制逻辑的工作量和耗时：

```powershell
python tools/benchmark_generation_control.py --device cpu
python tools/benchmark_generation_control.py --device cpu --batch-size 1 --prompt-tokens 16 --vocab-size 131072
```

该脚本只测重复惩罚和 stop 检查，不运行模型 forward、HTTP 或真实权重，报告
final_scores_equal、decisions_equal 和实际解码 token 数。不能把它的加速比
当作模型 tokens/s、服务首 token 延迟或完整线上推理的加速比。

训练/推理后端分离的借鉴边界见 [异步采样后端设计](ROLLOUT_BACKEND_DESIGN.md)。

## 前一轮验证记录（2026-09-27，CPU）

已运行下列推理及训练兼容性相关回归，312 项通过；这是选定范围，不是全仓库测试：

```powershell
python -m pytest -q `
  tests/test_generation_sampling.py tests/test_generation_stop.py `
  tests/test_native_generation_session.py tests/test_native_streaming.py `
  tests/test_streaming_text.py tests/test_inference_batching.py `
  tests/test_generation_fastpaths.py tests/test_inference_cache.py `
  tests/test_native_model_lifecycle.py tests/test_release_pipeline.py `
  tests/test_model_evaluation.py tests/test_completion_evaluation.py `
  tests/test_model_network_optimizations.py tests/test_model_memory_fastpaths.py `
  tests/test_chunked_lm_loss.py tests/test_rope_cache_growth.py `
  tests/test_operator_backends.py tests/test_cli_output.py tests/test_webui.py `
  tests/test_public_api.py tests/test_packed_projection_checkpoint_edges.py
```

新增采样测试与独立的旧实现逐步比较 token 和处理后的分数，覆盖贪心/采样、
不同惩罚值、缓存策略、FP32/FP16/BF16 分数、int32 输入和提前完成行的 PAD。
停止判断与完整解码逐前缀比较，覆盖中文/emoji、不完整与无效 UTF-8、长 token、
被跳过的特殊/空 token、不同 decoder、清理空格和自定义 decode。
模型测试主要在 CPU 执行；分数 dtype 测试不是 GPU 模型验证。

同日 CPU 微基准示例：Intel Core i9-14900HX，PyTorch 线程数为 1，预热后平均 3 次。
这些数字只是本机当次结果，受设备状态与输入长度影响，不是通用性能承诺。

| 测量范围 | 输入设置 | 参考 / 优化耗时 |
| --- | --- | --- |
| 仅重复惩罚控制 | B=4，prompt=4096，词表=32768，decode=64 | 66.24 / 7.18 ms |
| 仅 stop 检查 | B=4，每行 512 token、3 个 stop | 80.57 / 23.96 ms |
| 仅重复惩罚控制（短 prompt） | B=1，prompt=16，词表=131072，decode=32 | 2.45 / 1.87 ms |
| 仅 stop 检查（短输出） | B=1，每行 128 token、3 个 stop | 3.34 / 2.66 ms |
| 随机小模型整段生成 | 2 层、hidden=64、词表=8192，prompt=256，decode=32，FP32 | 28.02 / 25.59 ms |

第一组 stop 的累计解码 token 数由 525312 降为 57768，第二组由 8256 降为 3306。
控制基准的 final_scores_equal / decisions_equal 均为 true。
小模型整段生成的输出 token 一致，词表投影位置数由 287 降为 32；该对照仅比较
完整 logits + 拼接缓冲区与最后位置 logits + 预留缓冲区，两边均使用当前几何 KV。
它不测本轮重复惩罚或 stop 优化，也不代表真实训练权重或 HTTP 服务。

上述记录可以这样复现：

```powershell
python tools/benchmark_generation_control.py --device cpu --batch-size 4 --prompt-tokens 4096 --vocab-size 32768 --decode-tokens 64 --stop-tokens 512 --repeats 3
python tools/benchmark_generation_control.py --device cpu --batch-size 1 --prompt-tokens 16 --vocab-size 131072 --decode-tokens 32 --stop-tokens 128 --repeats 3
python tools/benchmark_native_generation.py --device cpu --prompt-tokens 256 --decode-tokens 32 --vocab-size 8192 --repeats 3
```

该轮未补跑真实模型的 GPU 性能或混合长度 HTTP 压测，不能据此承诺 GPU tokens/s、
线上 TTFT、P95 延迟、并发容量或显存下降。上线验收仍需固定硬件、模型、精度和
真实请求分布，分别报告首 token 延迟、解码速度、吞吐、尾部延迟与峰值显存。
