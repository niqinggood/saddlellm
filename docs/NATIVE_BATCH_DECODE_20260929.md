# Saddle 原生合批解码：实现、实测与边界

最新行为见 [采样合批与显式吞吐模式](NATIVE_SAMPLING_THROUGHPUT_20260929.md)。本文保留初版 FP32 / greedy 实验及后续历史记录，采样已不再必然回退。

本轮优化的是 **Saddle 自己的模型执行层**，没有调用 vLLM/SGLang，没有改变权重或新增 C++ 编译依赖。
新增实验性 `NativeBatchDecodeRunner`：把多个兼容请求的下一 token 合成一次模型前向，并持续复用合批 KV。

初版交付范围是 **FP32、等上下文长度、greedy、稠密 MHA/MQA/GQA** 的显式 Python import API。
**没有接入默认 `saddlellm serve`，现有 8127–8130 服务没有切换或重启。**
初版 BF16 输出一致性未通过，曾与 FP16 一起回退。后续 [BF16 兼容修复](NATIVE_BF16_BATCH_FIX_20260929.md)
加入受限 CUDA BF16 逐行 Linear 路径并降低合组临时显存；FP16 与 CPU BF16 仍回退。本文原始实验/候选包记录保留，不覆盖历史结果。
再后续已接入 [显式原生 SSE 合批开关](NATIVE_STREAM_BATCHING_20260929.md)，并将组内行顺序稳定化以适应异步到达；
默认服务策略依旧不变。本文是初版执行器的历史记录，服务使用和新压测以该后续文档为准。

跨阶段的问题、改动、成果和下一轮优先级，见 [推理优化复盘与后续路线](INFERENCE_OPTIMIZATION_REVIEW_AND_ROADMAP.md)。

## 1. 改了哪一层，为什么能快

这里的旧基线是“多个独立 session 逐 token 交替执行”，不是框架此前完全没有批量计算。
已有服务支持显式的非流式固定微批；本轮没有做新执行器与该固定微批的直接性能比较。
新执行器的区别是以 step 为边界推进独立 session，并为兼容的活跃成员复用合组 KV；它仍不是完整的连续批处理服务。

原先逐 token 交替处理四个请求，仍要每轮执行四次 `model.forward([1, 1])`。
新执行器在兼容时改成一次 `model.forward([4, 1])`；减少重复的 Python 调度、算子发射及权重读取开销，
并让线性层有机会使用更有效率的批量计算。各因素分别贡献多少尚未做算子级性能归因。

流程如下：

1. 每个请求照常单独 prefill，保留自己的生成预算、停止条件、streamer 和 token 缓冲。
2. `runner.step(live_sessions)` 锁定请求；按已计算 KV 长度、设备和 dtype 分组。
3. 第一次合组，把各请求历史 KV 复制到 `[B, KV_heads, capacity, head_dim]` 的共享存储。
4. 请求各持有互不重叠的行视图；模型一次前向更新整组 KV。
5. 将每行 logits 交回原有 session，沿用各自的 repetition penalty、EOS、停止条件和输出逻辑。
6. 后续每步复用合组缓存，不逐 token 拼接全部历史 KV。容量不足时仍按原有几何增长方式扩容。
7. 请求结束/取消后，调用方移除并关闭它；成员变动时，兼容请求可以重新组队。不兼容则逐请求执行。

例如三个请求各生成 8 token：普通路径共 24 次 forward；测试中的新路径是 3 次单独 prefill + 7 次 B3 decode，
总计 10 次 forward。**这是实际批量前向，不是把串行工作移到多个线程。**

主要代码：

| 文件 | 职责 |
| --- | --- |
| [NativeBatchDecode.py](../saddlellm/models/NativeBatchDecode.py) | 兼容分组、合组 KV、行视图复用、异常失效、实验性执行器 |
| [NativeGeneration.py](../saddlellm/models/NativeGeneration.py) | `_step_locked()` 复用原有单请求生成逻辑；单步 API 仍自行加锁 |
| [InferenceCache.py](../saddlellm/models/InferenceCache.py) | 原有有界几何增长缓存，本轮直接复用 |
| [test_native_batch_decode.py](../tests/test_native_batch_decode.py) | token/logits 一致性、真实 forward 形状、加入退出、故障与回退 |

## 2. 同权重 A/B 实测

检查点：本地已有 `outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model`，约 28M 参数，
10 层、hidden 384、GQA。Windows，RTX 4090 Laptop 16 GB；已有服务仍在运行，**没有保证独占机器/GPU**。

四个不同但等长的合成输入，每个 32 token，固定生成 48 token，关闭 EOS、采样和前缀缓存。
两边都是同一个模型实例、同一精度和相同输入；基线交替执行四个独立 session，新路径执行合批。
计时包含 session 创建、全部 prefill/decode、采样、合组历史复制、结果复制与清理；不含模型加载和预热。
每轮各预热 2 次，然后 7 对交替 AB/BA。吞吐是四个请求的 **192 个输出 token / 整组耗时**。

| 环境 / 精度 | 第 1 轮：逐请求 → 合批 | 第 2 轮：逐请求 → 合批 | 配对耗时比中位数 |
| --- | --- | --- | --- |
| CPU / FP32 / Torch 1 线程 | 129.0 → 269.8 token/s | 134.5 → 281.5 token/s | 2.13× / 2.11× |
| GPU / FP32 | 150.3 → 574.5 token/s | 141.2 → 475.9 token/s | 3.70× / 3.37× |

四轮各有 7/7 对更快，全部生成 token ID 与各自基线完全一致。
GPU PyTorch 峰值 allocated 为 126,412,288 → 129,436,672 字节，增加约 2.4%；
这不是驱动层整卡显存，也没有计入其他进程。CPU 内存未测量。

表中的吞吐分别取各路线中位数；配对耗时比先按每对求比值再取中位数，两种统计不能直接互相相除代替。
第一轮 CPU 原始报告将未测内存记为 0，后续已修正为 null；这里不把 0 当作实际内存测量。

重要限制：

- 这是同权重、自身新旧执行方式对比，**不是 vLLM/Ollama 对比**。
- 是 28M 模型、B4、等长输入的执行层结果，**不是 8B，也不是 HTTP/SSE 服务吞吐或单用户速度**。
- 小模型更容易受主机调度/算子发射开销影响，不能把倍数外推到大模型。
- `first_round_seconds` 是所有请求首轮 token 计算完成耗时，不是客户端可见 TTFT。
- 7 个样本的 nearest-rank P95 就是该组最大值，不是有统计稳定性的线上 SLA。
- CPU 单请求负对照只有 3/7 对更快，配对比约 0.946×，P95 也未通过门槛，**不认定为单请求加速**。
- 不同长度、采样等工作负载尚无这条合批路径的吞吐收益证明。

原始记录保存在 [artifacts/native_batch_decode_20260929](../artifacts/native_batch_decode_20260929)：
`cpu_b4_run1.json`、`cpu_b4_run2.json`、`cuda_b4_run1.json`、`cuda_b4_run2.json`、`cpu_b1_control.json`。
记录含逐次耗时、输出 ID、合组统计、环境及当时源码 SHA256；保留失败记录，不覆盖原文件。

收窄低精度路由、补齐输入保护后的最终源码又独立复测了一轮：
CPU 136.9 → 291.4 token/s（配对比 2.12×），GPU 138.9 → 519.0 token/s（配对比 3.56×）。
两轮均 7/7 更快、输出一致且实际发生合批，完整开发门槛通过。
对应 [CPU 最终记录](../artifacts/native_batch_decode_20260929/cpu_b4_final.json)、
[GPU 最终记录](../artifacts/native_batch_decode_20260929/cuda_b4_final.json)。

## 3. BF16 的失败结果如何处理

在同一 28M 模型的 CUDA BF16 测试中，合批虽然更快，但部分输出发生分叉。
改成 `sdpa_reference` 仍未通过 token 一致性门槛，不能简单归因于 GQA folding。

进一步使用 teacher forcing，让两条路线每步都接收逐请求基线选出的 token：
首次 argmax 分歧出现在第 25 个生成 token、请求 0。基线 ID 6620，合批 ID 3367；
基线 top-2 差为 0.0625，该步最大绝对 logit 差也为 0.0625。
192 个请求步中有 3 个 argmax 不一致。

这证明相同历史输入下也有数值差异；**尚未定位到具体 GEMM/attention 算子，也不代表已证明模型质量下降**。
按当轮严格一致性门槛，初版只允许 FP32 进入合批路径，BF16/FP16 自动逐请求执行。
后续已定位所测案例的 Linear 形状差异并加入受限兼容路径，见 [修复记录](NATIVE_BF16_BATCH_FIX_20260929.md)；不等于裸 BF16 批量 Linear 已位级一致。
没有偷偷升精度、换权重或修改默认注意力后端来取得对比数字。

失败与诊断记录：`cuda_bf16_b4_run1.json`、`cuda_bf16_reference_b4_run1.json`、`bf16_teacher_forced.json`。
[诊断脚本](../tools/diagnose_native_batch_decode.py) 只用于实验，显式绕过低精度回退以复现问题，不更改服务。

## 4. Python 怎么用

需要项目推理依赖和一个本地兼容的原生检查点。以下示例不自动下载模型；实际 prompt token 长度不同会正常回退。
只有等 KV 长度请求碰到一起才合批，不能为了强行合批给单个请求随意加 padding。

```python
from contextlib import ExitStack

from saddlellm import NativeBatchDecodeRunner, load_native_causal_lm
from saddlellm.models.TokenizerLoader import load_tokenizer_compatible

path = "outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model"
device = "cpu"  # CPU 合批 FP32；CUDA 另有受限 BF16 兼容路径
model = load_native_causal_lm(path, device=device, dtype="float32", local_files_only=True)
tokenizer = load_tokenizer_compatible(path, local_files_only=True)
prompts = ["请介绍机器学习。", "请解释深度学习。"]

with ExitStack() as stack:
    sessions = [stack.enter_context(model.create_generation_session(
        tokenizer(text, return_tensors="pt")["input_ids"].to(device),
        max_new_tokens=48, do_sample=False,
    )) for text in prompts]
    runner = stack.enter_context(NativeBatchDecodeRunner(
        model, max_batch_size=4, max_batch_tokens=2048,
    ))
    while live := [session for session in sessions if not session.done]:
        runner.step(live)
    for session in sessions:
        ids = session.sequences()[0, session.prompt_tokens:].tolist()
        print(tokenizer.decode(ids, skip_special_tokens=True))
    print(runner.snapshot())
```

`step()` 返回与输入列表对齐的逐请求 `NativeGenerationStep`；`snapshot()` 的 `batched_forwards` 可检查是否真的合批。
`max_batch_size` 限制每次传入的 session 数量，不是后台等待队列。
`max_batch_tokens` 限制单组合批预留 token 槽位，**不是全局显存预算/自动防 OOM 保证**。
`cohort_history_copy_bytes` 只统计组建/重建时复制的历史，不包含几何扩容或算子内部复制。

调用方负责请求接入、排队、取消、输出和关闭。取消时在两步之间从列表移除并 `session.close()`；不能中断正在执行的 kernel。
共享存储的其他行仍有引用时，单行关闭不会立即释放整块存储；下一步会丢弃旧合组引用。
`runner.close()` 释放执行器引用，不代替调用方关闭 session；上面的 ExitStack 负责两者清理。
同一模型应保持只读 eval，不要跨线程/不同 CUDA stream 同时驱动这些 session，不要在执行中修改模型或缓存。

当 forward 或逐行输出回调失败，整组标记为失败，禁止从部分更新的 KV 继续解码；关闭相关 session 后重新创建请求。
多组一次 step 中途出错时，先前组可能已推进，调用方不能盲目重试整个列表。

## 5. 复现和验收

```powershell
python -m pytest -q tests/test_native_batch_decode.py tests/test_native_generation_session.py
python tools/benchmark_native_batch_decode.py --model outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model --device cpu --require-speedup --output artifacts/my_batch_cpu.json
python tools/benchmark_native_batch_decode.py --model outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model --device cuda --require-speedup --output artifacts/my_batch_cuda.json
```

输出路径必须是新文件。独立再运行一轮并换输出文件名；不要同时运行其他基准或测试污染时延。
开发门槛：实际执行合批、至少 7 对、token ID 全一致、配对中位数至少快 3%、至少 6/7 对更快、
P95/首轮耗时退化不超过 5%、GPU allocated 增长不超过 5%。CPU 内存未测，不作为通过证据。

研究原始低精度问题可加 `--dtype bfloat16 --probe-low-precision`；该参数仅限基准脚本，绕过后续 BF16 兼容保护，报告记录此覆盖。
`--attention-backend sdpa_reference` 可复查另一注意力路线。未通过时退出码非零，不能只截取吞吐数字。
没有研究开关时，合格 CUDA BF16 走后续兼容路径；FP16/CPU BF16 等不支持情况仍回退。

新增单测覆盖三种稠密注意力、三种内置后端、不同生成预算、repetition/EOS/stop、取消/加入、KV 扩容与视图复用、
前缀缓存不可变、外部单步使合组失效、锁释放、streamer 故障和低精度/MoE/窗口/采样等回退。
CI 已登记 Windows/Linux/macOS CPU 测试；这里只报告本地实际运行结果，不声称远端 CI 已通过。

### 本地验收结果

- 新执行器测试 39 项；包含相关服务、缓存、采样和原生模型的扩展回归共 **675 passed**，
  [报告](../artifacts/native_batch_decode_20260929/regression.xml)。109 条警告主要来自现有依赖弃用提示，不是零警告测试环境。
- 构建 wheel 后，逐个核对包内 Python 文件与工作区内容，一致；wheel 内容契约检查通过。
- 在源码目录之外、独立虚拟环境中安装该 wheel，原生合批/session/断连测试 **75 passed**，
  [报告](../artifacts/native_batch_decode_20260929/installed_regression.xml)。
- 独立安装包真实 CLI + TCP + OpenAI SDK 验证通过：聊天、补全、SSE、usage、资源释放及打包页面资源，
  [记录](../artifacts/native_batch_decode_20260929/installed-wheel-smoke.json)。这是原有服务接口回归，不表示服务启用了新合批器。
- 本文 Python 示例实际运行成功；README 和自动生成的函数索引已更新，函数索引检查通过。
- 现有 8127、8128、8129、8130 服务检查均 `ready=true`，未重启/升级这些服务。
- 新执行器相关源文件/测试的 Ruff F 检查通过。**全仓检查另有已有问题**：
  `_exports.py` 的 `RolloutSample` 分别声明了 alignment 与 runtime 两种导出，触发 F601。
  本轮保留它的既有绑定，未将此候选包视为全仓 lint/生产发布已完成。

候选包：[saddlellm-2.40-py3-none-any.whl](../artifacts/native_batch_decode_20260929/wheel/saddlellm-2.40-py3-none-any.whl)。
SHA256：`553e201348c19acc80e1dec87ca24eb7e2220fe8d29c83f816c7ed2a6b7da7cc`。
仅安装到本地隔离测试环境，没有上传 PyPI、推送提交或更新用户正在运行的服务。

## 6. 距离服务默认启用还差什么

这是有限的合批执行器，**不是完整的连续批处理、PagedAttention 或 vLLM 等价引擎**。
没有混合长度 ragged batch、分页 KV、调度公平性、分块 prefill、CUDA Graph、跨请求零拷贝 radix cache 或多卡 TP。

下一步先解决低精度数值差异，再验证真实聊天输入、多模型、多长度、加入/取消压力和显存上限。
之后才能接入原生 HTTP 调度器，做 streaming/non-streaming 的真实 TTFT、TPOT、P95、吞吐及质量验收。
保持外部 `saddlellm serve MODEL` 简单的前提，是内部调度路径先有证据；本轮不新增一个未经服务验证的 CLI 提速开关。
