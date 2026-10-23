# Saddle 自己的 Qwen3 推理后端

2026-09-28：通用 `native` 浮点加载入口已扩展兼容的 Qwen2/Llama 稠密适配器，
包括真实 DeepSeek-R1-Distill-Qwen-1.5B 验证；配置限制与国内完整版模型边界见
[多模型原生推理](NATIVE_MULTI_MODEL.md)。下文保留 Qwen3 专项与历史实验，不代表新模型都完成同等验证。

2.39 增加可选的 **W8A16 自有 CUDA/Triton 路线**：`--backend native --quantization int8`。
完整 8B 的同量化、同工作负载复测中，未合并投影 29.45 → 合并后 34.84 输出 token/s（+18.3%）。
这是有损量化的固定输入模型调用计时，不是 BF16 无损提速或 vLLM 对照。
使用、代码链路、质量/资源边界和最终包见 [完整 8B W8A16 指南](NATIVE_INT8_8B.md)。
下文 2.38 的浮点实现、测试和历史结果保留，不能与 2.39 量化成绩混算。

2.38 新增实验性的 `serve --backend native` 和 `load_native_causal_lm()`。
标准 Qwen3 safetensors 可以直接进入 `SaddleForCausalLM`：模型前向、增量 KV、采样、
生成循环和服务请求管理由 Saddle 实现，不构造 HF 模型、不调用 HF `model.generate()`。
Transformer 的矩阵计算仍使用 PyTorch 算子，tokenizer 可使用 Transformers；不要求用户编写 C++。

2.38/2.39 当时默认 `--backend local`；2.40 的 CLI/Python serve 默认 auto，
兼容环境可选中 vLLM。显式 local 仍是原生 Saddle 检查点用原生模型、普通 HF 模型用 HF。
选择 `native` 是明确的原生执行要求，不支持时直接报错，绝不暗中退回 HF。
`--backend vllm` 仍是另一条可选路径，见 [vLLM 服务指南](VLLM_SERVING_BACKEND.md)。

## 最简单的使用

在源码目录安装 `python -m pip install '.[serve]'`，或安装本轮本地 wheel 的 `[serve]` extra。
2.38 本轮尚未上传 PyPI，不能把旧仓库包的命令当成新版功能已经安装。

```bash
saddlellm serve /models/Qwen3-8B --backend native --dtype bfloat16 --local-files-only
```

仍提供 Web 对话页、`/v1/completions`、`/v1/chat/completions` 和流式 SSE。
只监听 `127.0.0.1:8000`，公网部署另配 TLS、访问控制与请求体限制。
远端模型 ID 也可用：`saddlellm serve Qwen/Qwen3-8B --backend native --dtype bfloat16`。
会先下载并验证配置，再按同一个不可变 snapshot revision 获取权重和 tokenizer，
不会先下载一个不支持的巨大模型再拒绝。`--local-files-only` 只使用已有目录/缓存。

**不要在当前共享服务器上直接照抄命令加载另一个 8B。** BF16 权重约 16.4 GB（十进制），
此外还有 KV、工作区、CUDA 上下文和单组投影合并的瞬时开销。加载器会检查 CUDA 的权重预算，
但这不是整个请求峰值显存的保证。原生后端只用一个设备，不偷偷 offload 到 CPU。

| 想控制什么 | 参数及效果 |
| --- | --- |
| 精度/设备 | `--dtype float32/float16/bfloat16`、`--device cpu/cuda/mps`；设备必须实际可用。 |
| 上下文/输出 | `--max-input-tokens`、`--max-new-tokens`、`--default-max-new-tokens`；实际还受模型上下文限制。 |
| 投影合并 | 原生 `serve` 对 Qwen3 BF16 默认只合并 gate/up，保留 QKV 独立计算；其他浮点路线仍默认合并两者。`--no-fuse-projections` 关闭全部合并，不重训、不改磁盘权重。 |
| 数值排错 | `--native-attention-backend eager` 选择数学参考路径；通常更慢，不作为加速开关。Qwen3 默认 SDPA，旧 Saddle 检查点默认保留自己的 attention backend。 |
| 固定合批 | `--batch-size 4` 合并符合条件的非流式贪心请求；不是连续批处理，采样/流式不自动进入此路径。 |
| 合批预算 | `--batch-token-budget` 限制 padding 后输入 + 输出槽位；`--max-queued-tokens` 限制队列预留。 |
| 前缀快照 | `--prefix-cache-mb 64` 按预算缓存完整 token 前缀的 KV/logits；默认关闭，不是答案缓存或分页共享。 |
| 思考模式 | `--enable-thinking` 交给 Qwen chat template；允许已有模型输出思考 token，通常更耗时，仍受输出预算限制。 |

GPU 至少能够容纳权重后，再按实际请求长度与并发验证 KV/峰值预算。
`--max-concurrency` 仍是本地并行 generate 许可，不是新增了一个连续调度器。
服务的鉴权、stop、EOS、真实 token 用量、取消与流式实现复用现有原生路径。
健康接口报告 `backend=native`、`execution_backend=native`、`streaming_backend=native_step`，
并给出实际投影合并、前缀缓存、批处理状态；不只展示操作者声明的后端名称。

## Python import

```python
import torch
from saddlellm import load_native_causal_lm, optimize_native_inference
from saddlellm.models.TokenizerLoader import load_tokenizer_compatible

path = "/models/Qwen3-8B"
model = load_native_causal_lm(path, device="cuda", dtype="bfloat16", local_files_only=True)
optimize_native_inference(model)  # 模型独占、开始请求之前；不要对训练 optimizer 原地执行
tokenizer = load_tokenizer_compatible(path, local_files_only=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "解释一下KV缓存。"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs = {name: value.to(model.embed_tokens.weight.device) for name, value in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, do_sample=False, max_new_tokens=128)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

公共 native loader 是推理入口，返回 eval/frozen 模型；不替换原有训练加载器。
不指定 dtype 或使用 `auto` 时，Qwen3 loader 采用 embedding 的存储精度；服务入口的 auto
仍沿用原来的设备精度策略，想严格比较 BF16 请明确指定。
现有 factory 同样可用：`create_inference_app(InferenceServerSettings(model_path=path, backend="native"))`。
模型可保存为 Saddle 原生格式，但部署导出目录也必须保存匹配的 tokenizer/chat template。

## 首版明确支持什么

支持 `model_type=qwen3`、`Qwen3ForCausalLM`、dense SwiGLU、GQA/MHA、Q/K RMSNorm、
全注意力、默认未缩放 RoPE、无 projection bias 的标准 float32/float16/bfloat16 safetensors。
支持单文件或完整分片 index、独立 head_dim、tied embeddings、多 EOS ID。
`generation_config.json` 的特殊 token ID 会读取；不偷偷继承其中全部采样/beam/logits processor 设置。
模型生成中的温度、top-p、重复惩罚、stop 等仍以 Saddle 的请求/生成参数为准。

Qwen3 专用加载器以下直接拒绝：Qwen3 MoE/其他架构、AWQ/GPTQ/FP8/bitsandbytes、独立 LoRA adapter、
缩放 RoPE（包括 YaRN）、滑动窗口混合、bias、自定义远端模型代码、缺失/多余/形状不符的权重。
adapter 需先合并成标准全量权重；想保留其他结构/量化，选择现有 HF 或已验收的外部引擎。
旧原生 Saddle 检查点仍支持原来的其他结构，这个限制针对新增的 **外部 Qwen3 适配**。
HF Qwen3 的 `.bin` pickle 不在这个新适配入口加载；原生 Saddle `.bin` 仍按既有 weights-only 加载。

## 实现逻辑：真正的原生执行

`cli.py` → `InferenceServerSettings(backend="native")` → `load_native_causal_lm()`
→ `NativeQwen3.load_qwen3_native()` → `SaddleForCausalLM` → `LocalGenerationEngine`
→ `NativeGenerationSession.prefill()/step()` → 原生 `InferenceLayerCache` → 文本/SSE。

关键点：

1. 先核对配置，限制真实计算语义；不是仅凭模型名字选择后端。
2. 在 meta 设备构造模型骨架，无完整随机 FP32 初始化。读取所有 safetensors header，
   严格检查名称、shape、dtype、index、分片缺失/重复和安全文件名后，再加载真实参数。
3. 权重逐 tensor 映射到目标设备/精度，不构造第二个完整 HF 模型或完整拼接 state_dict。
   tied head 与 embedding 共享同一 Parameter；重复保存的 tied 权重必须相同。
4. Qwen3 的 RMSNorm 先恢复激活精度再乘 weight；RoPE 常量与频率按 FP32 计算，
   后续 `.to(dtype)` 也重新保持 FP32 inverse frequency。新字段的默认值保留旧原生检查点行为。
5. `serve` 自动用现有原生优化器合并投影；prefill 只保留末位置词表投影，decode 使用请求私有
   几何增长 KV 和 token/mask 缓冲区。原生 stop、取消、采样和计数并不是另复制一套实现。

数学语义核对了官方 [Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)
和 [配置](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/configuration_qwen3.py)。
本机对应文件的 Git blob SHA 与官方 v4.57.6 分别匹配
`81b16c4ee6b6eb30acc531bf8cddd31d07065ac1`、`0b642913dce51c2b398bfbe0c25c4183821eb2f6`。

这仍不是分页 KV、连续批处理、分块 prefill、CUDA Graph 或新的 C++/CUDA 算子。
这些能力不能因新增 `native` 名称而声称已经实现。

## 正确性、数值差异与测速

新增离线测试直接保存 HF Qwen3 fixture，再从同一份 safetensors 加载 Saddle。
覆盖分片/tied、逐参数相等、prefill、cached decode、CPU eager/SDPA、CUDA FP16/BF16、
长位置 RoPE、原生保存重载、流式/usage、前缀命中、CLI 与禁止 HF fallback。
还覆盖损坏 header、重复 JSON、危险 shard 路径、错误 index、显存不足、元数据先验证与 revision pin。
这些随机小模型测试检验实现，不证明通用答题质量或完整 8B 性能。

CPU FP32 与 CUDA eager 数学参考检查通过。默认 SDPA 可以选择不同计算内核：
一个 tiny BF16 固定样例在同前缀第 4 步的 logits 最大差约 0.00232，HF 前两名仅差约 0.00195，
因而后续生成 token 不同；不是逐字无损保证。切换双方到 eager 后同前缀逐步 logits 完全相同。
测试没有删去这项差异：对默认 SDPA 做每步同前缀误差检查，测速工具保存所有生成 ID 与 top margin。

在真实 Qwen3-0.6B BF16 的 HTTP 对照中，默认合并 QKV 后，英文长回答与中文 KV 缓存回答
均出现了相对 HF 的分叉；只合并 gate/up 时，英文样例与 HF 一致，中文样例也避免了
“缓存模型参数”的错误说法。单独改用 eager 注意力不能消除这一差异。
因此服务加载真实 Qwen3 BF16 时自动跳过 QKV 打包，仍保留 gate/up 打包；
健康检查 `native_projection_packing` 会明确显示打包/跳过数量。
这不是通用质量等价证明，也不改变 INT8、FP32 或其他模型的打包策略。

2026-09-28 本机 RTX 4090 Laptop、真实 Qwen3-0.6B、CUDA BF16、关闭 thinking、
贪心、单请求 SSE、同一英文提示输出 96 token，三个服务交错复测 3 次的中位数：
HF local 为 34.46、native 全打包为 41.93、native 仅 gate/up 打包为 39.80 输出 token/s；
仅 gate/up 相对全打包约慢 5%，相对本轮 HF 约快 15%。该结果只代表这一小样本单流场景；
之前同机三轮在较慢的共同运行状态下分别观察到约 15 与 18 token/s，绝对速度存在明显波动。
由于全打包的答案已发生偏移，不能把它作为质量等价的加速结论。8B、并发、长上下文及任务质量尚未在本轮验证。

可复现工具：

```bash
python tools/benchmark_qwen3_native.py --model /models/Qwen3-8B --device cuda \
  --dtype bfloat16 --batch-size 1 --prompt-tokens 128 --decode-tokens 32 \
  --output artifacts/my_qwen3_native_comparison.json
```

工具需要同时驻留 HF 和 native 两份模型，CUDA 会先检查两份权重加工作区的最低预算。
**不是低显存 8B 方案**，不要在当前共享 GPU 上直接执行。
固定随机输入、关闭 EOS、无跨请求缓存，交错打乱两条路径测 7 次，计入完整 prefill + decode，
输出实际生成 token/中位耗时；不包括分词、HTTP、排队，也不叫线上 TTFT/吞吐。
logits 或输出不同默认非零退出并保存报告；`--observe-output-differences` 可做明确标注的观察性测试，
不是把输出差异改称正确性相等。报告/fixture 必须使用新路径，不覆盖既有证据。

HF 对照显式设置 `use_model_defaults=False`，避免较新版本把
`GenerationConfig(do_sample=False)` 重新填成 Qwen3 检查点的随机采样默认值。
两条路径必须采用同一生成策略，不能只看调用参数表面相同。

## 2026-09-27 最终验收与实际速度

### 小模型 GPU 对照：有收益，但不是 8B 成绩

同一份 **34,608,768 参数、未训练的 Qwen3 fixture**，Windows、RTX 4090 Laptop、
PyTorch 2.6.0+cu124、Transformers 4.57.6、Saddle 2.38。输入 128 token、输出 32 token，
预热 2 次、交错测量 7 次；双方同精度、同 attention 模式、贪心、EOS 关闭、无前缀复用。
native 启用了投影合并。这里的速率包括 prefill + decode，批量行的 token 相加，
不是 HTTP 并发吞吐，不是原有 28M 训练模型，也不是 Qwen3-8B 的速度代理。

| 精度 / attention / 批量 | HF 输出 token/s | Saddle 输出 token/s | 本次速率变化 | 数值及输出验收 |
| --- | ---: | ---: | ---: | --- |
| FP32 / SDPA / 1 | 156.64 | 204.45 | +30.5% | 8 步误差通过，所有生成 ID 一致 |
| FP32 / SDPA / 4 | 670.06 | 890.72 | +32.9% | 8 步误差通过，所有生成 ID 一致 |
| BF16 / eager / 1 | 126.07 | 185.69 | +47.3% | 8 步 logits 完全一致，生成 ID 一致 |
| BF16 / SDPA / 1 | 140.07 | 201.04 | **不作等价加速结论** | 生成 ID 一致，但 8 步 logits 门槛失败 |

BF16 SDPA 的最大 logits 差为 0.015625；保留原误差门槛，工具按预期退出 1。
eager 和 SDPA 不是同一种内核模式，不能跨行把绝对速度差解释成精度变化的收益。
上述百分比是这次测量的结果，受硬件、形状、系统负载影响，不是承诺值。

最终源码 SHA-256 与报告一致，原始输入哈希、权重哈希、每次耗时和全部输出均在以下文件：

- [FP32，批量 1](../artifacts/native_qwen3_20260927/cuda_fp32_b1_release.json)
- [FP32，批量 4](../artifacts/native_qwen3_20260927/cuda_fp32_b4_release.json)
- [BF16 eager](../artifacts/native_qwen3_20260927/cuda_bf16_eager_b1_release.json)
- [BF16 SDPA 失败记录](../artifacts/native_qwen3_20260927/cuda_bf16_sdpa_b1_release.json)

### 真实 Qwen3-8B：CPU 同权重校验通过，未证明 GPU 加速

在独立目录安装最终 wheel，而不是覆盖服务器业务环境。使用现有原始 Qwen3-8B 的
5 个 BF16 safetensors 分片，逐文件 SHA-256 再次匹配已有官方来源记录的 revision
`b968826d9c46dd6066d109eabc6255188de91218`；不是量化版或小模型替身。

本次仅用于验证执行链：CPU FP32、eager、单线程、批量 1、16 个固定随机输入 token、
8 个输出 token、预热 1 次、测量 1 次。原生参数数为 8,190,735,360，
36 层的 QKV 和 gate/up 均由 Saddle 优化器合并。
8 个同前缀步骤误差通过，最大 logits 差 **0.0000076294**；完整生成的 8 个 token ID 相同。

HF 约 0.657、native 约 0.654 输出 token/s，**基本持平，没有 CPU 加速收益结论**。
单次、单核心、短合成输入不是可靠的部署吞吐测试，更不是语义质量评测。
本轮没有做真实 8B BF16 CUDA、长上下文、HTTP/流式并发或答题质量验收。
此前 HF/vLLM/Ollama 的 GPU 数据不重新归属于这个 native 后端。

实验进程限制为 `CPUQuota=100%`、`MemoryMax=84G`、最长 600 秒，
`CUDA_VISIBLE_DEVICES` 为空；正常完成并退出，没有新增常驻推理服务。
业务 7 个 unit 全部 active，80/8000/8003 检查返回 200，GPU 显存仍为
35,632 MiB 已用、12,880 MiB 可用；没有停止或切换原有模型。

- [真实 8B 校验报告](../artifacts/native_qwen3_20260927/server/final/cpu_qwen3_8b_eager_verified.json)
- [服务状态核对](../artifacts/native_qwen3_20260927/server/services_final_check.json)
- [日志与原始证据归档](../artifacts/native_qwen3_20260927/server/native_validation_evidence.tgz)

首轮 `server/cpu_qwen3_8b_eager.json` 保留用于排错：其同前缀前向检查通过，
但 HF 默认值继承导致生成改为随机采样，**不属于有效的同策略输出/性能比较**。
修复脚本后才得到上述 verified 报告。早期 `*_verified.json`、`*_final.json` 和
`cuda_fp32_b1.json` 也保留，不覆盖；最终 GPU 表格只取 `*_release.json`。

### 回归、安装包与下一步

最终选定 21 个相关测试文件 **521 项通过、0 跳过、45 个依赖/弃用警告**，
其中 Qwen3 新增文件有 47 个测试案例；不是全仓库验收。
还修复了缺省 pad + 空 EOS 列表的配置边界，以及 wheel 检查脚本过时的 extras 清单。
新增清单一致性测试，避免可选后端已经声明、发布检查却误报。
新核心代码、测试与 benchmark 通过 E/F/I 检查，函数索引、版本一致性和实际 wheel 合约通过。
[JUnit 记录](../artifacts/native_qwen3_20260927/related_tests_release.xml)。

最终安装包是
[saddlellm-2.38-py3-none-any.whl](../artifacts/native_qwen3_20260927/wheel_release/saddlellm-2.38-py3-none-any.whl)，
SHA-256：`fee717e145528b6046c931ae70aa32a73e7cb98d625a6628f5b877d047637f8d`。
wheel 内运行时代码、服务器执行代码和上述最终报告的实现哈希已逐项核对。
`wheel/`、`wheel_final/` 是早期包；交付使用 `wheel_release/`，未上传 PyPI。

```powershell
python -m pip install 'artifacts/native_qwen3_20260927/wheel_release/saddlellm-2.38-py3-none-any.whl[serve]'
```

下一步先安排有独立显存预算的 8B BF16 CUDA 正确性及真实 HTTP 测试，
固定权重、提示、输出预算与并发，报告 TTFT、P95、实际输出 token/s 和显存，
确认低精度数值/质量后再决定 CUDA Graph 或专用算子是否值得实现。
当前生产部署继续使用已验证的默认 local 或按各自验收范围选择 vLLM；native 保持显式实验选项。
整个工作增强的是**模型执行与服务能力**，没有训练新的推理知识，也没有证明模型更聪明。
