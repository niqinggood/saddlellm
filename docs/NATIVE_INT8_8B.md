# Qwen3-8B：Saddle 自有 W8A16 推理、使用方法与验收

本指南对应 2.39 的实验性 `native + int8` 路线。目标是完整 8B 的实际提速和降低权重显存，
不是把小模型成绩当成 8B，也不是训练出更聪明的模型。2.39 当时默认 `local` 及浮点精度不变。
2.40 默认 auto，INT8 仍需显式选择；新部署与 HTTP 验收见 [OpenAI 服务手册](OPENAI_SERVING.md)。

## 1. 一条命令启动

在源码目录安装 `python -m pip install ".[serve]"`，或安装本轮提供的 wheel 与服务依赖。
另外需要可用的 NVIDIA CUDA、PyTorch 和与其兼容的 Triton；本功能不依赖 bitsandbytes。
本轮 wheel 尚未发布 PyPI，不能假定已有的旧版本包含此功能。

```bash
saddlellm serve /models/Qwen3-8B --backend native --device cuda --quantization int8
```

共享 GPU 上先使用短上下文、单请求。以下是本轮回环实验的保守配置，不覆盖业务端口：

```bash
saddlellm serve /models/Qwen3-8B --backend native --device cuda --dtype bfloat16 --quantization int8 --native-attention-backend eager --host 127.0.0.1 --port 8112 --local-files-only --max-input-tokens 1024 --default-max-new-tokens 128 --max-new-tokens 256 --no-web-ui
```

`/models/Qwen3-8B` 替换为完整的原始 Qwen3-8B safetensors 目录。Windows 使用实际目录路径即可。
`eager` 是本次数值验收和本 INT8 路线的默认 attention 路径，不是提速开关。
普通未量化的原生 Qwen3 默认 SDPA；INT8 也可显式选择 SDPA，但可能有浮点差异，
不能把本指南的 eager 吞吐直接当成 SDPA 成绩。

| 参数 | 使用者看到的效果 |
| --- | --- |
| `--backend native` | 模型前向、KV、生成循环使用 Saddle 实现；不支持的结构报错，不偷偷回退 HF。 |
| `--quantization int8` | 隐藏层权重压成 INT8；激活保持 BF16/FP16，不等于 `8bit` 的 bitsandbytes 路线。 |
| `--dtype bfloat16` | 激活、词嵌入、输出层等采用 BF16；不改变 INT8 权重，scale 保持 FP32。 |
| `--no-fuse-projections` | 关闭 QKV 和 gate/up 投影合并，用于同精度 A/B 对照。 |
| `--max-input-tokens` / `--max-new-tokens` | 限制实际输入和输出预算；请求输出超过上限会截到上限，非正整数报错；显存检查不保证任意长上下文都装得下。 |
| `--prefix-cache-mb 64` | 可选的原生前缀 KV 快照缓存，默认关闭；不是答案缓存或分页 KV。 |
| `--host 127.0.0.1` | 仅本机访问；公网部署还需要鉴权、TLS 和请求大小限制。 |

启动包含读取、合并、量化、转 GPU 和首次 JIT 编译。首次请求慢不代表稳态速度；先预热再计时。
访问 `/health` 可核对实际后端、`native_int8.method=saddle-w8a16`、
`kernel_backend=triton_w8a16`、实际 CUDA 层数和算子调用次数，而不是只看命令名称。

## 2. Python import 使用

```python
import torch
from saddlellm import load_native_causal_lm
from saddlellm.models.TokenizerLoader import load_tokenizer_compatible

path = "/models/Qwen3-8B"
model = load_native_causal_lm(
    path, device="cuda", dtype="bfloat16", quantization="int8",
    local_files_only=True,
)
tokenizer = load_tokenizer_compatible(path, local_files_only=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "请简要解释 KV 缓存。"}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
inputs = {key: value.to(model.embed_tokens.weight.device) for key, value in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

量化模型是冻结的推理模型，不能 `.train()`，也不能 `save_pretrained()` 冒充普通浮点检查点。
原始磁盘权重没有被改写。需要训练、LoRA 或导出时，重新加载原始浮点模型。
此入口暂不导出可重载的 W8A16 包；重启会从原始权重重新量化。

## 3. 为什么会快：代码链路

`cli → InferenceServer → load_native_causal_lm → 严格 Qwen3 加载 → CPU 投影合并`
`→ CPU 权重量化 → 显存预算检查 → 单个压缩模型上 GPU → 原生 prefill/decode → API/SSE`。

1. **先核对架构和 safetensors，再分配大模型。** 按配置构造 meta 骨架，逐 tensor 加载到 CPU，
   不创建额外的 HF 模型，也不先在 GPU 放一份完整 BF16 再压缩。
2. **减少投影启动。** 36 层 Qwen3-8B 的隐藏层投影由每层七次变成四次：
   Q/K/V 合为一次，gate/up 合为一次，保留 attention output 和 down；总数由 252 变为 144。
   没有删层、重训或改变 attention 结构。
3. **每个输出通道单独压缩权重。** `scale=max(abs(weight))/127`，
   `q=round(weight/scale)` 后保留 INT8；计算为 `y=scale*sum(x*q)`。
   零行有安全 scale，FP32 scale 不随 `.half()` 丢精度；词嵌入、lm_head、norm 不压缩。
4. **小批解码专用 GEMV。** 展平后 1–4 行输入用真实行数计算，不像实验 W8A8 路径为了 `_int_mm`
   把单行填充到 32 行；在 GPU 内部读取 INT8 权重，不生成完整的浮点反量化矩阵。
5. **较大输入用分块 GEMM。** 更多行的 prefill 走 Triton Tensor Core 矩阵计算，FP32 累加；
   输入行数保持运行时动态，避免每换一个提示长度就重新编译一套算子。
6. **复用原生生成优化。** prefill 仅计算末位置输出 head；decode 复用请求私有 KV/token/mask 缓冲，
   EOS、stop、采样、流式输出、真实用量和取消沿用原生实现；前缀缓存会检查权重/scale 变更。
7. **资源和状态保护。** 上 GPU 前后检查显存，默认保留 3 GiB；严格检查主机 RAM、有限数值、
   共享权重和 hooks。准备失败的模型被标记为不可服务，不静默切换量化算法或回退 HF。

这里新增了少量 **Triton GPU 算子**，不是完整重写 CUDA/C++ 推理引擎，也不需要用户编译 C++ 扩展。
已有 CUDA 多卡设备保护不等于已经实现 tensor parallel 或完成多卡性能验收。

| 文件 | 负责什么，读代码时抓哪里 |
| --- | --- |
| `saddlellm/models/ModelLoader.py` | `_load_native_qwen3_int8`：严格入口、CPU 准备、预算、单模型搬入 GPU。 |
| `saddlellm/models/NativeQwen3.py` | 配置语义和权重映射；拒绝不兼容架构，不让名字相同代替语义正确。 |
| `saddlellm/models/InferenceOptimizer.py` | 合并 QKV/gate-up、检查是否安全、报告真实合并数量。 |
| `saddlellm/models/CUDAInt8.py` | `prepare_cuda_int8`、`CUDAInt8Linear`：压缩权重、保持 FP32 scale、状态保护。 |
| `saddlellm/models/Int8WeightKernels.py` | `_matvec` / `_matmul`：解码与 prefill 两种 GPU 计算路径。 |
| `saddlellm/models/NativeGeneration.py` | 请求会话 prefill/step、增量 KV 与生成终止，不调用 HF generate。 |
| `saddlellm/models/NativePrefixCache.py` | 精确前缀复用与权重、scale、计算配置失效检查。 |
| `saddlellm/runtime/InferenceServer.py` | CLI 设置落地、请求预算、API/SSE 和实际后端健康状态。 |

面试可以这样讲：**同一份完整 8B，先用权重压缩降低带宽和驻留量，再用投影合并和解码专用算子
减少无效工作；模型调用吞吐与量化质量分开验收，不能把量化收益说成 BF16 无损收益。**

## 4. 完整 8B 实验与质量边界

测试为 Linux / NVIDIA L20、完整 8,190,735,360 参数 Qwen3-8B；BF16 激活、eager attention，
批量 1、输入 128 个固定种子随机 token、强制输出 64 token。不是自然文本或业务请求基准。
关闭 EOS 提前结束、前缀缓存及采样，HF 显式不继承
generation_config 的采样默认值。一次只放一个测试模型到 GPU，不停止既有业务服务。

测速采用两轮反转顺序，每轮每种实现预热两次并记录七次原始样本；报告以 14 次样本的中位数计算。
HF 对照也使用 **Saddle 的同一量化算法和 Triton 算子**，不是官方 HF/bitsandbytes/AWQ 性能。
最直接的框架收益是 Saddle 未合并投影与合并投影之间的同精度差异。

最终冻结版本的复测：

| 实现（同一套 W8A16 算子） | 中位模型调用耗时 | 输出 token/s |
| --- | --- | --- |
| HF Qwen3 架构 | 2353.27 ms | 27.20 |
| Saddle 自有模型，未合并投影 | 2173.37 ms | 29.45 |
| Saddle 自有模型，合并投影 | 1837.03 ms | 34.84 |

相对自己的未合并基线 **提升 18.3%**；相对同算子的 HF 架构对照提升 28.1%。
候选版 35.99 token/s 的报告也保留，但交付使用冻结版复测，不用较高的候选成绩替代。
这些是包含 prefill 的模型调用计时，不是 HTTP 吞吐，也不是纯 decode 每 token 速率。

八步同前缀 logits 检查和全部生成 ID 对照通过，说明同量化实现间的数值/输出验收通过；
**不说明 INT8 等于原始 BF16。** 另用八段自编文本、293 个预测位置抽查量化质量：
冻结版原始 CPU BF16 困惑度 20.4916，CUDA W8A16 20.6592，约上升 0.82%；top-1 一致率约 96.25%。
这是小样本、跨 CPU/GPU 的风险检查，不是标准数据集、通用能力或业务质量保证。

W8A16 常驻参数和 buffer 约 9.44 GB（约 8.79 GiB），对比原始 BF16 权重约 16.38 GB，
约减少 42%。不是所有参数都减半，也不是整个进程显存减少 42%；KV、CUDA 上下文和工作区另计。

没有在本轮相同条件下与 CUDA BF16 或 vLLM 做完整 8B 对照，因此不宣称超过它们。
失败路线也保留：BF16 CPU oneDNN 预排布未过数值/输出门槛，已禁用；W8A8 抽查困惑度约增加 21%，
未进入公开服务入口；TorchInductor 试验更慢，未提供面向使用者的编译加速开关。

## 5. 环境、限制和复现

支持完整 **dense Qwen3 的原始浮点 safetensors**；不支持 MoE、AWQ/GPTQ/FP8 预量化文件、
独立 adapter 或未支持的 RoPE/attention 结构。当前 INT8 入口只针对这一适配，不泛称所有 Saddle 模型可用。
需单个 NVIDIA CUDA 设备和可用 Triton；BF16 需设备具备相应能力，FP16 另需实际模型数值验证。
主机需要容纳原始浮点权重及量化准备工作区，通常建议至少 32 GB RAM 并检查实际可用量。
约 8.79 GiB 驻留量加默认 3 GiB 预留仍未涵盖所有 CUDA/运行开销，不能承诺任意 12 GB 卡可运行。

Linux L20 做完整 8B 实验，环境为 PyTorch 2.6.0+cu124、Transformers 4.57.6、Triton 3.2.0。
Windows CUDA/Triton 做了功能和算子回归，没有 Windows 完整 8B 吞吐结论。
CPU/MPS 的普通浮点路径保留，但本 W8A16 GPU 算子不适用于它们。首次 Triton JIT 有延迟，
生产前应测试自己的提示长度、并发、stop、长文本和质量；不要把配置里的最大上下文当成已测能力。
没有新增连续批调度、分页 KV、CUDA Graph、量化模型导出或 tensor parallel。

```bash
python tools/benchmark_qwen3_int8.py --model /models/Qwen3-8B --output artifacts/my_8b.json --activation-bits 16 --dtype bfloat16 --attention-backend eager --batch-size 1 --prompt-tokens 128 --decode-tokens 64 --correctness-steps 8 --warmup 2 --repeats 7 --rounds 2 --reserve-gpu-gib 3 --include-native-unfused --check-float-quality
```

报告保留各次耗时、生成 ID、误差、质量抽查、实际权重和实现 SHA256；输出文件存在时拒绝覆盖。
吞吐测试和 CPU 质量抽查较耗时，首次运行还需 JIT；不要在资源不足的业务机器盲目执行。
本轮相关回归为 570 项通过、57 个依赖/异步取消弃用等警告，并非声称全仓所有测试都执行。
原始实验证据和最终 wheel 位于 `artifacts/qwen3_8b_optimization_20260927/`。

## 6. 真实服务检查与交付

完整 8B 通过回环端口 8112 的实际 CLI 服务检查：中文聊天、重复请求输出/usage 一致、
文本补全、SSE 完成标记与 token usage、客户端断开后下一请求继续执行、前缀命中，
以及 `max_tokens=257` 被服务上限截为 256、`max_tokens=0` 被拒绝。
最终检查时实际 144 个 INT8 层全部位于 CUDA，Triton 算子计数增长；不是只测试 mock。

暖前缀的这一次 SSE 请求首个文本约 44.6 ms、64 个输出 token 约 1.93 s；
这是**单次功能检查、有前缀命中**，不是冷启动 TTFT、P95 或服务吞吐比较。
脚本早期对 null usage 和既有预算截断语义的误判报告也保留，未修改服务来迎合检查。
API/SSE 的完整响应和日志见下方记录。

实验结束已停止自己的临时服务，七个原业务服务仍为 active，Web/两个既有模型接口为 200；
GPU 回到实验前 35,632 MiB 已用、12,880 MiB 空闲。未替换生产模型或修改其依赖环境。

| 证据 | 内容 |
| --- | --- |
| [冻结版完整 8B 报告](../artifacts/qwen3_8b_optimization_20260927/server/w8a16-release/gpu_w8a16_b1_release.json) | 三种实现 14 次计时、八步误差、全部生成 ID、质量抽查、源代码与检查点哈希。 |
| [实际服务检查](../artifacts/qwen3_8b_optimization_20260927/server/w8a16-release/api_smoke_verified.json) | 聊天/补全/SSE/断开/预算、实际 CUDA 算子和前缀状态。 |
| [服务恢复状态](../artifacts/qwen3_8b_optimization_20260927/server/w8a16-release/server_final_state.json) | 业务存活、HTTP 状态、临时作业停止、资源和受测 wheel 哈希。 |
| [570 项相关测试](../artifacts/qwen3_8b_optimization_20260927/w8a16_related_tests_release.xml) | 本地 Windows CUDA 功能回归，0 失败/错误/跳过。 |
| [证据一致性核验](../artifacts/qwen3_8b_optimization_20260927/evidence_verification_final.json) | 本地运行时代码与 wheel、服务器执行实现哈希、原始五个分片与先前官方 provenance 一致。 |
| [含失败路线的完整记录](../artifacts/qwen3_8b_optimization_20260927/server/evidence_20260927_1813.tar.gz) | CPU/W8A8/编译/W8A16 候选与冻结版原始报告和日志，不挑掉失败。 |

交付 wheel：
[saddlellm-2.39-py3-none-any.whl](../artifacts/qwen3_8b_optimization_20260927/wheel_w8a16_release/saddlellm-2.39-py3-none-any.whl)。
SHA256：`7db811e9868d5320d7bd0e2a9fb7c47e67822ec90fdd13024a1c0a66360a897b`。
183 个包内 Python 文件与本地源码及服务器受测安装逐字节一致。
wheel 合约和版本一致性检查通过；README/本指南的实测说明在冻结后补充，未改变受测运行时代码。
不要把同目录的早期 CPU、W8A8 或候选 wheel 当作交付包。本轮未上传 PyPI。

```powershell
python -m pip install "artifacts/qwen3_8b_optimization_20260927/wheel_w8a16_release/saddlellm-2.39-py3-none-any.whl[serve]"
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
```

可用 `tools/smoke_native_int8_service.py` 重跑已有的回环测试服务（需配置与本轮相同的 256 输出上限及前缀缓存）。
它不创建或停止服务，报告拒绝覆盖。证据一致性检查可重跑：

```bash
python tools/verify_qwen3_int8_evidence.py --output artifacts/my_evidence_check.json
```

它不启动模型、不访问外网，核对的只是当前源码/安装包和保留实验记录一致，不代替业务质量验收。
