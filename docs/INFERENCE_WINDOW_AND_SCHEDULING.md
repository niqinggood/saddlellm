# 窗口 KV 与 token 预算调度（2026-09-27）

这是借鉴 MiMo 推理工程思路后落地的第一批改动：真正的窗口 KV 回收、
可选相近长度分桶、padding-aware 批预算、队列 token 背压和长请求防饥饿。
默认 CLI/import 入口不变，没有新增运行依赖或 C++/CUDA 编译要求。
这不是 MiMo 模型复现，也不是完整 SGLang/vLLM 引擎。

## 普通用户怎么用

已有原生模型只需沿用：

```powershell
saddlellm serve MODEL
```

如果模型某一层本来配置了 `sliding_window > 0`，默认几何缓存自动选择窗口环形存储。
全局注意力层不裁剪，KDA 层继续使用自己的递归状态。HF/custom 模型保留原生成路径。
不能为一个原本全局注意力的已训练模型随意启用窗口，并声称输出等价。

开启原生非流式贪心合批仍然只需：

```powershell
saddlellm serve MODEL --batch-size 4
```

客户端设置 `temperature=0`、`stream=false`。默认仍然严格等长分组。
相近长度合批和更明确的资源预算是可选项：

```powershell
saddlellm serve MODEL --batch-size 4 --batch-length-bucket-size 64 --batch-token-budget 4096 --max-queued-tokens 16384
```

| 参数 | 含义与默认值 |
| --- | --- |
| `--batch-size` | 每批请求数上限，默认 1；不保证达到该批量。 |
| `--batch-length-bucket-size` | 按实际 prompt token 数分桶，例如 64 表示 1–64、65–128 等；默认 0，保留严格等长行为。 |
| `--batch-token-budget` | 单批 `批行数 × (最大 prompt 宽度 + 最大输出预算)` 的上限，包含左 padding；默认输入上限加输出上限。 |
| `--max-queued-tokens` | 可选的待处理队列 `prompt + 输出预算` 预留总量上限；默认仅受已有 64 请求队列限制。执行中批次单独计数。 |

显式配置预算时，即使 batch_size=1，原生贪心 HTTP 路径也启用预算队列。
单请求超过批预算返回 400；队列数量或 token 预留满返回 503 和 `Retry-After: 1`。
流式、随机采样及 HF/custom 的原有逐请求路径不使用这个微批队列或其预算。
这些 token 数不是 GPU 字节硬上限：权重、工作区、KDA 状态、镜像窗口和并发批次也占内存。
建议先保留 `--max-concurrency 1`，再用实际模型测量。

Python 使用同样的执行核心，不需要启动服务：

```python
from saddlellm.runtime.InferenceServer import (
    CompletionRequest, InferenceServerSettings, LocalGenerationEngine,
    load_inference_model,
)

settings = InferenceServerSettings(
    model_path="MODEL", batch_size=4, batch_length_bucket_size=64,
    batch_token_budget=4096, max_pending_tokens=16384,
)
model, tokenizer = load_inference_model(settings)
engine = LocalGenerationEngine(model, tokenizer, settings)
results = engine.complete_batch([
    CompletionRequest(prompt="解释注意力机制", temperature=0, max_tokens=64),
    CompletionRequest(prompt="介绍 KV 缓存", temperature=0, max_tokens=64),
])
```

结果顺序与输入顺序一致，usage 使用各自真实 prompt 长度，不把新增左 padding 计入用量。
同步 import API 按批预算拆分，不使用 HTTP 队列等待、优先级或队列 token 上限。
重复惩罚不为 1、已有 padding/额外编码字段、自定义 forward/generate 等情况回退等长分组；
这样不会让新增 PAD token 改变重复惩罚，也不会猜测自定义模型的位置语义。
改变矩阵形状和低精度仍可能改变浮点舍入，应验证自己的模型输出。

## 窗口缓存的代码流程

1. [NativeGeneration.py](../saddlellm/models/NativeGeneration.py) 根据每层配置创建缓存。
2. [InferenceCache.py](../saddlellm/models/InferenceCache.py) 的 `length` 保存累计输入槽位数；
   `stored_length` 才是实际保留的历史长度，最多 W。
3. 单 token decode 使用镜像环形存储：每个新 KV 写入两份槽位，按时间顺序读取连续视图。
   不逐步 `cat` 完整历史，也不逐步移动 W 个旧 KV。物理缓存最多 **2W** 槽位，不是 W。
4. 多 token chunk 先用旧 W−1 个 KV 加上全部新 KV 计算，再在持久缓存里保留末尾 W 个。
   不能先只留 W 个再计算整个 chunk，否则早期 query 会丢失需要的上下文。
5. [SaddleModeling.py](../saddlellm/models/SaddleModeling.py) 按绝对 `key_start` 建立因果/窗口/
   文档掩码，query 的 RoPE 不随环形槽位重置；MLA latent 的可见 key 也按真实位置重建 RoPE。
6. 无 padding/文档限制的单 token 窗口 decode 不再生成冗余掩码，继续使用内置无 mask SDPA 路径。

训练与原有 tuple/dict 完整历史缓存接口保持兼容。
`cache_implementation="dynamic"` 仍可用于完整历史对照；它还有逐步拼接开销，
所以以下性能基准特意另用**完整历史几何缓存**作为对照，避免给基线额外加复制成本。

窗口模型当前不复用普通前缀快照。需要窗口尾部有效性验证的前缀树尚未实现；
[NativePrefixCache.py](../saddlellm/models/NativePrefixCache.py) 会安全跳过并清理不再可用的旧条目，
不会把 token 相同误判为各层所需 KV 都可用。

## 调度的代码流程

[InferenceServer.py](../saddlellm/runtime/InferenceServer.py) 分词并确定真实输入长度、有效输出预算，
生成严格等长或可选长度桶的兼容键；参数不同的请求仍不能合批。

[InferenceBatcher.py](../saddlellm/runtime/InferenceBatcher.py) 在请求数量/token 队列上限内接纳任务。
后续 HTTP 默认改为 `shortest_work`（输入加输出预算的粗略工作量），不只看短 prompt；
最老请求等待达到默认 1000 ms 后，在下个批次边界优先处理。
这个阈值不是延迟保证，也不会抢占正在执行的模型。Python settings 可设置 `batch_starvation_ms`。

挑选同兼容键请求时，同时检查 padded token 批预算；不合适的大请求留到后续批次。
取消待处理任务立即移除预留；取消正在计算的任务只设置其停止事件，直到实际模型线程结束
才释放执行许可和活动 token 统计。故障/关闭也沿用这套所有权管理。

`/health.batching` 可查看队列/活动 token、最大实际批 token、预算拒绝次数、调度策略和桶大小。
不返回提示词、token 内容或模型权重。

后续还落地了凑批提前唤醒、两个 SSE 入口的真实用量、首文本/P99 测量及部署探针；
具体使用与跨平台验证边界见 [跨平台部署指南](CROSS_PLATFORM_SERVING.md)。

## 独立 GPU 测量：缓存字节与耗时分开看

使用相同随机未训练权重、相同合成输入，RTX 4090 Laptop GPU，Torch 2.6.0+cu124，BF16。
4 层中 3 层窗口 W=128、1 层全局；B=1，输入 1024、输出 64，预热后交替测量 5 次。
测量进程未与本轮测试重叠。结果不代表真实训练模型质量、生产并发或 vLLM/SGLang 对比。

| 指标 | 完整历史几何缓存 | 窗口环形缓存 |
| --- | ---: | ---: |
| 持久 KV 张量字节 | 557,056 | 237,568 |
| 整段会话耗时中位数 | 458.33 ms | 442.15 ms |
| 额外 CUDA 分配峰值 | 5,786,624 B | 5,491,712 B |

这组持久 KV 减少 **57.35%**，整段耗时约 **1.04×** 加速。
不是总显存下降 57%，也不是长上下文理论容量收益直接变成同倍 tokens/s。
prefill 仍有完整投影、掩码和工作区，模型/token 缓冲区也未变为常数内存。
滑动窗口不同、全局层比例不同、真实请求分布不同，都需要重新测量。
分桶调度本轮以正确性和预算验收为主，未宣称已测得生产服务的 TTFT/P99 改善。

原始记录：[window_cache_20260927_bf16.json](../artifacts/window_cache_20260927_bf16.json)。复现：

```powershell
python tools/benchmark_window_cache.py --device cuda --dtype bfloat16 --prompt-tokens 1024 --decode-tokens 64 --window 128 --repeats 5
```

无 CUDA 可用 `--device cpu --dtype float32` 验证相同逻辑；不能直接比较不同设备的绝对速度。

## 已验证与尚未完成

专项覆盖窗口大小 1、多次环形回绕、超过窗口的大 chunk、GQA/MQA/MHA/MLA 两种缓存模式、
全局/窗口混合层、padding、打包文档位置、KDA 混合层、Full/Block AttnRes、训练梯度、
保存重载、CUDA FP16/BF16，以及分桶、实际 usage、停止、预算拒绝、取消和防饥饿。

```powershell
python -m pytest -q tests/test_sliding_window_cache.py tests/test_inference_token_scheduling.py
```

本轮新增 86 项专项用例，与既有选定回归同一次组合运行的最终结果为：
**759 passed，1 deselected，15 warnings，80.14 秒**。
这是选定范围，不是全仓库测试或大模型能力评测。唯一主动排除的是
`test_pinned_runtime_completes_real_post_training_smoke`（独立 HF/TRL 长链路真实后训练 smoke），
本轮不声称这条路径已经验收；原生 scratch、SFT/DPO、保存重载仍在实际测试范围内。
Megatron/SGLang 相关测试仍是配置/协议/替身检查，不是 Linux 多 GPU 真机验证。

完整选定回归可复现为：

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
  tests/test_public_api.py tests/test_packed_projection_checkpoint_edges.py `
  tests/test_runtime_backends.py tests/test_inference_optimizer.py `
  tests/test_serving_benchmark.py tests/test_gqa_decode_folding.py `
  tests/test_native_prefix_cache.py tests/test_deepseek_style_training.py `
  tests/test_model_composition.py tests/test_model_distributed_runtime.py `
  tests/test_scratch_pretrain_e2e.py tests/test_hybrid_attention_attnres.py `
  tests/test_hybrid_user_workflows.py tests/test_document_aware_packing.py `
  tests/test_quickstart.py tests/test_pretrain_quality_controls.py `
  tests/test_post_training.py tests/test_distributed_pretrain.py `
  tests/test_efficient_v2_preset.py tests/test_auto_parallel.py `
  tests/test_training_data_inspector.py tests/test_environment_diagnostics.py `
  tests/test_tokenizer_roundtrip.py tests/test_sliding_window_cache.py `
  tests/test_inference_token_scheduling.py `
  -k "not pinned_runtime_completes_real_post_training_smoke" --tb=short
```

尚未实现连续批处理、分页 KV、chunked-prefill 调度、窗口感知跨请求前缀复用、
CUDA Graph、生产级 MTP 缓存提交/回滚，以及完整的异步在线 RL 轨迹闭环。
下一阶段先测服务端混合长度的排队/TTFT/P99，再推进连续调度或 MTP，不能把本轮微批改进
表述为已经追平生产引擎。Megatron/SGLang 仍按需使用，不改变 Windows 默认依赖。
