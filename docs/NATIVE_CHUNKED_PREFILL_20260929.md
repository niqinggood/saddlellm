# 原生 token 预算调度与分块 prefill

本功能默认关闭。开启后，长提示分块建立 KV，已在输出的请求优先执行下一步。它主要解决长 prefill 阻塞其他请求的问题；分块本身可能增加长请求首 token 时间，不能理解为所有请求都更快。

## 开启方式

在已有原生服务命令上添加：

```text
--native-stream-batching --max-concurrency 4 --batch-size 4
--native-prefill-chunk-size 256 --native-max-step-tokens 256
```

完整 Linux 示例（端口需自行选择空闲端口）：

```bash
saddlellm serve /models/Qwen3-4B --backend native --device cuda --dtype bfloat16 \
  --local-files-only --host 127.0.0.1 --port 8123 \
  --native-stream-batching --max-concurrency 4 --batch-size 4 \
  --max-input-tokens 4096 --max-new-tokens 128 --batch-token-budget 20000 \
  --native-prefill-chunk-size 256 --native-max-step-tokens 256
```

`--native-fused-kernels` 是另一项独立的显式优化；本轮 L20 实测两边都开启它。上例是配置示意，不是对任意机器显存或并发容量的保证。

| 参数 | 含义 |
| --- | --- |
| `native_prefill_chunk_size` / `--native-prefill-chunk-size` | 每个请求一次最多计算多少 prompt token；0 关闭本轮功能 |
| `native_max_step_tokens` / `--native-max-step-tokens` | 调度轮中 decode + prefill 的总计算 token 预算；未设置时取 max(chunk_size, batch_size)，必须 ≥ batch_size |
| `batch_size` | 一轮最多选中的请求数；选中的工作可分多次 forward 执行 |
| `batch_token_budget` | 原有合批执行器的共享组预留槽位预算，不等于本轮计算预算，也不是全局 KV 显存硬上限 |
| `max_concurrency` | 驻留请求上限，仍配合已有准入/排队机制 |

`/health.stream_scheduling` 为 `native_token_budget`；`/health.native_stream_batching.scheduler` 提供开关、预算、调度轮数、计划 token、峰值轮预算、实际 prefill 分块数与 token 数。计划计数可能包含随后被取消的工作，不能直接当作实际吞吐。

## 执行与生命周期

聊天、补全、流式、非流式、greedy 与采样请求统一进入协调器。旧非流式固定微批在此模式下不再单独运行；流式按调用方消费速度提交下一步，不后台无限积攒输出。

每轮先选择已完成 prefill 的请求执行 decode，再用剩余预算推进 prompt。未完成的 prompt 放回队尾，并保留原始 Future，调用方只有在首 token 真正生成、取消或失败后才返回。不同 prompt 轮转推进；持续满载的 decode 仍可能延后 prompt，需要后续加入队列时限和公平性策略。

中间分块只建立 KV，跳过 `lm_head`；最后一块产生 logits，首 token 直接采样，不重复 forward。原有 `prefill()` / `step()` 也可以接续一个部分完成的 prompt。

取消在分块边界检查；正在执行的 CUDA 工作不会被强制中断。异常使该会话不可继续，工作线程仍可服务其他请求。退出时仍须等待当前工作完成，再释放会话和 KV。

## 支持边界与数值

首版支持单行、无 padding 的原生稠密 MHA/MQA/GQA、几何 KV 缓存。前缀缓存必须关闭；窗口 attention、MLA/KDA、MoE、AttnRes、原生 INT8 与自定义 forward 不在此路径内，明确拒绝，避免悄悄退回无限制整段 prefill。

仍使用独立连续 KV 和现有等上下文 decode 执行器，没有分页 KV、任意长度统一批处理或混合 attention kernel。采样请求可以共用调度器，但未因此获得批量采样执行优化。

分块会改变 GEMM/attention 的形状与低精度舍入。实际块大小还可能因同时就绪的 decode 数量而改变，因此不保证跨并发负载的逐 token 不变性。保留默认关闭；需要按目标模型和任务验证后启用。CPU FP32 测试检查注意力语义与 logits 容差；真实 BF16 的误差和输出检查另行记录。

## 验证与代码

本地 373 项相关回归通过；Linux 专项结果和 L20 / Qwen3-4B 数据见 [原始报告](../artifacts/native_engine_scheduler_20260929/REPORT.md)。碰撞实验测模型/调度器等待，不能用来宣称 HTTP 总吞吐或生产容量。

- `models/NativeGeneration.py`：有界 prefill、进度、可接续会话。
- `models/SaddleModeling.py`：内部 prefill-only 前向，保护训练/普通 logits 合约。
- `runtime/NativeTokenScheduler.py`：独立的 token 预算决策。
- `runtime/NativeStreamBatcher.py`：执行、重入队、取消和统计。
- `runtime/InferenceServer.py` 与 `cli.py`：统一路由与显式配置。
- `tests/test_native_chunked_prefill.py`：语义、顺序、预算、失败、引用释放和全部文本接口测试。

总体路线见 [原生引擎路线](NATIVE_ENGINE_ROADMAP.md)。
