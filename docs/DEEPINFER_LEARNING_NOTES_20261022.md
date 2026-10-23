# DeepInfer 推理工程学习笔记与 SaddleLLM 后续开发计划

记录日期：2026-10-22。当前状态：学习与规划，开发待后续触发。

DeepInfer 对 SaddleLLM 最有价值的部分，是固定缓冲区执行、分页 KV、CPU/GPU 流水调度和增量投机解码。适合先验证局部 CUDA Graph 原型，同时设计分页 KV 与变长 attention 的接口；完整 decode Graph 和异步流水线在执行状态稳定后推进。

上游阅读快照固定为 [`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9)，提交时间为 2026-10-21 UTC。下文的上游支持范围以此快照为准；SaddleLLM 状态对应记录日工作区，重新启动开发时先核对代码变化。

## DeepInfer 的支持范围

| 项目 | 阅读快照中的范围 | 对我们的含义 |
| --- | --- | --- |
| 模型 | DeepSeek-V4.1-Flash | 现有 Qwen、小稠密模型需要另外适配 |
| 硬件 | NVIDIA SM100、Huawei Ascend 950 | 4090 Laptop、L20 不在这版声明的支持范围内 |
| NVIDIA 构建环境 | Linux、Python 3.12、GCC 至少 13、CUDA 13 | 应在独立环境研究，不能直接套用现有 Windows/CUDA 12.x 环境 |
| 开源交付 | 离线推理及面向规模化服务的核心引擎能力 | 接入 HTTP、缓存服务和集群调度仍有工程工作 |
| 完整服务缺项 | 完整 KV 缓存系统、VE/Prefill/Decode 分离部署、跨 DP 副本负载均衡 | 单靠此仓库不能复现完整生产服务 |

这些范围来自 [上游 README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)。其中“完整 KV 缓存系统缺项”不表示仓库没有 GPU KV 实现：底层已经有 block 管理、block table 和分页读写相关代码，仍可学习其执行接口。

DeepSeek 完整版模型与 DeepSeek-R1-Distill-Qwen 等蒸馏模型采用不同结构。我们已经适配蒸馏版，不能据此推导为支持 DeepInfer 当前目标模型。

## 我们已经具备什么

| 层次 | SaddleLLM 当前基础 | 后续主要缺口 | 代码入口 |
| --- | --- | --- | --- |
| 生成会话 | 独立 prefill、逐 token step、采样、EOS、停止与流式交付 | 可供 Graph/异步流水线使用的稳定设备状态和执行接口 | [NativeGeneration.py](../saddlellm/models/NativeGeneration.py) |
| KV 存储 | 几何增长缓存、模型原有窗口对应的环形缓存 | 分页池、block table、按实际显存准入、页级共享 | [InferenceCache.py](../saddlellm/models/InferenceCache.py) |
| decode 合批 | 等长度共享 KV cohort；受限的不同长度请求可共享投影，attention 逐请求执行 | 真正的变长批量 attention，减少合组及扩容复制 | [NativeBatchDecode.py](../saddlellm/models/NativeBatchDecode.py) |
| 数值模式 | `compatible` 在受保护的 CUDA BF16 路径保留 B1 形状 Linear；`throughput` 允许批量 Linear | 跨 batch 形状的数值验证和更高效的兼容执行 | [NativeBatchDecode.py](../saddlellm/models/NativeBatchDecode.py) |
| 调度 | 后台流式合批、decode 优先、token 预算、chunked prefill | CPU 调度与 GPU 执行的跨步重叠；成本感知预算 | [NativeStreamBatcher.py](../saddlellm/runtime/NativeStreamBatcher.py)、[NativeTokenScheduler.py](../saddlellm/runtime/NativeTokenScheduler.py) |
| 前缀缓存 | 带预算、LRU 和失效检查的快照；命中后恢复请求私有 KV | 页级共享与 copy-on-write，以及与分块 prefill 的组合 | [NativePrefixCache.py](../saddlellm/models/NativePrefixCache.py) |
| 局部融合 | 可选 RMSNorm/RoPE decode 内核及预热 | 完整解码执行图、更多融合机会及相应验证 | [NativeFusedKernels.py](../saddlellm/models/NativeFusedKernels.py) |
| 投机解码 | 正确性参考实现，使用全上下文 forward 和 CPU float64 概率处理 | 增量 KV、验证批次、拒绝回滚、服务接入 | [lightweight.py](../saddlellm/compression/lightweight.py) 的 `speculative_generate` |

采样进入合批、不同 KV 长度的投影合批、合批 token 回传等能力已经实现。历史调研中的“采样完全不能合批”“不同长度只能全部独立执行”属于旧状态，后续不应重复立项。更新过程见 [采样与混合长度合批](NATIVE_SAMPLING_THROUGHPUT_20260929.md) 和 [2026-10-02 推理优化](INFERENCE_SPEED_20261002.md)。

## CUDA Graph 需要稳定的执行状态

普通 eager decode 每步都会从 Python 调用模型并发起一系列 GPU 操作。CUDA Graph 捕获一段执行，再用较少的 CPU 提交开销回放，适合验证小模型、低并发和单 token decode 的启动开销。

上游 [CudaGraphRunner](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/cuda_graph_runner.py) 值得按以下顺序阅读：

1. `build_graph_batch_sizes` 构造捕获的 batch 桶。
2. `capture` 预先分配输入 metadata、设备缓冲和输出，预热后捕获各个桶。
3. `select` 选择能覆盖当前请求数的桶。
4. `replay` 回放图并返回预分配输出。
5. `destroy` 释放图及其持有的缓冲。

对我们而言，关键不只是 KV 张量地址固定。**Python 中的缓存长度、切片宽度、位置与分支，在捕获后可能被固定下来。** 后续回放必须实际读取新 token、写入正确 KV 槽位，并让 attention 看到正确的历史长度。只在同一个上下文长度反复回放，不能证明支持连续生成。

建议先选标准稠密模型，固定权重和 dtype，对局部投影或受控 decode 做原型，测清捕获成本、额外显存和稳定执行收益。完整 B1 decode 还需选择明确的动态长度方案，例如设备端长度 metadata 配合可支持它的 attention，或经过验证的长度分桶及 mask 方案。

验收应逐步改变 token 和位置，跨越 KV 扩容或长度桶边界，比较 eager 与回放的 KV、logits、生成结果和 EOS。若当前缓存接口不能可靠表达这些变化，先完善接口，再扩大 Graph 覆盖范围。

局部 Graph 原型可以提前探索；长期完整 decode Graph 仍遵循 [原生引擎路线](NATIVE_ENGINE_ROADMAP.md) 中先稳定缓存与批执行接口的依赖关系。

## 分页 KV 要与变长 attention 一起设计

当前连续 KV 通过几何增长减少逐步拼接，但扩容仍需要搬运历史。等长请求共享 cohort 时，请求成员变化也可能触发复制。不同长度请求虽然能合批投影，attention 仍需分别处理各自历史。

分页 KV 将历史切成固定容量的页，每个请求通过 block table 描述逻辑历史所在的物理页。上游 [BlockManager](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/block_manager.hpp) 展示了空闲页管理、引用计数、分配/释放以及资源不足时等待或无法分配的区分。

适合借鉴的开发拆分是：

| 部分 | 要明确的行为 |
| --- | --- |
| 页池 | 页尺寸、真实字节预算、分配/回收、统计与所有权 |
| 请求 metadata | 每请求 block table、有效长度、绝对位置、当前写入槽位 |
| attention | 直接读取 block table，正确处理不同长度、因果关系和尾页 |
| 共享前缀 | 引用计数、尾页写入隔离、copy-on-write、模型变更失效 |
| 显存准入 | 同时考虑权重、KV、Graph 池、工作区和其他状态的预算 |
| 生命周期 | 取消、异常和关闭时，等待实际 GPU 使用结束后再回收页 |

第一版先服务单模型稠密 MHA/GQA。上游模型和内核有其专用结构及硬件条件，移植 block 管理不等于直接获得可用于 Qwen 的分页 attention。

需要同时验证 allocator 与 attention：页边界、混合长度、请求进出、共享页写入、OOM 前背压和故障回收。只有页分配器通过测试，仍不能宣称完成 PagedAttention。

## 异步调度要重叠 CPU 和 GPU 工作

当前 `NativeStreamBatcher` 将请求集中到后台线程，执行一轮 runner 后交付结果，再推进下一轮。合批已减少部分采样与回传开销，但还没有建立上游描述的跨步流水线。

串行执行中的依赖可简化为：

```text
CPU 排批 N → GPU forward/sample N → CPU 获取结果和后处理 N → CPU 排批 N+1
```

上游让 GPU 执行当前步时，CPU 准备后续步并处理较早一步的结果，以减少 GPU 等待 CPU 的空档。参见 [异步调度说明](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#asynchronous-scheduling-zero-overhead-batch-scheduler)，再读 [zos.cpp](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/zos.cpp) 和 [model_runner.py](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/model_runner.py)。

我们应先用时间线确认 CPU 排批、设备到主机回传和后处理是否造成明显空档。若瓶颈主要在 GPU 计算，增加流水线状态管理未必划算。

实现时要明确双缓冲、CUDA event、设备 token 到下一步输入的连接，以及结果尚未返回时的请求状态。EOS、取消和停止必须限制后续计算与交付；即使流水线已有多余计算，也不能把 EOS 后 token 发给用户或计入 usage。KV 和输出缓冲的回收要以实际使用完成为界。

第一版可先限制为标准稠密 greedy 路径，再扩展到采样和服务；自定义 streamer、停止回调等保留已有执行路径，直到其跨步语义经过验证。

## 投机解码先补增量执行

投机解码让便宜的草稿模型先提出多个 token，再由目标模型验证，期望一次目标模型调用交付多个有效 token。收益取决于草稿成本、接受长度、验证成本、额外 KV 和调度开销。

我们当前 `speculative_generate` 已有 greedy 和拒绝采样的参考逻辑，但使用全上下文重算，不能据此推断已有高效的服务执行路径。后续开发可以保留它作为语义对照，逐步补齐：

1. 草稿与目标模型独立的增量 KV 和 token 位置。
2. 一次验证多个草稿 token 的前向路径。
3. 拒绝位置之后的 KV 截断或回滚，以及全部接受时的 bonus token。
4. EOS、输出预算、取消、每请求 RNG 和服务交付的一致处理。
5. 在相同采样策略下，与目标模型单独生成比较实际总耗时。

上游 DSpark 更进一步：用 Confidence Head 预测接受概率，结合硬件吞吐曲线分配每个请求的验证长度。这依赖配套模型能力，普通 Qwen 权重没有同一套现成的头。应在增量固定长度验证稳定后，再判断是否值得训练或适配置信度预测。参见 [投机解码说明](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#speculative-decoding) 和 [变长验证内核](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py)。

## 数值一致性和采样复现要分别验证

改变 batch 形状可能改变 GEMM、softmax 或归约的浮点舍入；即使 logits 完全相同，共享 RNG 的消费顺序变化也可能改变采样。固定 seed 不能单独解决前一个问题。

上游提供 batch-invariant 模式；README 中的确定性采样配置还要求关闭变长 MTP 验证，并给每个序列固定 `seq_seed`，支持范围为 NVIDIA。配置条件见 [README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)。

其 [batch-invariant 测试](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/tests/models/test_batch_invariant.py) 先合批生成，再以 `golden_token_ids` 固定历史逐条回放，对比 token 和 logprob 等结果。这个思路适合隔离数值差异，但仍需另外测试自由生成与独立请求 RNG。

对我们的 `compatible` 和 `throughput` 模式，建议分别保存同历史 logits 误差、argmax 分歧、自由生成差异及任务质量结果，再比较延迟和吞吐。`compatible` 是执行策略名称，并不自动保证任意模型、内核或调度下逐 token 恒等。

## 结构化输出和窗口重放的用途

结构化输出通过语法状态生成合法 token mask，可约束 JSON schema、正则和 EBNF。它对决策模型、工具参数输出可能有产品价值，但应单独衡量编译、mask 和状态推进开销。上游将编译与运行期分开，例子见 [structured output](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/_structured_output.py)。

SWA bounded replay 利用目标模型的窗口结构，只重放必要后缀。它不等于给普通全局 attention 模型任意截断历史；上游保留长度还需覆盖窗口与最大回退距离。我们可研究模型原有窗口与回滚缓存的组合，不能直接把其默认的 140 token 配置套到 Qwen。条件见 [README](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md)。

## 后续开发顺序

| 阶段 | 开发内容 | 完成标准 |
| --- | --- | --- |
| P0 | 核对当前能力，保存版本、源码和基线，重新 profile 代表模型 | 明确 CPU、launch、GEMM、attention、KV 复制的主要成本；保留可复跑 A/B |
| P1 | 局部 CUDA Graph 原型；同步明确分页 KV 与变长 attention 接口 | 原型能验证动态输入和状态正确性；局部收益按局部口径报告 |
| P2 | 页池、block table、显存准入及单模型稠密分页 attention | 混合长度请求能共同执行 attention；页边界、请求进出、取消和回收正确 |
| P3 | 稳定缓存上的完整 decode Graph，逐步扩展 batch 桶 | 连续生成跨长度/页边界正确；实际端到端收益覆盖额外显存和 padding 成本 |
| P4 | CPU/GPU 流水调度 | 时间线确认重叠；EOS、取消、停止、慢客户端和缓冲寿命满足已有合约 |
| P5 | 增量投机解码，随后评估自适应验证长度 | 实际接受及交付 token 的收益超过草稿、验证和回滚成本 |

数值验证贯穿各阶段。结构化输出可按业务需求单独启动；Wide-EP、EPLB、跨机 Prefill/Decode 分离等投入，留到模型和集群规模确实需要时再评估。

## 性能验收口径

已有 [推理验证方案](INFERENCE_VALIDATION_PLAN.md) 和 [远程 L20 对照](REMOTE_INFERENCE_COMPARISON_20261002.md) 可作为模板。每阶段固定权重、dtype、设备、输入/输出工作量、采样和缓存设置，使用同环境交替 A/B；先测单请求，再测等长并发和长短混合负载。

| 方面 | 应保存的结果 |
| --- | --- |
| 延迟 | 模型计算时间、服务首个可见文本、完整响应及 P95；具备真实逐 token 时间戳时再报告 ITL |
| 吞吐 | 实际完成并交付的输出 token、测量 wall time、失败数；不计 padding 槽和被拒绝草稿 |
| 数值与质量 | 同历史 logits、argmax、生成结果和相应任务质量 |
| 内存 | allocated/reserved 峰值、页池/Graph 池/工作区、请求结束及服务关闭后的状态 |
| 生命周期 | 请求加入退出、EOS、stop、取消、断连、异常、关闭，以及许可和缓存释放 |
| 重现信息 | 环境、模型标识、源码快照/哈希、启动参数、基准命令及原始逐请求结果 |

特别注意上游 [benchmark 说明](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/README.md)：其中 decode throughput 使用 `avg_SPS × decode_num_requests × (1 + nextn) / cp` 的全部草稿接受口径，不随真实接受率变化；部分 sweep 绘图另按给定接受率缩放。引用这些数值时要保留计算口径，不能当作我们实际在线交付的 token/s。

SaddleLLM 已有可复用入口：

| 用途 | 文件 |
| --- | --- |
| decode 热点定位 | [profile_native_decode.py](../tools/profile_native_decode.py) |
| 流式合批和调度 A/B | [benchmark_native_stream_batching.py](../tools/benchmark_native_stream_batching.py)、[benchmark_native_stream_scheduling.py](../tools/benchmark_native_stream_scheduling.py) |
| 服务交付与跨执行器对照 | [benchmark_serving.py](../tools/benchmark_serving.py)、[benchmark_inference_compare.py](../tools/benchmark_inference_compare.py) |
| 原生合批与采样回归 | [test_native_batch_decode.py](../tests/test_native_batch_decode.py)、[test_native_sampling_batch.py](../tests/test_native_sampling_batch.py) |
| 调度与生命周期回归入口 | [test_native_stream_batching.py](../tests/test_native_stream_batching.py)、[test_inference_token_scheduling.py](../tests/test_inference_token_scheduling.py) |
| 前缀及投机语义参考 | [test_native_prefix_cache.py](../tests/test_native_prefix_cache.py)、[test_speculative_generation.py](../tests/compression/test_speculative_generation.py) |

这些入口覆盖已有行为；分页、Graph 和流水线还需要补充相应的动态状态与集成验证。

## 以后启动开发时可以这样提需求

先做最小实验：

> 请阅读 docs/DEEPINFER_LEARNING_NOTES_20261022.md，核对当前 SaddleLLM 推理实现和文中上游快照。先完成 P0 基线与热点定位，再做 P1 局部 CUDA Graph 原型。固定权重、dtype 和生成策略，保留同环境 A/B、动态 token/位置/KV 验证及原始结果。根据结果说明是否具备扩大到完整 decode 的条件，未达性能门槛的路径保留实验状态。

推进并发引擎：

> 请依据 docs/DEEPINFER_LEARNING_NOTES_20261022.md，从当前源码设计并实现单模型稠密 MHA/GQA 的分页 KV、按显存准入和变长批量 attention。沿用现有生成、停止、取消和服务接口，明确页所有权及回收时机，完成混合长度、页边界、请求进出与真实 HTTP 验证，再推进完整 decode CUDA Graph。

开始时一并阅读 [原生引擎路线](NATIVE_ENGINE_ROADMAP.md)、[最近推理优化](INFERENCE_SPEED_20261002.md) 和 [采样与混合长度实现记录](NATIVE_SAMPLING_THROUGHPUT_20260929.md)，以当前源码和最新验证结果调整范围。
