# L20 上的 Qwen3-4B 推理部署比较（2026-10-02）

本轮将候选 SaddleLLM 2.40 安装包部署到用户指定服务器的独立目录，比较原生引擎、Transformers 和 vLLM。已有业务服务继续运行。本文记录整个部署路线的表现；本轮代码改动的独立 A/B 收益见 [本地优化报告](INFERENCE_SPEED_20261002.md)，两者不能混为同一指标。

## 环境与实验约定

- Ubuntu 22.04.5、Xeon Gold 6462C、16 vCPU、约 132 GB RAM；NVIDIA L20，驱动报告显存 49,140 MiB。
- 原有五个 GPU 进程常驻，测试前显存空闲约 11,558 MiB。一次只启动一个实验引擎；不是独占 GPU 测试。
- 所有引擎使用服务器已有的同一套 Qwen3-4B BF16 权重和 tokenizer。权重与 tokenizer 文件逐一计算 SHA256，见 [模型清单](../artifacts/inference_speed_20261002/remote_evidence/prepared/manifest.json)。
- 统一离线套用原始 chat template，`enable_thinking=False`，传入 raw prompt；greedy、`temperature=0`、`top_p=1`、最多输出 64 token，保留 EOS。
- 36 个不同提示用于 2 轮预热与 7 轮正式测量，每轮 4 个请求。每个引擎使用完全相同的顺序；在用户消息前部放置等 token 长度的不同标识，避免反复测相同提示的长前缀命中。
- C1 为客户端逐个请求，C4 为四个闭环客户端并发；每种配置有 28 个正式样本。并发吞吐是所有请求合计输出 token/s，不是单个用户的生成速度。
- 首字指客户端收到首个非空文本块，含 tokenizer 缓冲；文本块间隔不是逐 token 延迟。热模型测量不包含下载、转换、加载、编译时间。
- 原生与 Transformers 使用 PyTorch 2.6.0+cu124、Transformers 4.57.6；vLLM 0.30.0 使用其独立环境的 PyTorch 2.13.0、Transformers 5.17.0。依赖不同，不能把结果解释为仅一个内核或编程语言的差别。
- 原生与 Transformers 的 Torch allocator 上限设为整卡的 0.21，加载后释放一次未使用缓存；vLLM 设置 `gpu_memory_utilization=0.21`。这两种预算机制不完全等价。启动后检查空闲显存，正式测量前至少保留 1,024 MiB。
- 全卡显存每 0.5 秒采样，含其他业务进程，不是引擎独立峰值或 Torch allocated 峰值。

## 引擎配置

原生兼容模式使用 `--performance throughput`，批大小与并发均为 4，流式合批开启，数值模式保持 `compatible`。显式关闭前缀缓存。另测 `--native-decode-mode throughput`，允许批量线性计算产生数值差异；此选项不会被性能预设自动开启。

Transformers 对照使用 Saddle 的 `--backend local`，生成许可为 1，因此 C4 包含排队。这是当前串行 Transformers 服务基线，不能代表所有经过批处理优化的 Transformers 部署。

vLLM 直连与 Saddle `--backend vllm` 使用相同的 `max_model_len=1024`、`max_num_seqs=4`、`max_num_batched_tokens=256`，显式关闭前缀缓存，基础比较开启 `enforce_eager`。Saddle 保留相同简单入口，实际生成由 vLLM 执行；不能将 vLLM 的速度归为自研原生内核的速度。

## 已完成的基础比较

| 部署路线 | C1 输出 token/s | C4 合计输出 token/s | C1 P95 首字 | C4 P95 首字 |
|---|---:|---:|---:|---:|
| Transformers，经 Saddle 串行服务 | 54.13 | 54.34 | 104.19 ms | 3572.19 ms |
| Saddle 原生，compatible | 49.67 | 78.21 | 111.11 ms | 341.94 ms |
| Saddle 原生，显式 throughput 数值模式 | 未测 | 127.61 | 未测 | 264.68 ms |
| vLLM 直连，eager | 77.33 | 282.59 | 41.10 ms | 85.80 ms |
| Saddle + vLLM，eager | 77.42 | 274.37 | 40.99 ms | 87.66 ms |

所有上述配置均实际输出 28 × 64 = 1,792 token。Saddle vLLM 与直连在所测场景表现接近；约 2.9% 的 C4 总速率差来自顺序测量，不能精确归因为适配层固定开销。原生吞吐模式比兼容模式快，但二者存在允许的数值路径差别，不能称为不改变计算路径的增量优化。

原生兼容模式 C1 与 Transformers C1 的 28 个输出文本完全相同；C4 比较有 24/28 相同，其余请求输入/输出 token 数仍相同。`compatible` 在这个完整模型、并发运行中并不构成逐 token 恒等保证。不同文本需要任务评测才能判断质量，当前速度实验没有给出质量优劣结论。

vLLM 直连与 Saddle vLLM 的 C1 文本为 28/28 相同，C4 为 21/28 相同；原生 throughput 与 compatible 的 C4 文本为 21/28 相同。所有对应输入/输出 token 数一致。不能仅由这些记录把文本差异归因到某个调度或计算组件。

原始证据目录：

- [Transformers 串行](../artifacts/inference_speed_20261002/remote_evidence/results/local_hf_budgeted/)
- [原生兼容](../artifacts/inference_speed_20261002/remote_evidence/results/native_compatible_budgeted/)
- [原生吞吐](../artifacts/inference_speed_20261002/remote_evidence/results/native_throughput_budgeted/)
- [vLLM 直连 eager](../artifacts/inference_speed_20261002/remote_evidence/results/direct_vllm_budgeted_retry1/)
- [Saddle vLLM eager](../artifacts/inference_speed_20261002/remote_evidence/results/framework_vllm_budgeted_retry1/)

### Graph 与编译联合配置

进一步取消 `enforce_eager`，其余显存、上下文、并发与提示配置保持一致。运行日志确认同时启用 `torch.compile` 和 CUDA Graph；这不是隔离测量 Graph 一项的收益。

对齐 `cudagraph_capture_sizes=[1,2,4,8]` 后，直连 vLLM 的 C4 合计吞吐为 **326.07 token/s**，Saddle vLLM 为 **325.48 token/s**。后者 P95 首字 85.83 ms，P95 完整响应 786.12 ms。两者表现接近；不能将小于 1% 的差异解释为确定的适配器开销。

另保留一组直连 `[1,2,4]` 的附加试验，C4 为 325.82 token/s；它的 capture 配置与框架默认值不同，不作为严格对齐的主比较。所有 Graph/compile 测试保持原有业务进程运行，并在预算内完成，没有增大显存比例或降低权重精度。

## 复现与部署位置

远端目录为 `/root/saddle-speed-20261002.dd8b7eb_`。原生/Transformers 与 vLLM 分别使用独立测试 venv；依赖从既有环境只读复用，候选 Saddle 包安装在各自测试环境。没有覆盖原业务安装。

候选 wheel：`saddlellm-2.40-py3-none-any.whl`，SHA256 为 `c1d7fd3b6921a516e1a8a5a63e33ccf774a566bfd54a80963d53308c2471088c`。本机 wheel 内容、部署后 import 位置和主要模块散列均已核对。

每组结果目录的 `environment.json` 保存实际启动命令、版本、客户端和启动脚本 SHA256。各请求的原文、长度、首字与完整响应耗时保存在 `c1.json` / `c4.json`；客户端使用 [benchmark_serving.py](../tools/benchmark_serving.py) 的 `--unique-prompts` 调度。

结果中的 `output_text_equal_to_first_target=true` 在只有一个 target 的文件里只是自比较，不能证明跨引擎输出相同；跨文件核对另行记录。客户端自身也不验证远端权重，因此其 `remote_weights_and_hardware_verified=false` 保留原值，独立模型清单和服务器检查提供对应证据。

[独立审计记录](../artifacts/inference_speed_20261002/remote_audit.json) 重新计算样本分位数、token 总量，核对提示顺序、源码散列、跨组文本、业务健康与 GPU 进程。原生兼容/Transformers 的较早两组未单独记录就绪监听进程的 PGID 归属，只有启动日志、健康信息和前后快照等间接证据；后续组增加了明确归属记录。客户端未保存原始逐轮 wall time，所以审计可核对吞吐计算公式、rps 一致性及延迟界，不能仅凭 JSON 独立重算计时分母。

## 无效尝试与适用边界

初次原生/Transformers 启动未限制 allocator，曾大量保留空闲显存，相关初始结果不进入最终表。随后使用明确预算和加载后释放缓存重新测量。vLLM 的旧日志开关参数和缺少 venv `bin` 的 PATH 曾导致启动失败，修正后重测；这些失败不算性能样本。

不同引擎按顺序运行，未在引擎之间做随机交错，已有业务可能带来噪声。28 个请求的 P95 属于小样本开发测量；本轮未做长时间稳定性、开放到达流量或任务质量基准，不能从短文本输出相同或不同推导质量相当、提升或退化。

本次仅部署和测试独立实例，没有替换既有业务服务，也没有发布安装包到 PyPI。
