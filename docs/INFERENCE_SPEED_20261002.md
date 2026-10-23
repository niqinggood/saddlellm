# 推理提速与简单部署预设（2026-10-02）

本轮在已有原生推理、KV 缓存和流式合批上继续优化。CLI、Python `serve()`、Web 和 OpenAI 文本接口保持兼容。结果按模型、负载和指标分别报告，不将局部收益或不同实验倍数相乘。

## 已实现的缓存热路径优化

`NativePrefixCache` 每次 lookup 仍检查模型结构、参数/缓冲区身份与版本、设备/精度、forward/hook、配置和算子执行设置。将三个模块树扫描合并为一个，减少对不存在属性的重复查找；长 token 前缀按块比较。没有通过省略权重变更检测、共享可写 KV 或缓存答案换取速度。

快照仍有 LRU 和字节预算，命中后复制到请求私有 KV。失效、取消、跨 CUDA stream 和清理保持原有合约；`.data` 等无法追踪的修改仍要求显式 clear。

## 实际测量

环境：Windows、RTX 4090 Laptop 16 GiB、PyTorch 2.6.0+cu124、本地 DeepSeek-R1-Distill-Qwen-1.5B、BF16、一个 Torch CPU 线程。模型计算精度、权重和生成策略在各组之间相同。测试没有独占桌面 GPU，结果属于开发验证。

模型层对照使用 768 个输入 token、8 个固定 greedy 输出 token，部分命中共享前 512 token。每场景 7 轮随机交错，旧/新 fingerprint 完全相同，所有输出 token 一致。

| 模型层指标 | 修改前 | 修改后 | 配对速度比中位数 | 更快轮数 |
|---|---:|---:|---:|---:|
| 完整命中 prefill | 44.73 ms | 27.10 ms | 1.631× | 7/7 |
| 部分命中 prefill | 226.22 ms | 174.16 ms | 1.277× | 7/7 |
| 冷缓存 prefill | 230.13 ms | 190.84 ms | 1.241× | 5/7 |

耗时列各取中位数，配对比先逐对计算，两者的比值不必相同。完整命中含 8-token 生成的配对比为 1.051×、5/7 更快，不能把 prefill 的 1.631× 说成整体生成加速。

负对照：新实现的部分命中仍比不使用缓存慢。当前不缓存 prefill 中位数 151.09 ms；部分命中对它的配对速度比为 0.880×、仅 2/7 更快。冷请求也有建立快照的成本。

### 真实 HTTP SSE

通过同一个已加载模型启动三个临时回环服务，依次测试不缓存、旧缓存和新缓存，使用 `/v1/completions`、768-token 左右的中文提示、8 个固定 greedy 输出 token。两次预热、7 组随机交错；正式请求验证命中数与实际复用 token 增量，结束时验证许可释放和缓存字节归零。

| 重复完整提示 | 不缓存 | 旧缓存 | 新缓存 |
|---|---:|---:|---:|
| 首个可见文本中位延迟 | 182.89 ms | 63.19 ms | 38.09 ms |
| 完整响应中位耗时 | 1.275 s | 1.139 s | 1.075 s |

新缓存相对旧缓存的完整响应配对速度比 1.0347×、6/7 更快；相对不缓存为 1.1928×、7/7 更快。文本、结束原因、usage 全部一致。首个可见文本包含 tokenizer 缓冲，不能称为 GPU 首 token 时间。

另一组每轮更换短问题后缀、复用超过 90% 输入的 SSE 对照：新缓存相对旧缓存完整响应配对比 1.0518×、6/7 更快，但相对不缓存仅 1.0167×、5/7 更快，未达到本轮 3% 且至少 6/7 更快的开发门槛。其首字为不缓存 205.89 ms、新缓存 211.59 ms。这说明仅提高查找速度还不足以保证多轮部分复用获益。

进一步尝试了累计至少 256 个新 token 才发布扩展快照。变化后缀测试中，快照发布从 9 次降至 1 次、延迟发布 8 次；文本、结束原因、usage 仍完全一致。相对不缓存完整响应配对比 1.0229×、6/7 更快；相对已经优化查找的缓存为 1.0267×、4/7 更快；首字两个对照都只有 5/7 更快。仍未通过门槛。因此 `latency` 预设最终只启用完整提示匹配，部分复用和延迟发布不自动启用。

延迟发布作为显式实验选项保留：Python `serve(..., backend="native", prefix_cache_mb=64, prefix_cache_partial=True, prefix_cache_min_tokens=256, prefix_cache_min_store_tokens=256)`。默认 `prefix_cache_min_store_tokens=0` 保留原发布行为；它可能增加后续重新计算的 token 数，不保证加速。旧条目被逐出、clear/close、模型状态变化时仍执行相应失效和发布规则。

原始证据：

- [GPU 缓存模型层记录](../artifacts/inference_speed_20261002/prefix_qwen1_5b_compare.json)
- [模型层配对统计](../artifacts/inference_speed_20261002/prefix_qwen1_5b_paired_summary.json)
- [CPU 对照](../artifacts/inference_speed_20261002/prefix_cpu_compare.json)
- [HTTP 完整命中](../artifacts/inference_speed_20261002/prefix_http_full.json)
- [HTTP 变化后缀](../artifacts/inference_speed_20261002/prefix_http_partial.json)
- [HTTP 延迟快照发布实验](../artifacts/inference_speed_20261002/prefix_http_deferred.json)（此实验的缓存基线是已优化查找的版本）

复现完整命中 HTTP 对照：

```text
python tools/benchmark_prefix_serving.py --model LOCAL_MODEL --baseline artifacts/inference_speed_20261002/baseline/saddlellm/models/NativePrefixCache.py --output NEW_REPORT.json --tokens 8 --fixed-length
```

增加 `--vary-suffix` 测量部分前缀。只有基准进程关闭 EOS 以控制输出长度；产品服务不会这样做。旧源码由显式指定的可信本地文件加载，不能指向不可信下载内容。

## 并发采样与 token 交付

CUDA 原生合批现在合并相同参数请求的采样逐元素操作，以及整批 token 的 GPU→CPU 回传。sort、softmax、cumsum 仍保持逐请求 B1 形状；multinomial 按原提交顺序调用，保留独立、共享及全局 RNG 行为。直接批量 reduction 曾改变低精度概率和输出，已拒绝该实现，不能引用其性能作为成果。

自定义 session、采样函数、停止条件及不受信任 streamer 保留原路径。Web 流只为标准 fast tokenizer 登记内部 streamer；Python 自定义 decoder、方法覆盖等回退，避免任意回调影响后续请求的随机数顺序。

| 原生 session 对照 | 配对速度比中位数 | 更快轮数 | 结论 |
|---|---:|---:|---|
| 28M，C4、等长、throughput 数值模式 | 1.0559× | 6/7 | 所测吞吐提升约 5.6% |
| 1.5B，C4、混长、compatible | 1.0161× | 6/7 | 未达到幅度门槛 |
| 1.5B，C1、compatible | 0.9928× | 3/7 | 单请求负对照，无提速结论 |

三组均恢复保存的旧 `_sample`、`_step_locked` 和 batch runner 作为基线，权重和各自数值模式不变，输出 token 和 Recorder 交付完全一致。28M 峰值分配增加约 1.73%，整个并发组首轮完成时间中位数增加约 6.0%；后者不是 HTTP 首字延迟。该测量包含 prefill、decode、采样、EOS 和 host Recorder，不是 GPU 算子微基准，也不是 Web 吞吐。

证据：[28M 精确基线](../artifacts/inference_speed_20261002/delivery_28m_final_exact_baseline.json)、[1.5B 混长](../artifacts/inference_speed_20261002/delivery_15b_mixed_compatible.json)、[单请求负对照](../artifacts/inference_speed_20261002/delivery_15b_c1_compatible.json)。最后追加的 `_end_streamer` 身份检查仅收紧自定义回调资格，不改变上述已测内置 Recorder 路径。早期使用重构后 `_sample` 的基准单独保留，不替代这三份精确基线记录。

### 流式交付的实际 HTTP 验证

28M、BF16、compatible、四客户端混合长度提示、每请求 32 个固定输出 token、temperature=0.7/top_p=0.9：同一已加载模型、相同当前调度器，仅切换保存的旧 generation/batch core 与候选。预热两轮，7 轮随机交错。

| SSE 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 全部正式请求合计输出吞吐 | 39.85 token/s | 45.17 token/s |
| P95 首个可见文本 | 486.24 ms | 363.17 ms |
| P95 完整响应 | 3942.69 ms | 3258.25 ms |
| 峰值 Torch allocated | 110,492,672 B | 111,670,784 B |

**配对吞吐速度比中位数 1.0709×，6/7 更快**；不能用上表两个总速率的比值替代配对统计。候选含预热实际执行了 288 次批量 host transfer 和 288 次批量采样，交付 1112 个合批 token。没有请求失败，许可/调度状态回到空闲；所有轮次都验证方法恢复，源码散列稳定、服务关闭正常。该 HTTP 测试未设置每请求 seed，不要求不同排队时序下的采样文本相同；逐 token/RNG 等价性由独立核心测试覆盖。

原始记录：[有效 HTTP SSE 对照](../artifacts/inference_speed_20261002/delivery_http_28m_validated.json)。更早的 `delivery_http_28m_compatible.json` 已作废：基准适配器曾在并行 dispatch 回调中 patch 全局方法，导致错误恢复、候选未进入新路径。修复后改为整个顺序 workload 持有基线方法，完全排空后恢复，并增加恢复断言及 `--require-delivery-batching` 门槛；不是生产调度器缺陷。作废数据不用于任何提速结论。

## 简单入口

```text
saddlellm serve MODEL --backend native --dtype bfloat16 --performance latency --dry-run
saddlellm serve MODEL --backend native --dtype bfloat16 --performance throughput --open
```

Python 对应 `serve(path, backend="native", dtype="bfloat16", performance="throughput")`。示例显式指定 BF16，预设本身不会改变 dtype、量化、思考、输入预算或数值模式。省略 performance 使用原有 auto 行为。

`latency` 默认设置 64 MiB 缓存和 256-token 最短门槛，只复用相同的完整长提示。适合重试、重新生成等请求；普通续聊提示逐轮增长，不承诺因此加速。显式指定 `prefix_cache_partial=True` 仍会保留，但不属于验证通过的预设默认值。

`throughput` 补齐未提供的流式合批、四并发和 token 预算，保持 `compatible`。用户的显式参数优先。加载模型后检查资格，失败时原子撤销预设添加项，避免留下多线程独立并发而未合批的半套设置。`--dry-run` 表示计划，`/health` 的 `backend_selection.performance` 表示加载后的实际状态。

普通 HF、adapter 和量化模型不会因预设自动转成原生模型。Linux CUDA 的 vLLM 预设使用已有引擎的前缀缓存，保留其调度参数；远程 L20 的 vLLM 直连与 Saddle 接入实测见 [跨引擎部署比较](REMOTE_INFERENCE_COMPARISON_20261002.md)。配置 token 预算不等于实现了全局显存准入。

## Go / Rust 与后续执行器

本轮沿用 Python/PyTorch/CUDA 执行边界。Ollama 的 Go 层负责队列和 runner 生命周期，其 llama 执行路线通过本地接口调用 llama-server；该底层构建使用 C/C++ 和 GPU 后端。当前 Ollama 另有 MLX 路线，不能将所有模型都概括为同一个执行器。值得借鉴的是清楚的服务/执行器边界，而不是从语言名称推断推理速度。

官方源码：[调度](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/server/sched.go)、[执行器适配](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llm/llama_server.go)、[底层构建](https://github.com/ollama/ollama/blob/b0c1ca4f7549d7acdfa52a7dcffc934bc63a43ce/llama/server/CMakeLists.txt)。

CUDA Graph、分页 KV、变长批量 attention 和 Go/Rust 执行器仍需分别实现与验证，本轮不将其列为已交付能力。没有更新既有常驻业务服务，也没有发布到 PyPI。

## 回归与安装包

最终选定推理/服务回归 **676 passed、1 skipped**；唯一跳过是本机 tokenizers 对包含 Python custom decoder 的 Sequence 构造直接抛出上游 panic，标准 decoder、单独 custom decoder 和方法覆盖回退均通过。新增跨引擎基准的唯一提示调度另外 **25 passed**，避免反复使用同一提示让引擎自带前缀缓存混入冷前缀比较。

范围包含原生采样/合批/缓存、stop/EOS、断连取消、SSE、OpenAI 文本协议、vLLM 适配、CLI 与性能预设；不是全仓测试或模型任务质量评测。所改文件的 Ruff `F601,F821,F822,F823`、生成函数索引、发布元数据和补丁空白检查通过。全仓同类 Ruff 检查仍发现此前已有的 `_exports.py` 重复 `RolloutSample` 键，本轮未修改该导出约定。

生成本地 `2.40` wheel 并通过内容契约和七个主要修改模块的源码散列一致检查。安装包 SHA256：`c1d7fd3b6921a516e1a8a5a63e33ccf774a566bfd54a80963d53308c2471088c`。初次无隔离构建因本机旧 setuptools 不识别项目现有 license 格式失败，使用项目声明的隔离构建依赖后成功，没有修改全局环境或项目元数据绕过该要求。

证据：[本地验收清单](../artifacts/inference_speed_20261002/local_validation.json)、[最终回归](../artifacts/inference_speed_20261002/final_regression.xml)、[基准工具回归](../artifacts/inference_speed_20261002/benchmark_regression.xml)、[安装包源码核对](../artifacts/inference_speed_20261002/wheel_source_check.json)、[相对本轮开始快照的生产代码差异](../artifacts/inference_speed_20261002/task_production_changes.patch)。远程 L20 的跨工具比较单独记录，不混入上述本机 A/B 数字。
