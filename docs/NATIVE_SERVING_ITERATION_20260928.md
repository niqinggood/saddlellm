# 原生推理服务迭代：数值兼容修复与调度实测

本轮继续使用 [验证方案](INFERENCE_VALIDATION_PLAN.md)，针对上轮发现的 DeepSeek 输出分叉和流式并发排队做改进。
默认 CLI/Python 使用方式不变，没有自动增大并发、量化、关闭思考或改变模型权重。

## 1. 修复已观察到的 DeepSeek 输出分叉

上一轮原生 SDPA 与 HF 的 BF16 执行路径不同，真实 DeepSeek-R1-Distill-Qwen-1.5B 第 5 步发生 argmax 分叉。
新增 `sdpa_reference` 算子路由：无 mask 时保留 compact GQA，允许 PyTorch 选择数学回退；有 mask 时展开 KV，保留同一 attention mask。
不做原生速度路径的单步 GQA folding，也不修改全局 SDPA 开关。模型前向与生成循环仍属于 Saddle，不调用 HF `generate()`。

CUDA BF16 的 Qwen2/Llama 稠密适配器，在没有 FlashAttention 的构建中且用户未覆盖 attention 时，自动采用该路径。
旧 Saddle 检查点、Qwen3 和 INT8 的默认策略不变；这是当前已验证环境的保守选择，不保证任意未来 Transformers 版本逐 bit 一致。

真实 1.5B 验证：

- HF SDPA 对原生 `sdpa_reference`，同前缀 32 步，logits 最大绝对差全部为 0。
- 中英文解释、英文列表、代码共 4 条原始 completion 提示，各输出最多 96 token：HTTP SSE 文本和 usage 全部相同。
- 后一项含一次冷启动、无预热，仅用于输出验证；HF/native 本轮约 37.79/23.37 token/s，不能据此宣称原生更快。
- 新路径可能比速度优先 SDPA 慢；本轮优先修复已发现的分叉，不把答案不同的速度数字当成果。

[同前缀报告](../artifacts/native_service_iteration_20260928/deepseek_reference_logits.json)、[四提示 HTTP 验证](../artifacts/native_service_iteration_20260928/deepseek_http_reference_prompts.json)。
原有失败样本不删除，见 [上轮记录](NATIVE_MULTI_MODEL_RESULTS_20260928.md)。

显式选择仍可用：

```powershell
saddlellm serve /models/DeepSeek-R1-Distill-Qwen-1.5B --backend native --device cuda --dtype bfloat16 --native-attention-backend sdpa_reference
```

在上述自动匹配的环境中可以省略最后一项。`sdpa` 是原速度优先路径，`eager` 是数学参考路径；不能只把名字相同当作跨框架内核相同。
`/health.native_attention_backend` 和 `native_source.attention_policy` 显示实际选择。

## 2. 生产默认保持稳定，增加调度可观测性

原生流式请求继续由请求许可限制并发，使用 asyncio 默认线程池执行阻塞步骤；HF/custom generate 和非流式合批机制不变。
新增的独立有界线程池以及单工作线程、逐 token 公平轮转都只保留为内部实验，**生产默认不启用**。
候选已加入断连、慢客户端、失败隔离、线程回收与隐藏 token 调度边界测试，取消后等待实际模型步结束再释放资源。
调度正确不等于性能更好；在吞吐或尾延迟回归时不能以“架构更专业”为由替换稳定路径。

用户仍只需：

```powershell
saddlellm serve /models/DeepSeek-R1-Distill-Qwen-1.5B --backend native --device cuda --dtype bfloat16 --max-concurrency 4
```

默认仍是 1。显存不足不要设置 4；并发越大，同时驻留的 KV 与工作区越多。
`/health` 增加 `stream_scheduling`、`stream_worker_limit`、`stream_resident_limit`，区分实际线程池与声明的并发上限。
默认分别为 `request_permit`、`null` 和配置的并发上限；`null` 指未创建专用工作池，不代表没有并发限制。
这些调度候选不等于连续批处理、分页 KV 或 GPU 内核融合。

## 3. 没有默认启用的候选

开发了按 token 轮转的流式候选，加入断连、慢客户端、失败隔离等测试。
首版发现多余的空调度跳转及工作线程切换，随后固定工作线程、合并调度边界。
第二轮相对旧线程并发的首字中位数约 168.62 → 132.78 ms，但吞吐 22.23 → 21.84 token/s，P95 总延迟 8694 → 11375 ms，未通过尾延迟门槛。
因此该候选只保留在内部实验/回归中，`serve` 不会默认启用，也没有增加面向普通用户的新调度开关。

[首版失败记录](../artifacts/native_service_iteration_20260928/deepseek_fair_streams_run1.json)、[修正后记录](../artifacts/native_service_iteration_20260928/deepseek_fair_streams_run2.json)。
两次输出长度不同，不跨轮计算改进百分比；每份报告内部变体条件相同。

随后比较有界线程池：同为 4 并发、每次最多 48 输出 token，先预热、各两轮，旧线程池约 28.75、有界线程池约 22.50 token/s；候选没有优势。
为排查显存压力，暂停闲置的 8129 测试服务后复测，并新增逐轮显存记录。文件名虽含 `isolated`，但 8127/8128 仍在线，**不是整机独占基准**。
复测每种方案预热一轮，再各跑三轮、每轮四请求；记录的设备可用显存约 9.6 GiB，输出和 usage 全部一致、结束时在途请求全部归零。

| 同一次复测的策略 | 总输出 token/s | 首字 P50 (ms) | 请求总延迟 P95 (ms) |
| --- | ---: | ---: | ---: |
| 请求串行（并发许可 1） | 31.91 | 2374.22 | 6136.08 |
| 已有线程池（许可 4） | 31.19 | 128.50 | 7623.65 |
| 有界线程池候选（许可 4） | 26.23 | 139.29 | 8766.72 |
| 逐 token 轮转候选（许可 4） | 33.15 | 93.18 | 5919.86 |

最后一轮轮转候选好看，但前轮尾延迟明显回归，且未完成规定的至少 7 组配对和独立复验，不能据此默认启用。
有界线程池在两次对照中都降低吞吐，同样关闭。**本轮没有可发布的吞吐提速结论，落实的是正确性修复、可复验基准和默认策略保护。**

[有界线程池首测](../artifacts/native_service_iteration_20260928/deepseek_bounded_streams_run3.json)、[带显存记录的复测](../artifacts/native_service_iteration_20260928/deepseek_bounded_streams_isolated.json)。
测试设备为 Windows、RTX 4090 Laptop 16 GB、BF16 DeepSeek-R1-Distill-Qwen-1.5B，不能外推到 8B 或 vLLM。

## 4. 如何复验

```powershell
python tools/validate_native_dense_logits.py --model /models/DeepSeek-R1-Distill-Qwen-1.5B --attention sdpa --native-attention sdpa_reference --steps 32 --output artifacts/my_reference_logits.json

python tools/benchmark_native_stream_scheduling.py --model /models/DeepSeek-R1-Distill-Qwen-1.5B --tokens 48 --repeats 3 --output artifacts/my_stream_scheduling.json
```

后者只加载一份本地权重，启动四个临时 loopback HTTP 服务，交错测串行、旧线程池、有界线程池和实验轮转方案，结束后关闭临时服务。
模型、精度、输入、token 上限与采样均相同；报告包含文本哈希、实际 usage、TTFT、P95、健康状态和代码哈希。输入原始 completion 提示，不是聊天质量评测。
输出或用量不一致、请求未完全释放时非零退出。性能优劣需要同时审阅吞吐和延迟，不能只看退出码。

完整大模型、8B、多 GPU、Linux/MPS、长上下文与长期稳定性仍需各自验证，本机结果不能替代它们。

## 5. 交付与回归状态

- 最终相关回归 **369 项通过**，覆盖架构/注意力、原生生成、缓存、断连、调度候选、OpenAI 协议、平台规划和 Web UI。不是全仓所有测试；依赖库有弃用告警。
- 初次回归的旧空跳转计数断言失败，已改为逐次检查每个拉取至多推进一个 token，同时验证最终文本和资源回收；初次报告保留，不覆盖。
- [最终回归报告](../artifacts/native_service_iteration_20260928/regression_final.xml)；变更文件 Ruff `E9,F63,F7,F82` 检查通过。
- 8127 主页面与 8128 HF 对照服务未重启；8129 Qwen3 原生测试服务、8130 DeepSeek 原生测试服务已恢复。四个端口健康检查均就绪、无在途请求。
- 两个恢复后的原生服务均通过 OpenAI Python SDK 实际聊天、completion、SSE usage 和错误协议冒烟验证：[DeepSeek](../artifacts/native_service_iteration_20260928/deepseek_native_sdk_final.json)、[Qwen3](../artifacts/native_service_iteration_20260928/qwen3_native_sdk_final.json)。这不是质量或吞吐基准。
- 重新构建 [2.40 wheel](../artifacts/native_service_iteration_20260928/wheel/saddlellm-2.40-py3-none-any.whl)，包内容契约与七个核心变更文件的源码一致性检查通过。未发布 PyPI；本地服务从源码运行，不声称已部署此 wheel 到远端。

wheel SHA256：`3774fc74d12c1940519a05992389d45c66381f1387258c78b527332a5d1ccf48`。
