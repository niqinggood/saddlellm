# 推理优化验证与迭代方案

目标：让相同模型在 Saddle 自有推理路径上更快，同时保留正确性和简单的 CLI/import。
本方案不把接入第三方引擎、降低精度、缩短回答或关闭思考产生的差异当成原生无损加速。

## 1. 先固定对照条件

- 同一模型 snapshot、权重、tokenizer、chat template、精度和 attention 后端。
- 同一输入 token、输出上限、采样参数、EOS/stop、thinking 设置。
- 记录设备、PyTorch/CUDA 版本、模型来源、代码哈希；GPU 不同时运行其他测试任务。
- 冷启动单独统计；性能对照先预热，每个变体至少 7 次，AB/BA 交替。
- 前缀缓存的关闭、冷缓存和热命中分开测，不把缓存命中当作模型解码变快。
- 基线和优化版使用同一权重，控制变量；HF/vLLM 等跨引擎比较另列，不能代替自身 A/B。

## 2. 三层验收

| 层次 | 测什么 | 接受条件 |
| --- | --- | --- |
| 正确性 | 相同前缀 logits、贪心 token、固定种子采样、EOS、多 EOS、最小输出、stop、padding、保存重载 | 控制逻辑优化要求 token/文本/结束原因完全相同；涉及数值内核时单独检查误差和任务质量，不能隐瞒分叉 |
| 单模型性能 | prefill+decode 总时长、实际输出 token/s、首个非空文本时间、PyTorch 峰值显存 | 输出相同；先要求中位吞吐至少提高 3%、至少 6/7 对更快，再独立复测一次；未达标标注“收益未证实”，不宣传提速 |
| 服务质量 | OpenAI 文本 API、SSE、usage、stop、断连取消、排队/超限、健康状态与并发 | 无错误/泄漏、计数准确、取消后可继续请求；并发吞吐和 P95 延迟一起看，不只报告总 token/s |

3% 是本项目本轮的初筛门槛，不是统计显著性证明。正式发布性能结论前，增加轮数和不同提示，报告分布与配对结果。
首字延迟中位数或峰值显存恶化超过 5% 时必须解释并复测；不因吞吐提高自动放行。
共享 GPU 上的测试只算开发样本，部署验收应使用隔离负载。

## 3. 工作负载矩阵

先跑本机可承载的 Qwen3-0.6B、DeepSeek-R1-Distill-Qwen-1.5B；Llama 目前先做同权重小 fixture。
后续在显存允许的独立机器验证 7B/8B，再扩展完整 MoE；不能用小模型结果推断大模型。

每个实际可运行模型覆盖：

- 提示长度 128 / 1024 / 4096，输出 32 / 128 / 512；不得超过模型/服务上限。
- 单请求流式与非流式；并发 1 / 2 / 4 / 8，分别记录总吞吐、每请求延迟、TTFT、失败数。
- 中文解释、英文列表、代码、数学、短答；质量集与性能长输出提示分开保存。
- 随机采样固定种子；长输出自然结束时按真实 usage 计数，不能强行以预算作分子。
- 重复提示与不同提示分别跑；前缀缓存未命中、部分命中、完全命中分别跑。
- 服务稳定性至少 30 分钟，包含客户端中断，观察活跃任务和显存是否回落到稳定水平。

这是验收矩阵，不代表全部项目已经跑完；实际完成范围见每次实验记录。

## 4. 可复现命令

在项目目录执行（模型必须已缓存/下载，不会自动拉取大权重）：

```powershell
python -m pytest -q tests/test_native_architectures.py tests/test_native_qwen3.py tests/test_generation_single_row.py tests/test_native_generation_session.py tests/test_generation_sampling.py tests/test_native_streaming.py tests/test_generation_stop.py tests/test_native_prefix_cache.py

python tools/benchmark_native_single_row.py --model Qwen/Qwen3-0.6B --tokens 96 --repeats 7 --warmup 2 --output artifacts/my_single_row_run1.json

python tools/benchmark_native_single_row.py --model Qwen/Qwen3-0.6B --optimization rotary --require-speedup --output artifacts/my_rotary_run1.json
```

脚本在同一已加载的模型上交替切换选定候选（单请求控制或解码 RoPE 合并）；保存每轮 token ID、文本哈希、耗时、首个可见文本时间、峰值显存和代码哈希。输出不一致时非零退出，报告不覆盖旧文件。
`--require-speedup` 额外启用上表的自动性能门槛，任意检查不通过即非零退出，并在报告中逐项说明原因。
计时包含会话初始化、prefill、decode 和文本 streamer，不含加载、输入分词或 HTTP；首个可见文本不等于首个 token。
显存为当前进程 PyTorch allocated 峰值，不是整卡占用。

对已经启动的服务测量：

```powershell
python tools/benchmark_serving.py --target native=http://127.0.0.1:8130/v1 --model DeepSeek-R1-Distill-Qwen-1.5B --requests 8 --concurrency 1 --max-tokens 96 --stream --repeats 3 --capture-text --output artifacts/my_http_c1.json
```

再分别测并发 2/4/8，使用不同新报告路径。该脚本测 `/v1/completions` 原始提示；聊天模板效果另测 `/v1/chat/completions`，不要混算。
不在正在跑模型 A/B 时同时压测服务。

## 5. 如何根据结果继续迭代

1. 正确性失败：停止性能结论，定位最早不同的 token/logits；修正或禁用候选路径。
2. 输出相同但没有稳定加速：保留原始记录，做 profiler 定位；先查 GPU 同步、小算子、缓存复制与 CPU 调度，不盲目添加新开关。
3. 单请求快、并发差：检查排队与批处理；不能把增加工作线程当成连续批处理。
4. 短提示快、长提示慢：分别定位 prefill 与 KV 读写，再评估分块 prefill、缓存或 attention 内核。
5. 达到门槛且复测通过：保留针对性回归测试，更新支持矩阵和实测记录；未验证平台明确标注。

所有优化都经历“假设 → 单变量实现 → 正确性 → 性能 → 修正/保留/撤回”。不通过验证的候选同样写入记录，避免重复踩坑。
