# 推理迭代：减少取消请求的无效计算

这次加强的是 Saddle 自身的推理执行与资源管理，不是改变模型权重或提高知识/推理正确率。
`saddlellm serve` 和原有 import 用法不变，没有新增必填参数。

## 已落地：非流式断连取消

原先非流式 HTTP 路由不会主动读取客户端断连通知。即使用户离开，排队请求仍可能执行，
正在运行的长回答也可能一直生成到输出上限。直接取消 asyncio 任务又不能终止底层模型线程。

现在 `/v1/completions`、`/v1/chat/completions` 的 local/native 路线：

1. 解析完请求体后，独立等待 ASGI `http.disconnect`，不定时轮询、不抢占 SSE 的 receive。
2. 尚未执行的请求取消排队；微批按行设置停止标记，不取消同批其他客户。
3. 原生单请求在 token 步之间检查 CPU Event；没有增加每步 GPU 张量或同步。
4. HF/自定义 `generate` 通过 stopping criteria 接收停止信号。
5. 模型线程真正退出后才释放模型并发许可；反复取消也不能绕过清理。
6. 请求结束/失败后回收断连监听任务，防止后台监听累积。

同步 Python API 也可选传 `engine.complete(request, cancel_event=event)` 或
`engine.chat(request, cancel_event=event)`，普通调用不受影响。取消发生在执行前会抛出
`concurrent.futures.CancelledError`；执行中的协作停止可返回已生成的部分结果。
HTTP 中已断开的请求不会得到伪造的模型完成响应。

限制：不能强行中断一个正在执行的 CUDA kernel/prefill；自定义模型若忽略 stopping criteria，
仍须等待它实际退出，不能安全杀掉 Python 线程。反向代理若未把断连传给后端，后端也无法提前感知。
这不是新增服务端请求截止时间，也不是连续批处理。vLLM 的现有 abort 路线没有改动。

## 验证：真实 TCP 断连后，其他请求更快得到服务

使用本地已有 28M Saddle 检查点 `outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model`，
CPU FP32、1 个 Torch 线程、同一模型实例、C1、无前缀缓存。
长请求预算 64 token，在首次 forward 进入后关闭 TCP 连接，随后发送预算 2 token 的短请求。
两条路线都关闭 EOS，固定该工作负载；不是模型质量测试。

旧行为基线只关闭断连监听，其他代码、模型、请求保持相同。每轮预热后做 7 对 AB/BA，独立启动两轮。
计时是后续短请求的真实 HTTP 响应时间，不是整段混合负载的 tokens/s。

| 观测 | 第一轮：旧 → 新 | 第二轮：旧 → 新 |
| --- | --- | --- |
| 废弃长请求 forward 步数（每轮 7 次） | 每次 64 → 1 | 每次 64 → 1 |
| 后续短请求中位响应 | 480.3 → 34.3 ms | 1258.4 → 85.5 ms |
| 后续回答/usage | 完全一致 | 完全一致 |
| 请求完成后的 active | 全部为 0 | 全部为 0 |

两次机器负载/绝对时延有明显波动，不能把这组结果当成稳定 SLA。
它验证了“废弃工作不再阻塞后续请求”，**不等于普通请求或 8B 模型整体加速十几倍**。
没有测 8B、没有与 vLLM/Ollama 做新吞吐对比。

原始记录：[第一轮](../artifacts/inference_lifecycle_20260929/disconnect_cpu_run1.json)、
[第二轮](../artifacts/inference_lifecycle_20260929/disconnect_cpu_run2.json)。

复现（输出必须是新文件）：

```powershell
python tools/benchmark_disconnect_recovery.py --model outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model --output artifacts/my-disconnect-recovery.json
```

脚本只启动自己的临时回环服务并在退出时关闭，不连接、不重启现有服务。

## 没有默认启用的提速候选

尝试复用微批取消标记、去掉原生生成循环重复的 EOS 检查。
控制逻辑微基准有收益，但完整模型的配对结果没有达到既定门槛：

| CUDA / 28M / B4 / 输出预算 48 | 第一轮 | 第二轮 |
| --- | --- | --- |
| 控制微基准配对中位速度比 | 1.595 | 1.515 |
| 完整模型调用配对中位速度比 | 1.017 | 1.025 |
| 候选更快的配对数 | 5 / 7 | 7 / 7 |
| 输出文本、结束原因、usage | 一致 | 一致 |
| ≥3% 且 ≥6/7 配对更快 | 未通过 | 未通过 |

共享 GPU 测试仅作为开发样本；没有测 HTTP、TTFT，也没有完成隔离显存/长压验收。
不能把控制微基准的 1.5 倍说成模型快 1.5 倍。
候选已从生产路径撤回，只保留在 `tools/benchmark_batch_control.py` 和针对性测试中；
没有增加用户开关，现有 `serve` 不会启用它。

[第一轮](../artifacts/inference_lifecycle_20260929/control_eos_cuda_run1.json)、
[第二轮](../artifacts/inference_lifecycle_20260929/control_eos_cuda_run2.json)。
更早的纯标记复用记录也保留；其中 CPU 首轮与测试并行，只能作为探索样本，不作为性能验收。

## 代码与后续工作

- `runtime/InferenceLifecycle.py`：断连监听、取消传播、任务清理所有权。
- `runtime/InferenceServer.py`：连接 HTTP、排队、HF/native 停止与模型许可。
- `tests/test_inference_disconnect.py`：排队/执行断连、重复取消、批内隔离、真实 TCP、监听任务不泄漏。
- `tests/test_batch_generation_control.py`：候选与旧逻辑逐步一致性及失败门槛。

接下来更有价值的是隔离 GPU 的实际 Qwen3/DeepSeek 工作负载分析，再决定原生流式连续批处理、
分块 prefill、KV 管理等较大改造。不能把它们写成已经具备的能力。
30 分钟混合负载、8B、多 GPU、其他平台仍需单独验收。

## 本轮交付验收

- 相关回归 **490 项通过**，包括真实 TCP 断连、原生/HF 输出一致性、批内取消隔离、50 次成功/失败后的监听任务清理，以及原有服务/训练导出相关回归。
  [测试报告](../artifacts/inference_lifecycle_20260929/regression.xml)。不是全仓库测试全绿声明。
- 新 wheel 已在之前独立创建的 Windows CPU venv 中安装，离开源码目录执行隔离安装验收；
  `pip check`、真实 CLI、OpenAI SDK chat/completion/SSE、usage、打包资源和 active 回收均通过。
  [安装包报告](../artifacts/inference_lifecycle_20260929/installed_wheel_windows.json)。
- 在该独立 venv 的新版 Web 依赖栈上，再对**已安装 wheel 的运行时**执行 14 项断连测试，全部通过；
  断言实际加载的是 site-packages 而非工作区源码。测试辅助文件来自源码目录，不是完全无源码测试。
  [安装包断连回归](../artifacts/inference_lifecycle_20260929/installed_wheel_disconnect.xml)。两个改动运行时文件与 wheel 内容逐一相同。
- wheel 内容契约、发布元数据、函数手册检查和错误级 Python 静态检查通过。
- CI 加入新用例；跨平台作业尚未远端执行。本机 Python 仍为 3.11.0rc2，不能代替稳定版平台矩阵。
- 8127–8130 原有服务未重启/升级，结束时均 ready、active=0；新改进要在后续重启或新部署后生效。

本轮候选：[saddlellm-2.40-py3-none-any.whl](../artifacts/inference_lifecycle_20260929/wheel/saddlellm-2.40-py3-none-any.whl)。
SHA256：`16b713f095b741f7253df03f3461a03002b0a1b797470e56f0dd86e90a9c30c9`。
未提交、推送、上传 PyPI 或部署到现有服务；保留此前候选，便于核对与回退。
