# 原生 SSE 合批：使用、实现与验证

2026-09-29。本次把 Saddle 自己的 `NativeBatchDecodeRunner` 接入 OpenAI 兼容流式服务，
不调用 vLLM/SGLang，不改变模型权重，不需要自行编译 C++。
**实验性、显式启用、默认关闭；不是完整的分页注意力或任意长度连续批处理引擎。**

后续已支持 [默认采样合批、显式 BF16 吞吐模式与不同长度投影合批](NATIVE_SAMPLING_THROUGHPUT_20260929.md)，并统一流式/非流式逐步调度。下文性能数据保留原轮次口径。

后续已在 [远程 L20 / Qwen3-4B](REMOTE_QWEN3_4B_SSE_VALIDATION_20260929.md) 验证：
四客户端下，相对原生默认串行许可约 +39% 总吞吐；相对原四并发许可配置约 1.9 倍。
下面 28M 的数字仍是原始本机实验，不能混作 4B 或 8B 的绝对速度。

## 使用

```powershell
saddlellm serve ".\your-native-compatible-model" --backend native --device cuda --dtype bfloat16 --native-stream-batching --max-concurrency 4 --batch-size 4 --batch-token-budget 8192
```

或者在主入口脚本中：

```python
from saddlellm import serve

if __name__ == "__main__":
    serve("./your-native-compatible-model", backend="native",
          device="cuda", dtype="bfloat16", native_stream_batching=True,
          max_concurrency=4, batch_size=4, batch_token_budget=8192)
```

调用 `/v1/chat/completions` 或 `/v1/completions` 时可使用默认采样或 `temperature=0`。
需要真实 token 统计时再设 `stream_options={"include_usage": true}`。
不设 `temperature` 的请求默认采样，也可进入合批。标准稠密模型的不同 KV 长度可共享投影计算，attention 仍逐请求执行。

参数含义：

| 参数 | 作用与边界 |
| --- | --- |
| `--native-stream-batching` | 显式开启本功能；必须同时选择 `--backend native`，不自动改用外部后端 |
| `--max-concurrency 4` | 原有服务许可上限；流式请求驻留受其约束，SSE 关闭且计算真正结束后归还 |
| `--batch-size 4` | 单次最多协调四个请求；启用本功能后流式与非流式共用逐步调度，停用旧固定微批 |
| `--native-decode-mode throughput` | 显式使用 BF16 批量 Linear，可能改变生成 token；默认 `compatible` 保留逐行 Linear |
| `--batch-token-budget 8192` | 执行器共享组的预留 token 槽位预算；不是整卡显存保证，也不是单轮生成 8192 个 token |
| `--batch-wait-ms 5` | 收集一批请求的等待窗口；Python/OS 定时器不是硬实时，实际唤醒可晚于这个值 |

该开关要求 `batch_size >= 2, max_concurrency >= 2`。只接受浮点 FP32 或 CUDA BF16；
FP16、CPU BF16、Saddle W8A16/其他量化权重不能打开此服务选项。
真正批量前向还要求无 padding、合格的稠密 MHA/MQA/GQA 几何缓存等条件；等 KV 长度使用共享组，不同长度使用受限的独立 KV 路径。
不合格项保留逐请求执行，不假装已经合批。CUDA BF16 默认沿用 [逐行 Linear 兼容路径](NATIVE_BF16_BATCH_FIX_20260929.md)。

## 怎样确认真的在用

访问 `/health`：

- `stream_scheduling=native_batch_decode` 表示已配置新调度。
- `native_stream_batching.batched_forwards`、`batched_tokens` 和 `largest_batch` 表示实际批量执行；仅 `enabled=true` 不代表发生了合批。
- `rowwise_linear_forwards` 记录 BF16 兼容批量前向次数。
- `decode_mode`、`batched_linear_forwards`、`sampling_batched_tokens`、`batch_size_histogram`、`individual_reasons` 反映执行模式、实际采样合批及回退情况。
- `independent_kv_forwards` 记录不同 KV 长度共享投影前向的次数；不代表 attention 已分页。
- `cohort_builds` / `cohort_history_copy_bytes` 用于检查 KV 重组成本。
- `single_request_steps` 记录单请求直通提交；`alignment_deferrals` 记录有限的步数对齐让行。
- 请求全部完成后，`registered/pending/inflight/retained_cohorts` 应归零。工作线程空闲但仍存活是正常的；服务关闭时再退出。

## 关键代码流程

1. `InferenceServer.stream_response()` 保留原有鉴权、参数校验、HTTP 接入限制和 semaphore 许可。
2. `LocalGenerationEngine._stream_native()` 创建请求独立 session、streamer、停止条件和文本缓冲。
3. 原生请求注册到 `NativeStreamBatcher`，每次只提交一个 token 步；流式消费方没有再拉取时，不继续在后台生产文本。
4. 只有一个合批参与者时直接在当前拉取线程执行，省掉收集等待和模型工作线程交接；仍使用共同执行锁。
5. 多个参与者提交后，常驻工作线程选取有限个任务，交给 `NativeBatchDecodeRunner.step()`。单独 prefill；兼容 decode 共用模型前向；标准层的不同 KV 长度可批量计算投影，再逐请求计算 attention。
6. 同长度 prompt 的新请求若只落后一两步，可先推进已排队的落后者，让它们加入同一组；每个已提交步最多让行两次，不等待尚未提交的慢客户端来追赶。
7. 执行器按稳定 session 身份排列组内行，避免网络到达顺序变化导致 KV 全量重组；结果仍映射回原来的请求。
8. 每个 session 独立处理预算、停止词、EOS、惩罚与 usage。SSE 发送仍使用既有文本规整和 `[DONE]` 协议。
9. 断连设置取消信号；已经进入模型的调用必须结束后才能 unregister、close KV、归还许可。清理和前向共用执行锁。
10. 关闭服务时唤醒/结束等待任务、等待实际模型执行、关闭 runner，最后清理前缀缓存。空闲循环不持有上一个 Future/异常的临时引用。

实现入口：`saddlellm/runtime/NativeStreamBatcher.py`、`saddlellm/runtime/InferenceServer.py`、
`saddlellm/models/NativeBatchDecode.py`。不是把多个 Python 线程叫作真正的 GPU 合批。

## 验证与复现

```powershell
python -m pytest tests/test_native_stream_batching.py tests/test_native_stream_fairness.py tests/test_native_batch_decode.py tests/test_serving_plan.py
python tools/benchmark_native_stream_batching.py --model outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model --output artifacts/my_sse_c4.json --require-speedup
python tools/benchmark_native_stream_batching.py --model outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model --concurrency 1 --output artifacts/my_sse_c1.json
python tools/benchmark_native_stream_batching.py --model outputs/zh_en_small_pretrain_v2_20260923/run_28m/final_model --prompts-file tools/fixtures/native_stream_mixed_prompts.json --output artifacts/my_sse_mixed.json
```

每次使用新报告名，工具拒绝覆盖旧实验。基准在本机随机空闲端口启动两个临时服务，共享同一份模型，
顺序、打乱顺序配对测量 OFF/ON；每个目标两轮预热、七轮测量，真实 HTTP/SSE、真实 usage，EOS 不人为关闭。
记录可见文本首包延迟、完成延迟、输出吞吐、峰值 allocated 显存、输出文本散列、token 数和健康状态。
源码哈希在测量前后核对。首包是首个非空可见文本，包含 tokenizer 缓冲，不等同于 GPU 产生第一个 token 的时间。
加 `--require-speedup` 才要求完整性能门槛通过；不加时返回成功不代表性能提升，仍须阅读报告的 `development_gate`。

单元/集成覆盖：FP32 CPU 与 BF16 CUDA 的真实批量形状、两种 OpenAI SSE 端点、文本/usage/结束原因、
采样旁路、单请求直通、到达顺序变化、有限步数对齐、停止词、慢消费者、断连、排队取消、关闭期间在途任务、
异常恢复、驻留限制、清理与前向竞争、CLI 参数及安装包文件契约。

## 实测结果与迭代记录

### 最终 C4 同长度负载

本机 Windows、RTX 4090 Laptop、28M 原生 checkpoint、CUDA BF16、48 个输出 token 上限，
四个并发请求使用相同提示词。对照是当前 Saddle 原生服务的合批开关 OFF/ON，而不是旧非流式固定微批、HF、Ollama 或 vLLM。
现有服务未关闭，外部负载未隔离；不能把这些数字外推到 8B。

| 独立测试 | OFF → ON 总输出 token/s | 配对耗时比中位数 | P95 可见首包 OFF → ON | 输出/usage | 全部开发门槛 |
| --- | --- | --- | --- | --- | --- |
| 最终轮 1 | 134.79 → 276.89 | 2.050× | 42.31 → 41.62 ms | 一致 | 通过 |
| 最终轮 2 | 136.73 → 279.54 | 2.048× | 39.70 → 40.92 ms | 一致 | 通过 |

每轮各七对测量，七对均更快；各目标有 28 个完成请求。所有请求实际生成 48 token。
门槛同时检查实际发生批量前向、输出文本/计数一致、资源归零、至少七对、至少 3% 的配对中位提升、
至少 85% 配对胜出、P95 可见首包回退不超过 5%、峰值 allocated 显存增幅不超过 5%。
这不是严格的质量评测或长时间稳定性证明。第二轮首包略慢约 3.1%，不能说每个指标都提升。

原始证据位于 `artifacts/native_stream_batching_20260929/bf16_c4_aligned_run1.json` 和
`bf16_c4_aligned_run2.json`，两轮源码在测量前后均未变化。峰值显存是当前测试进程的 PyTorch allocated，
不是整卡显存或部署所需总预算。不同目标依次预热会建立线程/算子工作区，因此不可只拿未预热的最小显存当基线。

### 如何迭代到上述版本

最终源码的单请求复测 `bf16_c1_aligned.json` 为 129.03 → 127.66 token/s，P95 可见首包 11.39 → 12.65 ms。
它没有达到性能提升门槛，也仍有约 1.3 ms 的首包代价；**不要为单用户加速而打开此开关**。
后续应优先做不同长度请求的真正合批，以及长 prefill 对流式延迟的隔离，而不是宣布所有负载都提速。

最终混合长度复测 `bf16_c4_mixed_aligned.json` 为 121.16 → 129.27 token/s，但实际批量前向仍为零，
P95 可见首包 78.88 → 84.50 ms，未通过全部门槛。它只能说明逐请求回退能正确工作，不能当作混合长度合批收益。

早期 HTTP 压测暴露清理与下一批前向争锁，已统一串行化并添加可确定复现的测试。
之后发现按到达顺序排列组内行会频繁复制 KV：一个九轮 C4 记录中发生 197 次组创建、约 146.8 MB 历史复制；
稳定行顺序后的另一轮降到 12 次、约 4.49 MB。两个实验还有收集策略变化，不能把所有速度差都归因于行排序。

单请求初版约 131.0 → 124.4 输出 token/s，P95 可见首包 12.2 → 35.5 ms，属于回退；
直通版本复测约 128.1 → 127.4 token/s，首包 13.24 → 13.87 ms，消除了这轮明显额外等待，但没有证明单请求更快。
直通最初又使早到者固定领先一步而分成 B1+B3，因此继续补充了有次数上限的步数对齐。

混合长度四提示的先期测试约 126.1 → 131.8 token/s，但实际批量前向为零，配对中位收益未达门槛，首包回退。
这不是混合长度加速成功。所有请求文本和 token 数仍一致，功能正确不能代替性能验收。

中间某轮 C4 虽吞吐提升但 P95 首包回退，未通过全部门槛。不能挑最好的一轮写成普遍承诺。
`bf16_c4_stable_rows_run2.json` 测量期间改过 health 展示字段，其测后源码哈希不能代表当时完整已加载源码，
保留为探索记录、不作为最终源码验收依据；后续工具补了测量前后源码一致性检查。

## 未解决的边界

不代表 8B、所有架构、所有 GPU、长上下文或开放流量下均有同样收益。
混合长度没有 padded/ragged 合批，长 prefill 仍可能阻挡 decode，没有 chunked prefill、分页 KV、跨请求 KV 内存池，
也没有新增量化算子、CUDA Graph 或 speculative decoding。本轮没有验证历史 8B W8A16 固定微批的质量问题。
已有 8127–8130 服务未重启、未切换模型。新功能继续默认关闭，以便逐模型、逐业务验证后启用。

## 交付验收与候选安装包

- 相关回归 **892 passed**，包含 CUDA BF16、原生生成、缓存、HTTP/断连、CLI、训练与已有服务回归：
  `artifacts/native_stream_batching_20260929/regression.xml`。不是声称跑完了仓库所有测试。
- 较新 FastAPI/Starlette 等依赖环境的定向复测 **86 passed, 2 skipped**：
  `current_dependencies.xml`；该隔离环境是 CPU Torch，两个 CUDA 用例跳过，不重复算成 GPU 验证。
- 工作区外独立安装 wheel，以 `python -I` 启动真实 CLI：默认模式、显式合批模式均通过 OpenAI SDK 的
  chat/completions/SSE、网页资源、禁用历史记录、依赖完整性和许可释放检查。
  `installed_wheel_batched.json` 中真实 `largest_batch=4`，三个批量前向，输出相同且所有请求/组引用归零；
  `installed_wheel_default.json` 验证原默认路径。
- wheel 文件契约检查通过，包内 192 个 Saddle Python 文件与本地对应源码逐字节相同。
- 本次相关代码静态检查通过；全仓 F601/F821/F822/F823 仍有一处既有错误：
  `saddlellm/_exports.py` 的 `RolloutSample` 重复字典键。本轮没有改动该导出选择，不能称正式发布门槛全绿。
- 本机旧 setuptools 不满足项目的构建要求，`--no-isolation` 构建失败；使用项目声明的隔离构建成功，
  未为绕过构建工具版本而降级项目元数据，也未升级用户全局环境。

候选包：`artifacts/native_stream_batching_20260929/wheel/saddlellm-2.40-py3-none-any.whl`。
SHA256：`272af3a9adc8d960780e4175c224c17e7bd4afb08b62731317c5a8ba07f0df64`。
它包含本轮之前已有的工作区修改和本轮改动，仍沿用当前版本号；未发布 PyPI、未推送代码、未部署到既有服务。
