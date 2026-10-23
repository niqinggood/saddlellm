# 原生服务：采样、混合长度合批与显式吞吐模式

本轮落实[吞吐诊断](NATIVE_THROUGHPUT_DIAGNOSIS_20260929.md)的三项：采样请求共享前向、BF16 真正批量 Linear，以及不同 KV 长度的投影合批。原生逐步合批仍需显式启用，默认数值模式仍为 `compatible`。分页 attention 和 CUDA Graph 尚未实现，不能将本轮称为完整生产级推理引擎。

## 启用与回退

```powershell
saddlellm serve "YOUR_MODEL" --backend native --device cuda --dtype bfloat16 --native-stream-batching --native-decode-mode throughput --max-concurrency 4 --batch-size 4 --batch-token-budget 8192
```

在启动前追加 `--dry-run` 可以检查解析后的配置。`throughput` 必须同时启用 `native_stream_batching`，不满足时直接报错，避免接受一个未生效的选项。

如需回到原有 BF16 逐行计算，将 `--native-decode-mode` 改为 `compatible` 后重启服务。关闭逐步合批则移除 `--native-stream-batching`，并省略 `--native-decode-mode throughput`。本次开发没有重启用户已有服务或更改已部署配置。

| 模式 | CUDA BF16 Linear | 数值边界 |
| --- | --- | --- |
| `compatible`，默认 | 各行维持 B1 计算形状，其他合格算子合批 | 延续已有数值路线；不是任意模型、算子版本下逐位一致的保证 |
| `throughput`，显式启用 | 合格 decode 使用真正批量矩阵计算 | 舍入差异可能造成生成 token 分叉；需对目标模型验收质量 |

FP32 两种模式均继续使用批量 Linear。FP16、CPU BF16、INT8 等未验证服务配置仍拒绝开启此选项。模式不修改权重、检查点或进程全局 matmul 设置；单请求和不合格请求仍保留原有执行路径。

Python 执行器使用 `NativeBatchDecodeRunner(model, decode_mode="throughput")`；服务设置对应 `native_decode_mode="throughput"`。

## 请求行为

- 默认 `temperature=0.7` 的请求也能参与合批；不需要将聊天改成 greedy。不同 temperature、top-k/top-p、重复惩罚、EOS、stop 和生成预算仍由各 session 自行处理。
- 开启原生逐步合批后，流式与非流式文本请求共享同一个 `NativeStreamBatcher`，避免两套模型执行队列并行竞争。旧固定非流式微批在该配置下停用。
- 该模式拒绝固定微批专属的 `max_pending_tokens` 和 `batch_length_bucket_size`，避免配置被默默忽略。排队请求数由 `max_pending_requests` 限制；每请求输入/输出上限和驻留请求数共同限制负载，但尚非按 GPU 实际空闲显存准入。逐步调度不使用固定微批的排序策略。
- 合批仍要求同模型、无 padding、支持的稠密 MHA/MQA/GQA、几何缓存及 dtype。相同 KV 长度使用原有共享组；不同长度可在标准 decoder/attention/MLP 上共享投影前向，attention 逐请求读写各自 KV。model/layer/attention 的自定义 forward 或观察 hook 等不支持该缓存接口时，保留原有等长度分组/逐请求回退。
- 请求持有自己的 token/KV/采样状态。原生 Python API 显式 `seed` 创建独立 generator；已有 generator 原样保留。无 seed 的请求仍使用全局 RNG，共享 generator 的请求也仍共享随机数流，因此其输出会受调度顺序影响。组内采样按调用者提交顺序执行，缓存行顺序保持稳定。HTTP API 未在本轮新增 `seed` 支持，原有参数校验继续生效。
- 断连、stop 和 EOS 不会把其他行标记为完成。实际模型计算结束后才允许关闭 session、释放许可；合批 forward 失败仍使相关组失效，防止半更新 KV 被继续使用。

## 可观测性

`/health` 的 `native_stream_batching` 对象新增：

| 字段 | 含义 |
| --- | --- |
| `decode_mode` | 当前配置的数值执行模式 |
| `sampling_supported` | 执行器已支持采样参与合批，不等于每次都会合批 |
| `sampling_batched_tokens` | 批量前向处理的采样请求行数 |
| `batched_linear_forwards` | 使用批量 Linear 的前向次数 |
| `independent_kv_forwards` | 不同 KV 长度共享前向的次数；attention 仍按请求分别执行 |
| `rowwise_linear_forwards` | 使用 BF16 逐行兼容 Linear 的前向次数 |
| `batch_size_histogram` | 实际批大小及其前向次数，例如 `{"4": 10}` |
| `individual_reasons` | 已完成逐请求 step 的原因统计 |

`individual_reasons` 包括 prefill/待消费 logits、没有兼容同伴、组容量预算不足、未支持的 dtype/模型/结构/请求/缓存、autocast 及位置不匹配。`no_compatible_peer` 表示该提交没有可共同执行的同伴，不单独证明具体原因就是输入长度。

合批计数记录成功返回的模型前向及其行数，不是已发送 HTTP token 数。后续采样/streamer 失败时，前向可能已计数；计费和完成量继续以请求的实际 usage 为准。`batch_size_histogram` 加权求和等于 `batched_tokens`。未请求完整服务快照的瞬时读取不保证跨字段事务一致。

请求结束后检查 `registered/pending/inflight/retained_cohorts` 归零。已有 `cohort_builds/cohort_history_copy_bytes` 继续反映 KV 合组成本。批大小和合批行计数只有在并发请求满足条件时才会增长。

## 验证和基准

新增 `tests/test_native_sampling_batch.py`，覆盖固定 seed、全局/共享 generator 顺序、请求加入退出、不同采样参数、混合 greedy/采样、EOS/stop、真实 BF16 projection 形状、参数和全局计算设置不变，以及无法合批的原因统计。

`tests/test_native_independent_kv.py` 验证不同长度的 logits / 每层 KV 对照，覆盖投影合并、QKV bias、QK Norm、三种 attention 后端和并行残差；还检查 KV 无合组复制、共享组/独立缓存切换、采样停止、取消、部分更新失败以及过期位置视图拒绝。内部 `_IndependentLayerCache` 只作为单次前向的行引用；它的最大长度用于 RoPE 表范围，attention 使用每行实际长度。缓存视图重复使用会在修改 KV 前被拒绝。

HTTP 集成覆盖两个 OpenAI 文本端点、CPU FP32 / CUDA BF16、两种执行模式、默认采样、流式与非流式、usage、取消与清理。CLI 验证配置可往返，并拒绝未启用逐步合批的吞吐模式。

新增真实 loopback SSE 基准：

```powershell
python tools/benchmark_native_decode_modes.py --model YOUR_LOCAL_MODEL --tokens 32 --repeats 7 --fixed-length --output artifacts/modes_equal.json
python tools/benchmark_native_decode_modes.py --model YOUR_LOCAL_MODEL --tokens 32 --repeats 7 --fixed-length --prompts-file tools/fixtures/native_stream_mixed_prompts.json --output artifacts/modes_mixed.json
```

工具共享同一份本地模型，默认 temperature=0.7、top-p=0.9，使用相同请求许可上限，依次比较关闭合批、compatible、throughput。每种模式两次预热，七轮随机顺序测量；只绑定临时回环端口并在结束时关闭。计时包含 HTTP、prefill/decode、实际采样和文本输出，不含加载。`--fixed-length` 仅在该基准进程禁用 EOS，便于比较固定工作量；不传时按模型正常 EOS 结束，吞吐以实际 usage 计算。

报告保留失败请求、输出文本/usage、配对 token 速率、首个可见文本延迟、响应时长、文本块间隔、显存和健康状态。文本块间隔不是逐 token ITL。返回成功仅表示请求、清理、数值有限性和源码稳定性检查通过，不表示性能/质量门槛通过。

工具还对等长和不同长度的四个固定合成提示分别运行 48 步同历史 token 数值对照，报告 logits 最大/平均绝对误差、argmax 分歧与非有限值；这些只用于诊断，不替代真实任务质量评估。采样服务输出不会强行要求不同模式逐字相等。

采样与数值模式阶段回归为 **746 passed，0 skipped / failed / error**。加入不同长度支持后的最终扩展回归包含 **1009 项**：首次执行 **1008 passed、1 failed、0 skipped / error**；唯一失败是已有 GQA 离线微基准的子进程触发 30 秒超时。未改代码单独复跑该项，**1 passed（11.46 秒）**。全部 1009 项均有通过记录，但首次完整运行不是全绿；跨夜执行的总耗时不作为性能数据。

回归覆盖生成、缓存、流式协议、断连、模型生命周期、CLI、OpenAI 兼容和 wheel contract。Ruff 的 F601/F821/F822/F823 检查、版本一致性、生成函数索引检查通过。没有为本轮构建或部署新 wheel。

跨平台推理 CI 已加入采样合批、不同长度缓存与服务合批回归；全量质量检查也会自动发现新测试。CI 的 CPU runner 会跳过 CUDA 用例，本机上述回归包含实际 CUDA 执行，不能将 CI 配置等同于远程任务已运行。

本轮原始证据保存在 [验证目录](../artifacts/native_sampling_throughput_20260929)，包含阶段回归 `regression.xml`、最终回归 `regression_final.xml`、超时项复跑 `gqa_retry.xml`、修改前源码快照和对应文件差异。快照和差异只覆盖已记录基线的文件，不能代替整个工作区的提交清单。

## 本机服务测量：不同长度支持加入之前

RTX 4090 Laptop / Windows / Torch 2.6.0+cu124；同一已有 28M 检查点、BF16、服务默认投影合并开启。四并发、每请求固定输出 32 token、temperature=0.7、top-p=0.9、前缀缓存关闭。所有变体使用相同 admission 上限。吞吐是实际成功输出 token 除以总耗时，不是单请求速度。

首轮七组配对：

| 模式 | 聚合输出 token/s | 首个可见文本 P95 | 响应 P95 | 失败请求 |
| --- | --- | --- | --- | --- |
| 关闭合批 | 28.76 | 213.32 ms | 4733.05 ms | 0 |
| compatible | 35.64 | 240.43 ms | 3933.13 ms | 0 |
| throughput | 42.96 | 207.48 ms | 3219.33 ms | 0 |

配对速率中位数：关闭 → compatible 为 **1.235×**，compatible → throughput 为 **1.214×**，关闭 → throughput 为 **1.493×**；各项均 7/7 更快。兼容模式首字 P95 约退化 12.7%，不能声称所有指标均有改善。首轮没有出现 allocated 显存增长，所有变体请求完成后均清理干净。

独立进程复测同样七组：

| 模式 | 聚合输出 token/s | 首个可见文本 P95 | 响应 P95 | 失败请求 |
| --- | --- | --- | --- | --- |
| 关闭合批 | 28.80 | 207.94 ms | 4727.38 ms | 0 |
| compatible | 36.81 | 209.67 ms | 3566.10 ms | 0 |
| throughput | 42.48 | 215.82 ms | 3124.66 ms | 0 |

复测配对中位数分别为 **1.277× / 1.158× / 1.468×**，仍各自 7/7 更快。因此本负载中 compatible 相对关闭合批约 +23%～28%，throughput 约 +47%～49%；这不是 4B/8B 或不同长度请求的提升承诺。复测 throughput 首字 P95 相对关闭合批约慢 3.8%，兼容模式约慢 0.8%。两轮全部请求成功并释放驻留状态，均无 allocated 峰值增长；绝对显存值受进程初始状态影响，按同轮比较。

首轮完成后补了固定微批选项冲突校验，以及基准 CPU 显存统计和前后源码哈希检查；该阶段合批计算路径未改变。`sse_equal_repeat.json` 对应加入不同长度支持之前的运行库，且其测量前后源码一致。

同历史 token 数值诊断（4 请求 × 48 步）：compatible 的 logits 误差与 argmax 分歧均为 0；throughput 最大绝对误差 0.125，平均绝对误差约 0.00817，192 个请求步中 1 个 argmax 不同，全部为有限值。这不是实际任务质量下降/不变的证明，吞吐模式不能标注为逐 token 无损。

首轮健康统计显示实际主要是 B3 与 B1 交替，throughput 还出现 B2；并不是所有四并发请求都在 B4 中运行。这是加入独立 KV 路径之前的记录。

同阶段混合长度基准发现退化：关闭合批 28.29 token/s，compatible 23.61，throughput 23.69；配对速率中位数约 0.833 / 0.827，均 0/7 更快。该结果促使本轮继续实现不同 KV 长度的投影合批。失败实验保留在 `sse_mixed.json`，不能用前面的等长收益掩盖它。

## 不同 KV 长度支持后的测量

相同混合提示、四并发、默认采样、32-token 固定输出和七组配对。报告为 `sse_independent_mixed.json`：

| 模式 | 聚合输出 token/s | 首个可见文本 P95 | 响应 P95 | 失败请求 |
| --- | --- | --- | --- | --- |
| 关闭合批 | 27.35 | 514.67 ms | 5020.75 ms | 0 |
| compatible | 32.20 | 549.61 ms | 4037.69 ms | 0 |
| throughput | 36.77 | 434.00 ms | 3591.55 ms | 0 |

相对关闭合批的配对速率中位数为 **1.180× / 1.338×**，均 7/7 更快。兼容模式首字 P95 约慢 6.8%，吞吐模式约快 15.7%；所有变体均清理完毕，allocated 峰值未增长。每次完整基准中每种模式只有 28 个测量请求，P95 只描述本次小样本，不等于长时容量验收。

两种模式各实际执行 279 次独立 KV 前向，合组 KV 历史复制为 0；吞吐模式出现真实 B4 投影，也仍有 B3/单请求交替。该路径解决的是不同长度不能共享投影的问题，不会强迫未准备好或背压中的客户端入批。

新增不同长度的同历史数值对照：compatible 的 logits 误差及 argmax 分歧为 0；throughput 最大绝对误差 0.125，平均绝对误差约 0.00773，所测 192 步 argmax 无分歧。等长对照仍复现 1/192 的 argmax 差异，不能据此声称吞吐模式普遍无损。

2026-09-30 使用相同最终源码和配置，另启进程复测混合长度，报告为 `sse_independent_mixed_repeat.json`：

| 模式 | 聚合输出 token/s | 首个可见文本 P95 | 响应 P95 | 失败请求 |
| --- | --- | --- | --- | --- |
| 关闭合批 | 26.84 | 639.75 ms | 4920.90 ms | 0 |
| compatible | 28.00 | 559.13 ms | 4810.07 ms | 0 |
| throughput | 31.75 | 501.76 ms | 4310.76 ms | 0 |

配对速率中位数：关闭 → compatible 为 **1.035×**，compatible → throughput 为 **1.128×**，关闭 → throughput 为 **1.192×**，各项均 7/7 更快。最终源码两次混合长度基准中，吞吐模式相对关闭合批的配对提升为 **19%～34%**，兼容模式为 **3.5%～18%**；这是两次测量的范围，不是置信区间或收益承诺。第二次实际批大小还出现 B2，分布与第一次不同，不能只发布收益较高的那次。所有请求成功，状态清理完毕，allocated 峰值未增长；数值对照与第一次相同。

最终源码的等长请求复测为 `sse_independent_equal.json`，配置与前述等长基准相同：

| 模式 | 聚合输出 token/s | 首个可见文本 P95 | 响应 P95 | 失败请求 |
| --- | --- | --- | --- | --- |
| 关闭合批 | 23.99 | 236.80 ms | 5719.80 ms | 0 |
| compatible | 28.10 | 291.24 ms | 4803.65 ms | 0 |
| throughput | 32.07 | 316.21 ms | 4185.69 ms | 0 |

配对速率中位数分别为 **1.173× / 1.129× / 1.350×**，均 7/7 更快。吞吐模式相对关闭合批约提升 **35%**，但首个可见文本 P95 从 **236.80 ms 增至 316.21 ms（约 +34%）**；兼容模式也增至 291.24 ms。端到端响应 P95 降低，但不能把吞吐收益解释为首字延迟也改善。调度需要继续在目标延迟预算内验收；此处没有根据这组小样本改变默认开关。

最终三份基准使用相同源码哈希，且各自测量前后均未变化；共 **252 个正式测量 HTTP 请求，0 失败**，各模式完成后均无驻留请求、在途 step 或保留 KV 组，allocated 峰值未高于各自同轮关闭合批的结果。数值探针结果一致。汇总与源码指纹见 [verified_summary.json](../artifacts/native_sampling_throughput_20260929/verified_summary.json)。这不是长时压力、真实质量或目标服务器验收的替代品。

## 后续专业化工作

下一阶段根据目标 GPU profile 决定 paged attention 与 CUDA Graph：当前不同长度 attention 仍逐请求计算，没有页池、页表或按 GPU 实际空闲显存准入。还需要在目标模型与部署硬件上完成自然 EOS、多轮对话、持续到达压力、长上下文、真实质量集，以及满足延迟目标时的服务容量验收。以上项目没有由本轮测试自动覆盖。
