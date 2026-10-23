# 原生吞吐限制：控制变量诊断与下一步方案

后续前三项已落实，当前行为与实测见 [采样合批、显式吞吐模式与不同长度投影合批](NATIVE_SAMPLING_THROUGHPUT_20260929.md)。下文是实施前的诊断快照。

结论：当前有两类直接限制——请求难以组成有效批次，以及 BF16 合批后 Linear 仍逐行执行。本轮只增加诊断脚本和报告，没有修改运行库、服务配置或数值策略。

## 本轮实测

环境为本机 RTX 4090 Laptop GPU、PyTorch 2.6.0+cu124，已有 28M 检查点、BF16。四条不同内容但等长的 128-token 合成输入；固定生成长度，greedy，关闭 EOS。预热 prefill 和合组后，计时 24 个 decode 步。每个变体预热两次，七轮随机顺序配对；算子计数在单独的 profiler 运行中采集。未启用投影合并或额外 fused kernels。

此实验不含 HTTP、tokenizer、prefill 和初始化成本。GPU 为共享桌面设备，没有隔离其他负载。**只用同轮配对数据归因，不将绝对 token/s 与上一轮包含 prefill 的基准直接拼接；历史绝对速度差距也不能仅凭计时范围解释。** 28M 上的收益不能外推为 L20 / 4B / 8B 服务收益。

第一轮结果：

| 对照 | 结果 | 判断 |
| --- | --- | --- |
| B4 BF16 逐行 Linear → 批量 Linear，均逐行回传 token | 聚合吞吐 306.5 → 520.9 token/s；配对加速比 1.694，7/7 更快 | 逐行 Linear 路径是该测试中的重要瓶颈 |
| 同上，但两个变体均不逐 token 回传 | 配对加速比 1.700，7/7 更快 | Linear 收益不是由回传方式差异造成 |
| B4 逐行 Linear，每行回传 → 每个 decode 步统一回传 | 配对加速比 0.998，仅 2/7 更快 | 没有稳定收益，应降低优先级 |
| B1 每步回传 → 只在结束时取结果 | 配对加速比 1.013，5/7 更快 | 本轮回传不是主要限制；后者也不具备即时流式输出语义 |
| B4 每步 `aten::mm` 数量 | 284 → 71 | 真正批量 Linear 将该计数减少 75% |

独立进程重复同样的七轮对照：逐行回传条件下，B4 聚合吞吐 43.27 → 78.28 token/s，配对加速比 **1.816，7/7 更快**；不逐 token 回传时，配对加速比 1.841，7/7 更快。两轮算子计数相同，运行库源码哈希相同。两轮合计 14/14 对照支持逐行 Linear 的成本判断，但不能将幅度当成目标服务承诺。

**需要另查的基准稳定性问题：** 两进程的 B1 聚合吞吐分别约 145.8 和 21.9 token/s，B4 逐行路径约 306.5 和 43.3 token/s，绝对速度波动很大。两轮都只计预热后的 decode，所以这次跨进程差距不能归因于 prefill 计时范围。尚未定位是外部负载、功耗/调度状态还是其他进程级运行条件；没有证据将其归因于某一硬件问题。目标机器复验需同步采集 CPU/GPU 时间线、频率/功耗和其他负载，避免把绝对速度的变动当作代码收益。

复测中，逐行 Linear 的整批回传仍基本无收益（配对比 1.004，4/7 更快）；批量 GEMM 路径的整批回传约有 4% 收益，第一轮同一项约 0.2%。因此整批回传可留作次级优化，目前证据不足以将其列为主要限制。

本轮所测 token 对照一致，但只是四个短提示的有限样本。[历史 BF16 诊断](NATIVE_BF16_BATCH_FIX_20260929.md)已经复现过 B1 / B4 形状相关数值差异及生成分叉，不能据此删除现有保护或宣称批量 GEMM 无损。

逐请求 `.cpu()` 改为整批回传的变体只在诊断脚本中模拟，EOS/stop 关闭；不代表已实现完整流式交付、停止和取消处理。

## 合批资格也确实在阻断吞吐

对四请求做一次实际 runner 解码：

| 输入条件 | 合批 forward | 逐请求 step |
| --- | --- | --- |
| greedy、等上下文长度 | 1 | 0 |
| temperature=0.7、等上下文长度 | 0 | 4 |
| greedy、上下文长度 32/64/96/128 | 0 | 4 |

这些是本轮执行统计，不是仅从代码推测。

源码原因：

- [NativeBatchDecode.py](../saddlellm/models/NativeBatchDecode.py) 的 `_key()` 排除 `session.do_sample`，返回的分组 key 含 `cache.length`。
- [InferenceServer.py](../saddlellm/runtime/InferenceServer.py) 的 SSE 入口仅在 `temperature == 0` 或启用 chunked prefill 时绑定 step batcher；开启 chunked prefill 也不会解除执行器对采样 decode 的排除。
- 接口默认 temperature 为 0.7；服务设置默认 `native_stream_batching=False`、`batch_size=1`、`max_concurrency=1`。这是源码默认值，未检查或断言正在运行的服务使用这些值。
- [SaddleModeling.py](../saddlellm/models/SaddleModeling.py) 的 `_rowwise_decode_linear()` 对每行分别执行 projection 后 `cat`，覆盖 attention 和 MLP 投影及 lm_head。

因此，增加服务并发数、排队长度或调度器 batch size，本身不能消除以上执行限制。上一轮减少 KV 合组复制对 B3/B1 有收益，但没有改变它们。

## 建议实施顺序

1. **先让采样请求共享前向。** 同时修改服务入口与 runner 资格判断；模型前向合批，各请求独立处理 temperature、top-k/top-p、重复惩罚、EOS、stop 与生成预算。显式 seed 使用请求自己的 generator。没有 seed 或共享 generator 的请求需要明确随机数消费顺序，不能承诺改变调度后文本逐 token 一致。初版保留现有逐行 BF16 数值路径，单独验证采样合批收益。

2. **提供显式 BF16 吞吐模式，使用真正批量 Linear。** 保留现有兼容模式，以便需要保持已验证 B1 路径的任务选择。吞吐模式按同历史 token 的 logits 误差、argmax 分叉、实际任务质量和服务吞吐分别验收。不能把“没通过逐 token 相等”直接等同于质量下降，也不能把本轮 token 相等当作普遍证明。优先在目标大模型、目标 GPU 和真实采样参数上复验。

3. **拆开投影合批与 attention 缓存约束，先支持不同长度。** 可做一个过渡执行器：把各请求当前 token 的 hidden states 合起来执行 QKV / 输出投影 / MLP / lm_head，按各自 position 应用 RoPE，attention 暂时逐请求读写各自 KV，再汇合 hidden states。这样无需先把所有历史 KV 复制成一个等长大张量，也不必等待完整分页引擎才能研究混合长度吞吐。它仍保留逐请求 attention 开销，需要新的逐层执行接口与正确性验证；不是删掉长度判断即可完成。随后以测量结果决定是否替换为 paged / ragged attention。

4. **之后再决定 CUDA Graph 和进一步融合。** 先确认实际 batch 大小、Linear 执行模式和目标模型 profile。若 CPU 发射开销仍占主导，再引入固定输入/缓存地址、有限 batch 档位的 graph 路径，单独统计录图成本、额外显存和回退情况。当前本机对照不支持优先投资整批 token 回传。

每一步都记录有效 batch 大小、回退原因、合批 token 比例、实际完成 token/s，以及 TTFT / ITL 的延迟分布。验收覆盖 greedy / 默认采样、相同 / 不同输入长度、C1 / C4 / C8、请求加入/退出/取消。服务层吞吐需要另外测量，不能用本报告的纯 decode 数字替代。

原始结果和复现脚本位于 [诊断目录](../artifacts/native_throughput_diagnosis_20260929)。
