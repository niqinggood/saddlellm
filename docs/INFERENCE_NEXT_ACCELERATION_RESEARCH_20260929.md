# SaddleLLM 下一轮推理加速调研（2026-09-29）

后续：已完成[第一批增量优化与本机验证](NATIVE_INCREMENTAL_OPTIMIZATION_20260929.md)。本文保留调研当时的源码诊断与建议，尚未实现项不能由后续三项优化推断为已交付。

后续进展：[采样合批、不同 KV 长度投影合批与显式 BF16 吞吐模式](NATIVE_SAMPLING_THROUGHPUT_20260929.md)已实现；下文相关限制为调研时的历史快照。

结论：原生推理还有明确的优化空间。短期先解决 KV 合组反复复制和采样请求不能真正合批；单请求延迟重点验证 CUDA Graph；并发吞吐的主线是分页 KV、任意上下文长度 attention 和真正的 BF16 批量 Linear。前缀共享、量化与投机解码按负载需求继续推进。

本次核对了当前工作区源码、已有 L20 实验和官方资料，并运行一个 CPU FP32 最小诊断。本文中的 GPU 性能数字来自保存的历史实验；本次没有重跑 GPU/HTTP 性能测试，也没有改动运行库或部署服务。优先级和潜在收益是工程判断，尚未测量的方案不承诺加速倍数。

**已有基础及本次最重要的发现**

- 已有：几何增长 KV、窗口缓存、最后位置 logits、投影合并、受限 GQA decode、W8A16、前缀快照、等上下文 SSE 合批、RMSNorm/RoPE 局部融合、token 预算调度、分块 prefill。它们的支持范围不同，不能默认组合全部开关。
- 最近保存的融合实验中，Qwen3-4B / L20 / BF16 SSE 单请求为 48.85 → 56.45 输出 token/s；C4 同长度为 69.57 → 75.35，混合长度为 53.75 → 62.14。来源：[融合实验](../artifacts/native_acceleration_20260929/REPORT.md)。
- 融合后 16 步 decode 的 profiler 仍记录 15,136 次 runtime launch 和 2,896 次 driver launch，合计每 token 1,127 次。`aten::mm` 占该次记录 Self CUDA 时间的 86.61%，FlashAttention 约 2.05%。这是短输入、单请求的诊断，说明 launch 和线性层值得优先分析；不能外推为长上下文的耗时分布，也不能将 profiler 时间直接当实际服务性能。来源：[原始 profile](../artifacts/native_acceleration_20260929/remote_results/profile_fused.json)。
- 历史同运行库对照中，vLLM 对当时原生实现的优势从 C1 约 2 倍扩大到 C4 混合长度约 6.8–7.1 倍。这支持优先检查批执行能力，但比较发生在最新融合改动之前，不能当作当前差距；跨引擎回答也有不一致。来源：[原始比较](../artifacts/remote_native_compare_20260929_1328/REPORT.md)。

| 优先级 | 工作 | 主要受益场景 | 改动成本 | 判断依据 |
| --- | --- | --- | --- | --- |
| P0 | 保留未被当前 step 修改的有效 KV cohort | B3/B1 交替、请求进出和异步 SSE | 小到中 | 当前源码与本次最小复现均确认重复合组 |
| P1 | 采样请求共享模型前向；合并 token 后处理与回传 | 默认 temperature=0.7 的实际聊天 | 中 | 目前采样被 batch runner 排除；每行独立采样/回传 |
| P1 | BF16 真正批量 Linear 的独立数值策略 | B2/B4/B8 并发吞吐 | 中到大 | 当前所有受保护 Linear 使用 B1 循环 |
| P1 | 分页 KV + 任意长度 attention + 显存准入 | 不同 prompt 长度、长会话、高并发 | 大 | 当前按 KV 长度分组，历史 KV 仍需物理拼组 |
| P1 | CUDA Graph：先局部原型，再稳定缓存上的完整 decode | C1、低并发、每 token 延迟 | 中到大 | 每 token 仍有约 1,127 次 kernel launch |
| P2 | 页级前缀共享，与 chunked prefill 组合 | 长系统提示、重复文档、多轮对话 | 中到大 | 现有前缀缓存命中后复制，且与分块 prefill 互斥 |
| P2 | 自适应 prefill 预算与公平调度 | 长短请求混合、首字与 ITL 尾延迟 | 中 | 分块已能降低阻塞，但会增加长请求首字时间 |
| P2 | W4A16/FP8 与增量投机解码 | 带宽受限的 8B、低并发长输出 | 大，且需单独质量验证 | 有相关基础，但尚无成熟的原生组合执行路径 |

**1. 最先做：消除 KV 合组抖动**

定位：[NativeBatchDecode.py](../saddlellm/models/NativeBatchDecode.py) 的 `step()`，当前第 186 行每次把 `self._cohorts` 换为空字典，仅保存本次参与批量前向的组。独立 B1 调用不会保存未参与的 B3 组，下一次 B3 调用重新复制历史。

本次在 Torch 2.6.0+cu124 的 CPU FP32 上，用两层随机小模型、4 个会话做确定性诊断。所有会话先生成一个 token，然后比较 B3 连续推进 5 次与 B3/B1 交替推进 5 次：

| 指标 | B3 连续 | B3/B1 交替 |
| --- | ---: | ---: |
| B3 实际批量前向次数 | 5 | 5 |
| cohort 创建次数 | 1 | 5 |
| cohort 累计历史复制 bytes | 2,304 | 19,200 |
| 每轮结束 retained_cohorts | 1 | 0 |
| 三条 B3 请求的输出 | 两种执行顺序相同 | 两种执行顺序相同 |

复制计数约增至 8.33 倍。这不是速度比，计数也不包含所有缓存扩容和临时分配。历史远程测试已出现 423 次组创建、8,981,839,872 bytes 累计历史复制，与这个机制吻合，见 [SSE 验证](REMOTE_QWEN3_4B_SSE_VALIDATION_20260929.md)。

建议维护有界的活跃 cohort 表：本轮未触及且成员/长度/存储仍有效的组可保留；成员单独推进、换组、退出、取消或异常时使涉及的组失效。必须核算保留组的实际存储，避免引用维持过多旧 buffer；已有 `forget()` 与关闭流程需要一起检查。验收场景应包括 B3/B1、B2/B2 交替、成员交叉重组、扩容、异常及取消。

**2. 让默认采样聊天真正用上合批**

定位：[NativeBatchDecode.py](../saddlellm/models/NativeBatchDecode.py) 的 `_key()` 明确排除 `session.do_sample`；[InferenceServer.py](../saddlellm/runtime/InferenceServer.py) 的请求模型默认 `temperature=0.7`。启用 token 预算调度能让采样进入同一协调器，但不会改变这个模型前向限制。

第一阶段只让合格的采样请求共用前向，仍各自使用原有 generator、温度、top-p、惩罚和停止逻辑。greedy 与采样的差别主要在 logits 后处理，模型前向的共享值得单独实现和验证。第二阶段再考虑批量 GPU 采样。

另一个可测点在 [NativeGeneration.py](../saddlellm/models/NativeGeneration.py) 的 `_step_locked()`：batch runner 逐行调用采样和 `token.detach().cpu()`。可以先收集各行 token，再一次传回 CPU 并分发 streamer，减少逐行同步；需要明确 CUDA event、缓冲区寿命和输出交付顺序。不能为了少回传而攒多 token，损害首字或流式延迟。

验收保留每请求 RNG 状态隔离，检查改变到达顺序、batch 大小、另一个请求提前结束时的 seed 行为。优化采样算法若改变随机数消费规则，应单独标明，不能只检查几条文本相同。

**3. BF16 真正批量矩阵计算是吞吐主线**

定位：[SaddleModeling.py](../saddlellm/models/SaddleModeling.py) 的 `_rowwise_decode_linear()`，当前约第 332 行，对 batch 每一行分别调用 `projection()`，再 `torch.cat()`。Q/K/V、输出投影、MLP 及 lm_head 都有受保护路径。因此当前 BF16 合批主要共享 attention、norm、激活等计算，没有充分发挥 B×K 矩阵乘法对权重的复用。

建议明确两种执行合约：保留现有参考路径；新增显式吞吐路径，使用真正批量 GEMM，按 logits 误差、首次 greedy 分叉、任务质量和吞吐分别验收。若需求是跨 batch 一致，进一步评估固定归约方式和 batch-invariant kernels。相对于 B1 PyTorch 路径的逐位一致，与新 kernel 自身的跨 batch 一致，是两个不同目标。

vLLM 已有 beta 的 batch invariance 模式，使用确定性算子并接受性能代价；这是值得参考的设计方向，不是 Saddle 可直接复用的开关。[官方说明](https://docs.vllm.ai/en/stable/features/batch_invariance/)。现有 BF16 保护是针对已观察到的数值分叉，不应直接删掉保护再把差异称为无损加速。

**4. 分页 KV 与任意长度 attention 一起设计**

当前 runner 的分组 key 包括 `cache.length`，不同长度不能真正同批 attention。分块 prefill 的 `_execute_chunked()` 先执行 decode，再逐个执行 prompt chunk，也还不是统一的混合 GPU batch。

建议形成一组共同接口：页池、页表、每请求 seq_len/position、slot mapping、分配/回收/引用计数及真实显存准入。attention 直接读页表，各请求历史 KV 无需为合批拼成新的连续大张量。先支持已验证的稠密 GQA 与纯 decode，再接 chunked prefill 和混合批。

可先评估独立算子库 FlashInfer 的 paged decode/prefill 接口，保留 Saddle 的模型、调度和服务控制；另做受限 Triton 原型作为可控备选。接算子仍涉及权重/布局、RoPE、dtype、kernel 支持、版本和平台验证，不等于装一个库即可完成。原生 Windows 与 Linux/CUDA 应保留各自清楚的支持范围。[FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html)。

allocator 和 attention consumer 必须同时可用才有完整分页执行能力。分页的主要直接收益是容量与组合执行条件；单用户短上下文不一定更快。只做 padded KV 拼接可作为对照，不能误认为已经去掉了历史复制成本。

**5. CUDA Graph 值得提前做受限原型**

现有 profiler 给了明确动机，但动态缓存地址、每 token 改变的可见 KV shape、Python 长度和 `.cpu()`/`.item()` 都使整段 session 难以直接捕获。PyTorch 文档要求图重放保持 kernel 参数/地址等约束，动态形状可能导致多次录图。[PyTorch CUDA Graph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.md)。

不必等整个分页引擎完成才研究：先用 B1 的固定地址输入/工作区，对稳定形状的 Linear、norm、MLP 做 piecewise graph 原型，attention 和 CPU 控制留在图外。比较消除 launch 的收益是否足以抵消图边界拷贝。完整 decode graph 则等待静态/分页 KV 接口，建立 B1/B2/B4/B8 有界桶，动态 seq_len 用设备张量驱动，采样和输出交付明确边界。

vLLM 将完整 decode graph 与其他阶段的 piecewise graph 分开处理，说明分阶段实现有工程依据。[vLLM CUDA Graph 设计](https://docs.vllm.ai/en/stable/design/cuda_graphs/)。FlashInfer 的 graph wrapper 要求 batch 大小固定；固定 split 的确定性选项也不保证所有变长场景都兼容 CUDA Graph，必须验证组合。[FlashInfer 接口约束](https://docs.flashinfer.ai/api/attention.html)。

验收记录 graph 命中/回退、录图时间、额外显存、不同长度/桶切换的正确性及 C1 ITL。把编译启动时间与稳态分开，不把 padding 槽算作有效 token。不要把 `torch.compile(model)` 本身当作交付结果。

**6. 前缀共享与分块调度继续做实**

现有 [NativePrefixCache.py](../saddlellm/models/NativePrefixCache.py) 保存快照，命中后恢复私有缓存；分块 prefill 的资格判断和服务设置明确排除 prefix cache。下一步应先统一前缀已缓存长度与 prefill 游标，再在分页接口上支持不可变整页共享、尾页写时复制、引用计数/LRU。共享 key 要包含模型/adapter 和影响 KV 的配置；按 token 完全一致的前缀复用，不能对相似文字复用。

这适合长系统提示、同一文档、多轮共享前缀。RAG 内容不同的请求未必有较高命中率。前缀缓存主要减少 prefill，对生成每个新 token 的 decode 并不直接提速。[vLLM APC 说明](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/)。

分块大小也值得按目标负载调优。现有 4096-token 受控碰撞实验中，256-token 分块使短请求等待 416.48 → 41.34 ms，同时长请求首 token 398.24 → 639.67 ms。建议以 ITL/TTFT 时限和队列年龄控制预算，空闲时放大 chunk，忙时给 decode 留预算；另测试持续 decode 压力下 prefill 的最低进展。该实验不是 HTTP P95，见 [调度报告](../artifacts/native_engine_scheduler_20260929/REPORT.md)。

**7. 量化和投机解码的落点**

量化可评估 W4A16、FP8 W8A8，以及长上下文的低精度 KV。现有 W8A16 不等同于已利用低精度 Tensor Core；权重压缩、计算 dtype、反量化成本、带宽节省要分别测量。官方 vLLM 支持表可用于筛选目标硬件的候选路径，不能据此推断 Saddle 已兼容。[量化硬件支持](https://docs.vllm.ai/en/stable/features/quantization/)。先补齐现有 8B 合批数值诊断、量化格式加载和适配，再决定新增格式；量化包持久化主要缩短启动，不等同于稳态 token/s 提升。

现有 `generate_mtp_speculative()` 在 [SaddleModeling.py](../saddlellm/models/SaddleModeling.py) 约第 2195 行，使用 `self(out)`、完整 candidate verify，没有与 generation session 共享增量 KV，也没有拒绝后截断 KV 的执行接口。它目前是算法实验，不能当作可直接打开的生产加速。

更小的探索是针对复制/抽取类负载做 n-gram/prompt lookup 草稿；通用生成再评估兼容的 draft model/EAGLE/MTP。需要单次多 token verify、接受前缀提交、拒绝部分回滚、EOS/stop/budget 处理；采样还要正确的拒绝采样以保留目标分布。先测接受长度和草稿成本，再决定是否继续。vLLM 官方将降低中低 QPS、带宽受限工作负载的 ITL 作为投机解码的典型用途。[投机解码说明](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

**进一步源码检查：可以小步落地的优化**

以下是在用户要求继续检查后补充的发现。前两项还运行了 CPU 算子/shape 诊断；其余为源码确认的成本机制，实际性能收益待测。

| 项目 | 当前证据 | 建议及边界 |
| --- | --- | --- |
| 稠密推理不再创建无用辅助损失 | `SaddleDecoderLayer.forward()` 每层调用 `residual.new_zeros(())`；仅在有 labels 时才汇总 aux_loss。历史 36 层/16 步 profile 恰有 576 次 `new_zeros` | 仅在内部无 labels 的稠密推理路径省略辅助损失 tensor；保留训练、MoE、独立 layer 调用的返回合约。是低成本 launch 优化，不应承诺大幅提速 |
| prefill 末尾归一化裁剪 | `_prefill_only=True` 仍执行 `self.norm(hidden_states)`；最终 prompt chunk 也是先归一化全部 token，再切最后位置做 lm_head | 中间块可在明确不返回 hidden_states 的内部路径省去末尾 norm；最终块只对要投影的位置归一化。现有测试明确要求普通 `logits_to_keep` 保持完整 hidden_states，不能直接在公共 forward 中改 shape |
| 更深的只建 KV prefill 路径 | 中间块最后一个 decoder layer 在保存该层 K/V 后，仍完成 attention 输出和 MLP，而输出不会被 session 使用 | 对纯稠密、无 AttnRes/自定义 hook 的内部路径，可研究最后一层存好 KV 后提前结束。必须保留前面全部层及最后层的 K/V 计算；是受限候选，尚未实现/测量 |
| 推理资格检查提前固定 | `_key()` 在 BF16 每请求每步遍历层、`model.modules()`、类型和配置；forward 又检查部分相同静态属性 | 将不可变模型能力整理为加载/注册时执行计划，热路径保留请求长度、buffer 和状态检查。训练切换、adapter、hook、dtype、backend 改动必须显式失效，不能删掉防护 |
| top-p 全词表排序 | `_sample()` 在 `top_p<1` 时使用完整 `torch.sort`、softmax、cumsum、scatter，然后再 softmax/multinomial | 评估融合采样 kernel 或精确的小候选集路径；需要保留 top-k/top-p 应用顺序、阈值并列值、异常分布和 RNG 合约。不要只把 token 采样改为固定 top-k 截断 |
| 增量 detokenization 与输出线程交接 | 服务继承 Transformers `TextStreamer`；本机依赖的 `put()` 每 token 解码 `token_cache`，该缓存到换行才清空；服务还逐片段通过 executor 拉取 | 长无换行输出存在累计重复解码成本。可做 tokenizer 专属的有界增量 decoder，加上有界输出队列，减少 CPU 重复工作/交接。字节回退、中文、空白清理、stop 跨片段、背压与断连都要验收，不能简单逐 token decode 后拼接 |
| 前缀查找的 CPU 成本 | 每次 restore 做全模型 fingerprint，并从 GPU token `.tolist()`；partial lookup 会扫描缓存条目并逐 token 比较 | 从准备请求时保留 CPU token IDs；在不可变服务模型上用显式版本戳，配合 hash/radix 索引。版本戳必须覆盖权重/adapter/位置编码等变更，不能用过期 prefix |
| 进一步融合/预分配 | 仍有 residual add、SwiGLU 乘法、KV 写入和 token 输入 `cat` | 先根据新的 profile 做 residual+norm、SiLU×up、RoPE+KV 写入候选；固定输入及输出 buffer 配合 graph。融合可能改变舍入边界，需复用已建立的数值验证方法 |

本次两层小模型诊断结果：一个 decode step 有 2 次 `aten::new_zeros` 与 2 次 `aten::zero_`；16-token prompt 分成两个 8-token chunk 时，两个 chunk 都把 `[1, 8, 32]` 输入末尾 norm，即使第一块不产生 logits。这些是运算存在性的证明，没有做 GPU 加速测量。

top-k/top-p 无显式排序的融合采样已有官方算子接口，可作为实现参考：[FlashInfer sampling API](https://docs.flashinfer.ai/generated/flashinfer.sampling.top_k_top_p_sampling_from_logits.html)。接口包含 seed/offset 不代表输出必然与当前 PyTorch multinomial 在相同 seed 下逐 token 相同；采用新实现需要独立的兼容性定义。

**推荐执行顺序与实验口径**

1. 先修 cohort 保留、稠密辅助零张量及受限 prefill 末尾计算，分别做当前融合+调度版本上的 L20 A/B；同一轮建立真实采样负载基线。
2. 做采样前向共享、批量 token 回传和 BF16 GEMM 数值策略的独立实验。各项单独开关，便于归因。
3. 设计分页 KV/attention 共同接口；期间可单独验证 B1 piecewise CUDA Graph 的价值。
4. 稳定后接任意长度 decode、按显存准入、完整 decode graph，再加入页级 prefix 共享与混合 prefill/decode。
5. 只有当目标负载仍受权重带宽或 decode 次数限制时，继续量化/投机解码。

若近期目标是业务立即受益，可在项目已有的 vLLM 后端上做同权重、同精度、同 thinking 与采样参数的应用验证；这与 Saddle 原生引擎研发是两项交付，速度归属应分别记录。

实验应覆盖 C1/C4/C8（显存允许时）、真实到达间隔、全部不同长度的 128/512/2048/4096-token prompt、greedy 与 temperature=0.7、缓存冷/热和真实多轮请求。现有 25/23/36/36 的混合短提示集不足以代表这些负载。统计实际完成 token/s、GPU 首 token 与用户可见首字、ITL P50/P95/P99、队列等待、显存/复制 bytes、错误/取消及请求/KV 清理；持续流量测试用满足延迟目标的完成请求数衡量服务能力。

性能测试与质量测试分别运行：性能可以控制长度，质量保留 EOS 和完整回答；逐位一致、误差容差、跨 batch 可复现、任务质量各自有合约。先延续项目已有 7 对随机顺序 A/B 门槛，再对值得发布的方案增加足够请求数的尾延迟和持续负载测试。历史融合、调度、量化实验的收益不能相乘。

**最小复现与源码锚点**

以下 Python 可在项目根目录运行，只使用 CPU、小型随机模型，不加载已有模型或启动服务：

```python
import torch
from contextlib import ExitStack
from saddlellm.models.SaddleModeling import SaddleForCausalLM, SaddleModelConfig
from saddlellm.models.NativeBatchDecode import NativeBatchDecodeRunner

torch.set_num_threads(1)
torch.manual_seed(173)
model = SaddleForCausalLM(SaddleModelConfig(
    vocab_size=37, hidden_size=32, intermediate_size=64,
    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    max_position_embeddings=128, attention_kind="gqa",
    attention_backend="sdpa", eos_token_id=[],
)).eval()
outputs = []
for interleave in (False, True):
    with ExitStack() as stack:
        sessions = [stack.enter_context(model.create_generation_session(
            torch.tensor([ids]), max_new_tokens=8,
            do_sample=False, eos_token_id=[],
        )) for ids in ([4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15])]
        runner = stack.enter_context(NativeBatchDecodeRunner(model))
        for session in sessions:
            session.step()
        for _ in range(5):
            runner.step(sessions[:3])
            if interleave:
                runner.step(sessions[3:])
        print(interleave, runner.snapshot())
        outputs.append([s.sequences().tolist() for s in sessions[:3]])
assert outputs[0] == outputs[1]
```

诊断时 SHA256：`NativeBatchDecode.py` 为 `EBDD05DD25C41DA68785789B741E7D5C1F2B18E346E6A77873984511D068A6D4`；`SaddleModeling.py` 为 `B646784ABA46CAED0F67C53DAEAC573327262007B15C1A990DCDE94A5B89932F`。源码继续变化后，以上行号和诊断结果应重新核对。
