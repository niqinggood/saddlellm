# 大模型面试知识笔记

这份笔记覆盖模型原理、数据与预训练、后训练、模型压缩、推理服务、RAG、Agent 和评测。每个问题都应能讲清用途、机制、取舍和验证办法。对应到本仓库的实现状态、源码证据及待改进项，见 [SaddleLLM 实现对照](LLM_INTERVIEW_FRAMEWORK_AUDIT_ZH.md)。

题目范围参考了公开的 [LLM 面试知识目录](https://github.com/wdndev/llm_interview_note)。技术解释按下文原始论文与官方资料核对；题库里的公司题源、频率排名和效果承诺不作为事实依据。整理日期为 2026 年 10 月 18 日。

## 模型基础

### Q01 Token 和 tokenizer 怎么理解

Token 是模型处理的离散单元，可以是子词、字节或字符，不能直接等同于一个汉字或单词。Tokenizer 决定文本与 token ID 的映射；BPE 根据语料中的相邻单元频率逐步合并，Unigram 和 WordPiece 使用不同的词表构建规则。参见 [BPE 子词分词论文](https://arxiv.org/abs/1508.07909)。

工程上检查文本往返、UNK、特殊 token、中文 token 长度和词表一致性。词表大小相同，不代表 token ID 语义一致；换 tokenizer 后，原模型 embedding 和蒸馏 logits 的含义可能不再对应。

### Q02 Causal LM 的训练目标是什么

目标是在已有前缀条件下预测下一个 token：`L = -mean(log p(x_t | x_<t))`。输入位置 t 的 logits 对应位置 t+1 的标签，因此通常要做 shift。`labels=-100` 表示该目标位置不计入 CE；它不会自动禁止注意力读取该位置。

核验时手算一个短序列：第一个被监督的答案 token 应由它前面的上下文预测；padding、prompt 和跨文档边界是否进入损失，应有明确规则。

### Q03 Decoder only 为什么适合生成

Decoder only 用 causal mask 实现自回归建模，预训练和生成共享“给定前缀预测后续”的接口。Encoder 擅长双向表示，encoder-decoder 适合显式输入输出转换。任务、数据和推理约束决定选择，不能把 decoder only 当成所有任务的最优结构。

训练时可并行计算所有位置的 logits，生成时下一步输入依赖刚生成的 token；这两种并行性要区分。

### Q04 Self Attention 怎么计算

```text
Q = X Wq, K = X Wk, V = X Wv
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k) + mask) V
```

缩放控制点积随维度增大而产生的数值尺度。多头让不同子空间分别建模依赖。标准 dense attention 的训练/prefill 注意力计算随序列长度呈二次增长；模型总耗时还包括投影和 FFN。参见 [Transformer 原始论文](https://arxiv.org/abs/1706.03762)。

手写时最常见的错误是 softmax 轴、张量转置、mask 广播和输出 head 合并不一致。

### Q05 Causal mask 和 padding mask 有什么区别

Causal mask 防止当前位置读取未来信息；padding mask 屏蔽填充位置；loss mask 决定哪些目标参与优化。这三者职责不同。Cache 解码还要区分 query 长度和完整 KV 长度，不能把非方阵 causal mask 当作训练时的方阵照搬。

位置重置、插入 EOS 和设置 `labels=-100` 都不能单独证明两个拼接文档的注意力已隔离。

### Q06 MHA MQA GQA 有什么区别

| 结构 | Query 头 | KV 头 | 主要取舍 |
|---|---:|---:|---|
| MHA | H | H | 每个 query 头有对应 KV 头 |
| MQA | H | 1 | KV 最省，表示容量和并行分片方式需评估 |
| GQA | H | G | 每组 query 共享 KV，折中缓存开销和容量 |

GQA 缓存应按 KV 头数计算；若先把 KV 复制为 H 个头再长期保存，就失去了预期的缓存节省。参见 [GQA 论文](https://arxiv.org/abs/2305.13245)。

### Q07 RMSNorm Pre Norm 和 SwiGLU 分别解决什么

RMSNorm 按均方根归一化，不做 LayerNorm 的均值中心化；Pre Norm 在子层前归一化，改变残差路径上的梯度传播。SwiGLU 是带门控的 FFN，常见形式为 `down(silu(gate(x)) * up(x))`。它们分别涉及数值稳定性和表达方式，不能只根据名称承诺模型质量提升。参见 [RMSNorm](https://arxiv.org/abs/1910.07467) 和 [GLU 变体](https://arxiv.org/abs/2002.05202)。

### Q08 RoPE 为什么能表达相对位置

RoPE 按位置旋转 Q/K 的二维子空间，使它们的点积包含相对位置信息。它并不直接扩充模型见过的长距离分布；改变频率、缩放系数或最大长度后，仍要验证长序列质量。参见 [RoFormer](https://arxiv.org/abs/2104.09864)。

检查 prefill 与 decode 使用的绝对位置、cache offset、padding 和滑窗位置是否一致。RoPE、YaRN、NTK scaling 等名称不能互相替代。

### Q09 MoE 为什么参数很多但计算不一定同比增加

路由器为每个 token 选择少量专家，因此总参数量和每 token 激活参数量不同。代价包括专家权重驻留、路由与负载均衡、容量溢出和跨卡通信；稀疏激活不会自动降低总显存或端到端延迟。参见 [Switch Transformer](https://arxiv.org/abs/2101.03961)。

核验真实路由、专家梯度、负载统计和通信；“配置了专家数”还不能说明专家并行已经工作。

## 数据和预训练

### Q10 数据质量为什么不能用数量代替

重复、错误、低质量模板和 benchmark 泄漏都可能使 loss 更漂亮、泛化更差。应保留来源、许可、版本、完整内容指纹和清洗统计，并分别检查精确重复、近重复和同来源分组。

精确去重应基于完整规范化内容；长文只截一段去重会误删。不同文件路径也不证明 train、validation、test 的内容独立。

### Q11 Sequence packing 有什么收益和风险

把短样本装进长序列可以减少 padding，提高有效 token 占比。需要明确：是否允许跨文档注意力、是否插 EOS、位置是否重置、跨边界预测是否进入 CE，以及尾块如何处理。

若目标是文档独立训练，应同时验证注意力隔离和边界 loss mask；还要确认 Trainer 没有把文档 ID 等列过滤掉。

### Q12 Scaling law 和 Chinchilla 应该怎么用

Scaling law 用实验拟合模型规模、训练 token 和 loss 的关系。常见预算近似为 `C ≈ 6 N D`，但它不能覆盖所有架构和长上下文开销。Chinchilla 研究说明，在其条件下增加模型规模时也应相应增加训练数据；它不是所有项目通用的固定 token 倍率。参见 [计算最优训练研究](https://arxiv.org/abs/2203.15556)。

工程上先做小规模实验网格，再拟合、检查残差和外推范围；还要把部署成本、数据重复率和训练时长加入选型。

### Q13 BF16 FP16 和梯度检查点各节省什么

混合精度降低部分张量的存储和计算开销；FP16 的数值范围较小，通常需要关注 loss scaling，BF16 也仍需监控溢出和不稳定。梯度检查点少存前向激活、在反向时重算，用计算换激活显存。

显存应拆分为权重、梯度、优化器状态、激活和临时工作区。仅用“参数量乘两字节”只能估计一部分推理权重存储。

### Q14 梯度累积等不等于更大的 batch

常见有效 batch 为 `单卡 microbatch × 数据并行进程数 × 累积步数`。正确实现需要一致的损失归约、同步时机、梯度裁剪和 optimizer step；变长序列按样本平均与按有效 token 平均可能产生不同权重。

核验“小 batch 累积”与“大 batch 一步”的参数更新，而不只检查 loss 除了几次。Dropout 等随机性会影响严格等价条件。

### Q15 DDP FSDP ZeRO TP PP 怎么区分

| 方法 | 主要分工 |
|---|---|
| DDP | 每个进程保留模型副本，切数据并同步梯度 |
| ZeRO 1 | 分片优化器状态 |
| ZeRO 2 | 进一步分片梯度 |
| ZeRO 3 | 进一步分片参数 |
| FSDP | 按包装和分片策略管理参数等状态，实际行为取决于配置 |
| TP | 切分层内矩阵计算 |
| PP | 将不同层分到不同 stage，以 microbatch 调度 |

ZeRO 的目的在于消除冗余状态，参见 [ZeRO 论文](https://arxiv.org/abs/1910.02054)。选择时同时考虑显存、通信、计算粒度和 checkpoint 格式。`device_map="auto"` 的推理分配不能直接当作分布式训练策略。

## SFT 和能力保持

### Q16 SFT 到底训练了什么

SFT 仍然可以使用 next-token CE，但监督文本来自指令、对话和任务样例。Completion-only 配方通常只优化答案 token；多轮对话要决定是否监督全部 assistant turn。应按真实 chat template、EOS 和截断后的 token 构造 labels。

答案被截没、prompt 被误监督、EOS 丢失、pad 与 EOS 共用却掩码错误，都会造成“能训练但目标不对”。

### Q17 LoRA 为什么省资源

LoRA 冻结原权重，训练低秩增量：`W' = W + (alpha/r) B A`。低秩约束施加在增量上，不要求原矩阵本身低秩。它主要节省可训练参数及其梯度、优化器状态；激活等开销仍然存在。参见 [LoRA](https://arxiv.org/abs/2106.09685)。

Rank、target modules、学习率和数据影响结果。LoRA 不是天然的防遗忘或质量保证；合并后通常可消除独立 adapter 的额外计算路径。

### Q18 QLoRA 和 QAT 为什么不能画等号

QLoRA 在冻结的量化基座上训练低秩 adapter。原论文包含 NF4、double quantization 和 paged optimizer 等节省资源的设计。它主要回答“怎样省资源地微调”，不自动保证最终部署格式的数值误差已被训练匹配。参见 [QLoRA](https://arxiv.org/abs/2305.14314)。

QAT 则显式模拟目标量化数值；adapter 合并、再量化、换执行内核后都需要重新检查能力。

### Q19 SFT 为什么会遗忘 怎么补救

新任务梯度可能改变旧任务所需的表示或输出分布；数据过窄、学习率过大、更新过久和模板变化都可能加重退化。可以混入需要保留的通用/历史任务，降低更新强度，或在保留集上加入参考模型约束。

例如使用 `CE + λ T² KL(p_reference || p_student)` 时，必须说明 KL 的方向、温度、监督位置、归约方式及教师是否冻结。词表不一致不能直接对齐 logits。最终以“新任务收益＋旧任务退化预算”选择模型。

### Q20 为什么独立验证和 checkpoint 选择很重要

Train loss 下降不等于泛化提升；最后一个 checkpoint 也不一定最好。验证集应按来源、prompt、会话或用户等关联分组，明确主指标和退化门槛。

验证集用于选择配方，封存 test 用于最终评估。反复看 test 后再调参，会使它失去原来的独立性。阶段末做一次回归，也不等于训练过程中已按完整能力矩阵选择最佳 checkpoint。

## 偏好对齐和强化学习

### Q21 PPO RLHF 需要哪些部件

常见 LLM PPO 系统有策略、参考策略、奖励模型或奖励函数，以及价值估计器。过程包括 rollout、奖励与 KL、advantage/return 估计、旧策略 log-prob 缓存、clipped policy 更新和 value 更新。PPO 的核心是限制策略更新幅度的 surrogate objective。参见 [PPO](https://arxiv.org/abs/1707.06347)。

只写一项 reward 乘 log-prob，不能代表完整 PPO。需分别核验旧策略/参考策略、有效回答 token、EOS、padding 和损失归约。

### Q22 DPO 怎么直接使用偏好对

给定 prompt x、chosen y+ 和 rejected y−，一种标准目标是：

```text
Δ = [log πθ(y+|x) − log πref(y+|x)]
  − [log πθ(y−|x) − log πref(y−|x)]
L_DPO = − E log sigmoid(β Δ)
```

DPO 通过参考策略下的相对偏好构造优化目标，不需要先训练显式奖励模型再运行在线 RL。参见 [DPO](https://arxiv.org/abs/2305.18290)。

实现时说明 completion log-prob 怎样求和、是否含 prompt、reference 是否冻结，以及偏好样本的长度和来源偏差。DPO 不等同于两份 SFT loss 的任意加减。

### Q23 GRPO 与 PPO 有什么关键区别

GRPO 对同一问题采样多条回答，利用组内奖励构造相对优势，从而不必使用单独的 critic。原始 outcome-supervision 配方常用 `(reward − group_mean)/(group_std + ε)`，再配合 clipped ratio 和参考策略 KL。参见 [DeepSeekMath 的 GRPO 章节](https://arxiv.org/html/2402.03300v3#S4.SS1)。

同组奖励完全相同时，优势可能接近零。需要检查组的定义、旧策略采样概率、回答掩码、长度归约和奖励可验证性。不同 GRPO 变体的归一化与偏差处理不能混成一种算法。

### Q24 奖励变高为什么能力可能没变好

模型可能利用奖励函数漏洞、重复格式模板、倾向冗长回答，或过拟合训练题。可验证奖励也依赖答案解析器、执行环境和测试用例质量。

保留独立问题、隐藏测试和失败类型统计，比较奖励与真实任务成功率。CoT 更长、工具调用更多或格式更整齐，都不能单独证明推理更可靠。

## 蒸馏和模型压缩

### Q25 蒸馏除了学教师答案还包括什么

序列蒸馏将教师生成结果当监督数据；logits 蒸馏匹配软分布；特征蒸馏匹配中间表示。常见 logits 目标含 `T² KL(p_teacher(T) || p_student(T))`，并可加入真实标签 CE。参见 [知识蒸馏论文](https://arxiv.org/abs/1503.02531)。

不同词表不能直接逐维做 KL；hidden size 不同也不能直接 MSE。Padding/prompt 掩码、有效 token 平均和独立验证决定训练是否符合预期。On-policy/off-policy 还涉及学生实际访问的输入分布。

### Q26 PTQ AWQ GPTQ 分别是什么

PTQ 在训练后量化，通常借助代表性校准数据选择参数。GPTQ 使用近似二阶信息进行逐层误差补偿；AWQ 根据激活观察重要通道并寻找缩放，降低低比特权重量化误差。参见 [GPTQ](https://arxiv.org/abs/2210.17323) 和 [AWQ](https://arxiv.org/abs/2306.00978)。

需要说明 W4A16/W8A8 等具体格式、分组方式、对称性、校准分布、保留高精度的模块及部署内核。AWQ 不应被简单描述为“把重要的百分之一权重都保存为 FP16”。

### Q27 QAT 怎么缓解量化掉分

QAT 在训练中插入 fake quant，模拟量化及反量化的舍入、裁剪和缩放误差，再用适合的梯度近似更新浮点参数。训练结束后要转换成目标存储与执行格式。参见 [PyTorch QAT 官方说明](https://pytorch.org/blog/quantization-aware-training/)。

Fake quant、最终整数转换、重载和实际执行必须保持同一数值契约。仅给 loss 加“量化惩罚”或加载 4bit 权重，不足以证明实现了目标后端 QAT；正确性与内核速度也要分别验收。

### Q28 剪枝后为什么不一定加速

非结构化剪枝把分散权重置零；结构化剪枝通常删除通道、头或层，要求同步调整关联矩阵、残差和配置。把整行置零也仍可能保留原矩阵尺寸。

应预留恢复微调/蒸馏预算，记录掩码并检查恢复时零权重是否回生。验收同时看能力、真实维度、序列化体积、硬件支持和耗时，不能只看 sparsity。

## 推理和服务

### Q29 Prefill 和 decode 的瓶颈有什么区别

Prefill 处理输入 prompt，可并行计算多个位置；decode 通常逐步生成，依赖已有 KV。小 batch decode 常受带宽影响，但实际瓶颈会随模型、batch、上下文长度和内核改变。

把 TTFT、每 token 延迟和吞吐分开测；缩短 prompt 得到的收益不能直接归因于模型执行器优化。

### Q30 KV Cache 的显存怎么估算

在各层布局一致、KV 头维相同的常见情形下：

```text
KV bytes ≈ 2 × L × B × S × H_kv × D_head × bytes_per_element
```

其中 2 代表 K 和 V，L 是层数，B 是活动序列数，S 是缓存长度。混合层、滑窗、KV 量化和分页预留需另算。GQA/MQA 的头数区别参见 [GQA](https://arxiv.org/abs/2305.13245)。

KV 复用避免重算历史的 K/V；当前 token 仍需与缓存交互，并没有消除 attention 的全部计算。

### Q31 FlashAttention PagedAttention 和 continuous batching 怎么区分

| 技术 | 主要优化对象 |
|---|---|
| FlashAttention | Attention 的分块和显存读写，避免显式保存大注意力矩阵 |
| PagedAttention | 动态 KV 内存管理、碎片和共享 |
| Continuous batching | 请求完成或到达时，动态调整后续执行批次 |
| Chunked prefill | 将长 prompt 分块，控制其与 decode 的调度干扰 |

FlashAttention 计算 exact attention，并没有一般性地把 dense attention 的 FLOPs 变成线性。参见 [FlashAttention](https://arxiv.org/abs/2205.14135) 和 [PagedAttention](https://arxiv.org/abs/2309.06180)。

配置名或可选依赖存在，不能证明目标请求命中了对应内核；要看 dispatch 和实测。

### Q32 Prefix cache 需要什么失效条件

共享前缀可以复用已有计算，但缓存必须属于正确的模型状态、tokenizer、模板、token IDs、位置和相关执行配置。要明确容量预算、淘汰和多请求缓存的可变状态所有权。

正确性测试应覆盖前缀相同/不同、模型参数变化、命中/未命中、不同请求继续解码及并发访问。性能测试同时测冷缓存、热缓存和真实命中率。

### Q33 投机解码怎样保持目标模型行为

小模型提出候选，大模型验证。Greedy 模式应与目标模型 greedy 输出一致；采样模式需要按目标/草稿概率做接受拒绝，并在拒绝时使用修正分布，才能保持目标分布。参见 [投机解码论文](https://arxiv.org/abs/2211.17192)。

不能默认接受所有草稿，也不能在拒绝位置同时追加 target 和 draft token。需测试全接受、首 token 拒绝、中途拒绝、EOS、输出预算、词表兼容和分布一致性；acceptance rate 不能代替计时得到的 speedup。

### Q34 Temperature Top k Top p 影响什么

Temperature 重缩放 logits；Top-k 保留概率最高的固定数量 token；Top-p 保留累计概率达到阈值的动态集合，再归一化采样。它们改变解码策略和输出分布，可能影响质量、长度与重复。

Greedy 对比、采样对比要分开；不能一边改变采样配置和输出长度，一边把延迟变化全算成内核加速。

### Q35 大模型服务应该记录哪些性能指标

至少记录模型/权重版本、dtype/量化、硬件/后端、输入输出长度、并发、成功率、TTFT、每 token 延迟或 TPOT、端到端延迟、tokens/s 和显存峰值。SSE 连接建立或空 chunk 不应冒充第一个可见输出。

使用一致输入分布、实际生成 token 数和冷/热条件，报告 P50/P95/P99，并在优化前后检查输出正确性或能力门槛。

## 应用和评测

### Q36 RAG 和 SFT 怎么选

RAG 在推理时检索外部材料，适合知识更新、可引用证据和权限约束；SFT 更新参数，适合行为、格式、工具使用和任务模式。两者可以组合，不能保证任一方案单独解决幻觉。检索与生成结合的基本思路参见 [RAG 原始论文](https://arxiv.org/abs/2005.11401)。

先定位问题来自知识缺失、检索失败、上下文使用不当还是任务行为，再决定改数据、检索、prompt 或训练。

### Q37 一个完整 RAG 系统要验证什么

链路包括解析/清洗、切块、版本和权限元数据、索引、召回、融合/重排、上下文组装、生成与证据引用。按任务评估 Recall@k、排序质量、答案正确性、引用是否支持结论、无证据拒答、时效性和端到端延迟。

Embedding 和 reranker 是组件；有这两个服务，不代表已经具备完整 RAG。对私有知识库还要验证权限过滤在检索前后都有效。

### Q38 Agent 与 function calling 的边界是什么

Function calling 是模型提出结构化工具请求的一种接口；Agent 还涉及状态、步骤、预算、工具执行、观察与终止。ReAct 将推理与动作/观察交替，参见 [ReAct](https://arxiv.org/abs/2210.03629)。

合法 JSON 不代表参数有效或动作有权限。需检查 schema、工具白名单、参数约束、预算、超时和副作用；检索内容及工具输出应作为数据处理。URL 检查、API 鉴权和 prompt injection 防御也不是同一件事。

### Q39 幻觉和模型质量应该怎么评估

PPL 衡量给定参考文本的平均预测不确定性，不等于事实准确率、数学 EM、代码 pass@k 或安全性。PPL 依赖 tokenizer、预处理、有效 token 与 loss mask，不能跨不同 tokenizer 或损失归约口径直接比较。选择题、生成题、代码和检索任务需要匹配的评分器与失败定义。

LLM-as-judge 应检查位置/长度等偏差并用人工样本校准；benchmark 要报告版本、任务子集、样本数、prompt、解码与答案解析。数据/模型内容指纹能保证来源绑定，不能证明从未见过训练题。

### Q40 面试讲项目怎样提供可信证据

按“问题→基线→改动机制→对照实验→收益与代价→失败案例”讲。至少准备一个独立能力回归、一个延迟/显存对照和一个明确未达标案例。

质量退化、硬件不支持、统计不稳定和实验未覆盖的范围应落到具体配置。避免把单次 tiny-model 测试、模块名称或上游论文的速度，当成本项目的生产结论。

## 常见误解速查

| 常见说法 | 更准确的表达 |
|---|---|
| LoRA 不会遗忘 | 更新受低秩约束，仍需能力保持和回归 |
| QLoRA 就是 QAT | 微调资源优化与目标量化数值训练是不同问题 |
| EOS 或 label mask 能隔离文档 attention | 还需要实际 attention 边界约束 |
| 有 SDPA 或 flash 配置就是命中 Flash kernel | 取决于设备、dtype、shape、mask 和 dispatch |
| INT4/INT8 文件更小就一定更快 | 取决于加载后的执行格式和内核 |
| 结构化置零就是缩模 | 需确认关联维度真的被删除 |
| 有 reward 和 log-prob 就是 PPO/GRPO | 还要核验相应目标、旧策略、优势、mask 与更新 |
| 有工具适配就是本地支持 function calling | Provider 转发与本地约束生成是不同能力 |
| 有 embedding 服务就是完整 RAG | 还缺检索、上下文、引用和评测链路 |
| 测试全绿就证明所有算法正确 | 还要检查测试覆盖了哪些执行路径与数值性质 |

## 对照自己的框架

优先检查五条链：数据与文档边界；SFT/DPO/GRPO 的目标与掩码；压缩后的恢复和执行格式；推理正确性与真实计时；评测污染与产物身份。每条链都应能指向执行代码和独立核验，而不只是配置字段。

本仓库的逐项结论、问题复现和后续验收建议见 [框架采用情况审计](LLM_INTERVIEW_FRAMEWORK_AUDIT_ZH.md)。
