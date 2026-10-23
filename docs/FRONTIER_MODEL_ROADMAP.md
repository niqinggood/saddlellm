# SaddleLLM：训练自有高水平大模型的研发路线

核对日期：2026-09-27。本文区分现有实现、公开技术事实和后续规划；不是“已训练出头部模型”的宣传。

## 1. 我们要做什么

目标：SaddleLLM 以大模型为主，为训练自有高水平模型提供完整底座。使用者通过 CLI、
Python import 和配置完成数据、预训练、后训练、评测、轻量化和推理；底层专业、可恢复、
可追踪。暂不扩张到无关的通用 ETL、批处理或流处理平台。

必须区分三件事：

1. **适配已有模型**：加载别人的受支持 checkpoint，做微调/蒸馏或部署。
2. **训练相似架构**：从随机权重训练采用相近思想的自有模型。
3. **达到同等能力**：在约定的独立测试、推理预算和资源约束下，质量/延迟/成本可比较。

第一项不意味着第二项，第二项不意味着第三项。改名称、增大参数量、接上 Megatron 或
降低训练 loss，都不能单独证明追平头部模型。我们应把自己的产物称为 Saddle 模型；
不把结构相似的权重冒称为原版 DeepSeek、Kimi 或 Jev。

## 2. 三条路线不是同一种模型

| 对标方向 | 公开技术事实 | SaddleLLM 当前基础 | 当前不能声称的能力 |
| --- | --- | --- | --- |
| DeepSeek 风格生成模型 | DeepSeek-V3 的公开方案包含 MLA、MoE、辅助损失自由的负载均衡和 MTP；还包含大规模预训练与后训练。[官方仓库](https://github.com/deepseek-ai/DeepSeek-V3) | 原生 `ModelBlueprintLab.deepseek_style_moe`、`SaddleAttention`、`SaddleMoE` 和 MTP loss 能组成研究模型 | 不是完整复现 DeepSeekMoE、原版 MTP、FP8 集群训练或全部配方；原生 MoE 的专家循环不是 expert-parallel 引擎 |
| Kimi K3 风格生成/多模态模型 | 官方介绍强调 KDA、Attention Residuals、Stable LatentMoE、原生视觉和百万 token 上下文，模型约 2.8T 参数。[官方介绍](https://www.kimi.ai/blog/kimi-k3) | 已有 KDA 顺序参考、KDA+GQA/MLA 分层混合、Full/Block AttnRes，接通微型训练/缓存/重载；另有 MLA/MoE 实验组件 | 不是原版 Kimi 架构/权重或优化内核；尚无 Stable LatentMoE、MoonViT-V2 联合预训练或百万 token 训练验收；提高 context 配置不等于已支持该长度 |
| Jev 风格决策模型 | TypeSafe 2026-09-15 发布 Jev，输入状态和问题，直接输出结构化决定；发布说明称采用并行采样和 RLCD。[官方发布](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 已有 Choice 基线和独立多类型决策训练/服务闭环；支持 Choice/Score/Noul、骨干联合训练、温度校准和实验性 Brier 策略梯度 | 真实业务及通用能力尚未验证；官方材料不足以完整复现其架构和 RLCD，不能宣称兼容其权重或复刻其模型 |

DeepSeek-V3 在这里是可核验的技术实例，不是对“当前最新 DeepSeek 版本”的断言。
其官方记录为 671B 总参数、每 token 激活 37B、14.8T 预训练 token，以及完整训练
约 278.8 万 H800 GPU 小时。这说明资源和数据是训练计划的组成部分，不是一个 CLI 开关能代替的。
这些是作者公开报告，不是 SaddleLLM 独立复跑的成绩。[同一官方记录](https://github.com/deepseek-ai/DeepSeek-V3)

### Jev 值得借鉴在哪里

实现与学习笔记见 [Saddle 决策模型与 RLCD 学习笔记](DECISION_MODEL_NOTES.md)。2026-10-02 已在 L20 完成冻结骨干的 BANKING77 Choice 基线；2026-10-03 又完成多类型网络、真实 Qwen3-Embedding-0.6B 骨干联合训练、保存重载与 HTTP 验证。实验性 Brier 策略梯度是我们的独立研究。当前合成数据试验不证明通用 Choice/Score/Noul 能力，也不代表原版 RLCD 复现。下文初始缺口与 P2 目标保留为研究背景，已实现情况以训练笔记为准。

本节与 P2 的 Jev 专项补充核对日期：2026-10-02；其他路线仍沿用本文开头的核对日期。
早期内容研究可行性；后续实现和实测结果已补充到训练笔记。

Jev 的接口不是聊天文本：Choice 输出选项及概率，Score 输出评分及概率，Noul 输出
0–1 的真假概率；同一状态可同时提出多个独立问题。
[官方接口介绍](https://docs.typesafe.ai/introduction)

官方事实与边界：

- TypeSafe 的发布文章日期为 2026-09-15，公布端到端 70–500 ms、40–200 倍加速。
  这些是厂商在特定 System One 任务和服务条件下的结果；官方说明评测通常从靠近服务的
  美国西海岸发起，短而密集的输入也有利于演示。不能当作任意任务的保证或 Saddle 的成绩。
  [官方发布与评测说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- 当前文档列出 `jev-1.13.0`，输入价格为 $0.042/百万 token、输出免费。这是 API 售价，
  不是其 GPU 计算成本，也不能直接推算自有模型训练或本地部署成本。
  官方还明确不提供基于客户数据的 Jev 微调或 LoRA；不能把 Saddle 的训练命令直接指向 Jev
  就获得可训练权重。[模型、价格与定制说明](https://docs.typesafe.ai/models)
- Choice/Score 的 `confidence` 是由返回的概率分布计算的统计量，不是独立预测的正确率，
  也不等于已经通过目标领域的概率校准验收。Noul 只返回真假概率，没有单独的 `confidence`。
  [confidence 定义](https://docs.typesafe.ai/confidence)
- Score 是有序等级概率的期望分数；等级描述仍需由业务定义，数值精度不代表客观测量精度。
  [Score 定义](https://docs.typesafe.ai/primitives/score)

我们可以研究自有的决策模型路线，但应明确：

- 不是让模型生成 JSON 后再解析，就声称获得 Jev 的原生并行决策架构或速度。
- 不是普通固定标签分类器换名：若目标是通用决策，还需支持输入中的问题和动态候选标准。
- 格式合法不代表选项正确。约束输出不能保证事实判断零错误。
- softmax 概率不能未经验证就命名为“已校准可信度”。概率为 0.8 的一组事件应在独立数据上
  约有 80% 发生；这不是单次回答正确的保证。
  [官方校准解释](https://docs.typesafe.ai/introduction/machine-learning-primer)

初始研究阶段的代码基础与最小缺口：

| 范围 | 可复用基础 | 决策路线仍缺少什么 |
| --- | --- | --- |
| 模型 | [SaddleModeling](../saddlellm/models/SaddleModeling.py) 的文本骨干及隐藏状态 | 当前是 causal LM；需跳过词表投影的编码接口、决策读出、动态候选 mask 与模型配置/重载支持 |
| 训练与蒸馏 | [NativeTrainer](../saddlellm/training/NativeTrainer.py) 的 Trainer 集成与保存；[DistillationTrainer](../saddlellm/distillation/DistillationTrainer.py) 的 KL/CE 思路 | 现有 loss/蒸馏面向 token；需决策数据契约、collator、候选对齐及 CE/BCE/软标签 loss。SFT/DPO 不能视为 RLCD |
| 对照评测 | [MultipleChoiceEvaluator](../saddlellm/evaluation/MultipleChoiceEvaluator.py) 的单次前向候选标签概率与批量评测；[分类评估](../saddlellm/evaluation/LLModelEvalute.py) | 现有路径可作基线，尚未覆盖通用动态候选训练、多问题隔离和完整概率质量验收 |
| 概率校准 | [SpatialWorldModelEvaluation](../saddlellm/spatial/SpatialWorldModelEvaluation.py) 的二分类 Brier/ECE | 需通用决策评估、独立校准集及校准参数拟合/持久化；量化 calibration 不属于概率校准 |
| 导出与服务 | [ModelExporter](../saddlellm/runtime/ModelExporter.py) 的产物打包；[InferenceServer](../saddlellm/runtime/InferenceServer.py) 的认证、准入和服务框架 | 模型加载/合并仍面向 causal LM；需决策模型与校准参数一同导出、专用加载及一次前向返回 typed 数值的接口 |

以上独立研发建议是我们的推断，目标是**自有 Saddle 决策模型**。公开资料不足以完整复现 Jev
架构、训练数据和 RLCD；这里不承诺兼容 Jev 权重、复现其训练算法或获得相同质量/速度。

## 3. 当前代码能支撑什么

| 模块 | 已有基础 | 仍需补齐/规模化验证 |
| --- | --- | --- |
| 数据 | 语料处理、tokenizer、packing、数据检查与切分污染检查 | 统一来源/许可证/混合比例/实际 token 清单，流式恢复、去重规模化和封存测试保护 |
| 原生预训练 | 蓝图构建、causal LM loss、分块词表 loss、checkpoint；已有自有 28M 检查点 | 持续收敛、完整恢复可重复性、数据配方与同算力对照；28M 产物不代表已具备通用头部能力 |
| MoE/MLA/MTP | 原生前向、梯度、router 指标、共享专家、研究型多预测头 | 分布式专家派发/grouped GEMM、训练稳定性和质量消融；研究 MTP 不等于原版 DeepSeek MTP 模块 |
| 混合注意力/AttnRes | KDA 递归参考、KDA+GQA/MLA、Full/Block 深度混合，接通 checkpoint 和生成 | chunkwise 融合 KDA、两阶段 AttnRes 推理、TP/PP 通信、长序列吞吐/收敛和同预算质量评测；参考路径不保证加速 |
| 后训练 | 原生 SFT/DPO；HF 路径接入 SFT/LoRA/QLoRA 和 DPO/ORPO/KTO；蒸馏组件 | 原生路径不支持所有 HF 算法；可靠的可验证奖励、在线采样训练与长任务环境闭环仍需开发 |
| 分布式/可选后端 | 原生预训练的 DDP/FSDP/DeepSpeed 配置路径；可选 Megatron Bridge、SGLang 采样适配器 | 真实 Linux 多 GPU/多机、通信、故障恢复和吞吐验收；配置存在或替身测试通过不是生产集群验收 |
| 评测发布 | PPL、选择题、completion、rubric、配对比较、发布门禁和导出 | 统一训练实验总清单、客观任务沙箱、概率校准与能力退化报告；API/工具测试通过不是模型质量达标 |
| 推理轻量化 | 原生 KV/采样/投影优化、固定微批、可选前缀快照；若干量化/剪枝入口 | 连续批处理/分页 KV 尚无原生实现；KV 量化占位和物理剪枝不能冒称实际压缩/加速 |

代码入口：

- [ModelBlueprint.py](../saddlellm/models/ModelBlueprint.py)：定义/验证研究结构，`build_model()` 创建自有随机模型。
- [SaddleModeling.py](../saddlellm/models/SaddleModeling.py)：注意力、MoE、预测头、loss、前向和生成。
- [TrainingOrchestrator.py](../saddlellm/training/TrainingOrchestrator.py)：`model.backend: saddle` 下的 scratch/continue、训练阶段、恢复与产物。
- [NativeTrainer.py](../saddlellm/training/NativeTrainer.py)：原生 checkpoint 保存与 Trainer loss 集成。
- [NativePostTraining.py](../saddlellm/training/NativePostTraining.py)：原生 SFT/DPO，不把尚未接通的 RL 当作完成。
- [可选后端指南](OPTIONAL_RUNTIME_BACKENDS.md)：Megatron/SGLang 的适用环境和边界。
- [轻量化审查](MODEL_LIGHTWEIGHT_REVIEW.md)：真实实现、占位与持久化缺口。
- [混合注意力与 AttnRes](HYBRID_ATTENTION_AND_ATTNRES.md)：配置、数学定义、原生流程、缓存和上线边界。

原生蓝图路径与 HF 的架构注册/加载路径不同；不能只看到注册表里一个 `deepseek` 名称就
默认两条路径都支持完整从零训练。自定义原生 checkpoint 也不自动兼容所有外部推理引擎。

本轮修正了研究预设的一个具体问题：`deepseek_style_moe` 开启 `aux_loss_free=True`
时，显式设置 `router_aux_loss_coef=0.0`，避免默认仍混入 router 辅助 loss。
显式用户蓝图若希望混用两种机制，可以自行设置，不被全局禁止。

对应微型回归见 [训练/保存/重载测试](../tests/test_deepseek_style_training.py) 和
[完整 scratch/Trainer 恢复测试](../tests/test_scratch_pretrain_e2e.py)。前者验证普通/分块
loss、有限梯度、router/MTP 参数更新及重载一致性；后者保留 Dense 控制组，另测研究型
MLA/MoE/MTP 蓝图从随机权重到 checkpoint、恢复和最终生成。它们不验证正式任务质量，
也不代表多机训练验收。可运行：

```powershell
python -m pytest -q tests/test_deepseek_style_training.py tests/test_scratch_pretrain_e2e.py
```

2026-09-27，前一轮上述 5 个微型训练用例通过；连同推理、模型组合、分布式状态等
选定回归共 487 项通过。该轮复现命令见
[本轮回归验收](INFERENCE_PERFORMANCE_COMPARISON.md#本轮回归验收)，不是全仓库测试成绩。

随后补齐 KDA 混合注意力与 Full/Block AttnRes，并增加混合架构 scratch 恢复、
原生 SFT→DPO 和服务加载验收；同时修复使用过程中的混合精度/缓存/配置问题。
最新选定组合回归为 673 项通过、1 项主动排除；命令与边界见
[使用流程排错与回归验收](BUGFIX_VALIDATION_20260927.md)。

## 4. 实施顺序与每阶段验收

以下为规划，不是已新增的命令、已训练出的模型或速度/质量承诺。

### P0：可复现的自有模型研发闭环

先用小模型验证基础设施，保留 Dense 控制组和 MLA/MoE/MTP 实验组。

- 数据版本、切分、许可证、配置、种子、依赖、checkpoint、实际训练 token 和 GPU 时间可追踪。
- 训练从随机权重开始；“续训权重”和“恢复 optimizer/RNG/数据位置”明确分开。
- 对每个 checkpoint 测固定 validation，封存 test 不参与调参。
- 发布同时报告模型质量、峰值显存、训练/推理吞吐、失败率与资源消耗，不能只报训练 loss。
- tiny 测试必须覆盖有限 loss/梯度、参数更新、保存重载、生成、恢复和失败处理。

验收：陌生用户只用数据、配置、CLI/import 即可重跑，不修改核心代码；报告能说明
候选相对基线是否进步、代价多少、哪些能力退化。微型随机数据仅用于流程正确性，不用于能力宣传。

### P1：先提高单位算力得到的生成能力

- 用相同的数据预算和可比较计算量做 Dense / MLA / MoE / MTP 消融，而不是把所有新结构一次打开。
- 优化数据混合、学习率、初始化、router 稳定性和训练故障诊断。
- SFT、蒸馏、DPO 在同一独立任务集对照，避免只奖励答案变长或迎合评分器。
- 长上下文需要逐步训练、长度分桶评测和资源测试，不只改 RoPE/context 配置。
- 可验证奖励 RL 先定义奖励/环境/安全契约，再接真实 rollout 和策略更新；不把异步 HTTP
  请求视为已经完成在线 RL。代码任务在隔离沙箱验收，不直接执行任意生成代码。

验收：在约定的同规模或同预算基线上有可重复的质量提升；没有用测试污染、额外工具或
更高推理预算掩盖差距。先追单位资源效果，再决定扩模型规模。

### P2：独立的结构化决策模型实验线

本阶段尚未实现；本轮仅完成资料与源码研究。按以下顺序验证，避免同时引入新架构和新训练算法：

1. **最小监督基线与校准。** 先选一个有真实标签的领域，定义 `state + questions + criteria`
   数据契约；教师软标签需记录来源。复用预训练骨干，先用现有候选标签评分建立基线，再训练
   简单决策读出。独立保留训练、校准、验证与测试数据，在校准集拟合温度等参数，报告准确率、
   NLL/Brier、校准误差、可靠性曲线及拒判/人工接管指标。固定分类头只能证明该固定任务有效。
2. **动态候选决策。** 引入问题与候选描述条件化评分：Choice 对有效候选归一化，Noul 预测
   二分类概率，Score 预测有序等级分布并取期望。用 CE/BCE 或候选对齐的软标签 KL 建立监督
   基线；测试可变候选数、候选置换、中文和域外数据，再评估是否需要专用后训练。自行设计的
   算法不用 RLCD 的复现名义，`confidence` 公式也不能代替校准实验。
3. **多问题并行与共享状态。** 首先把独立问题展开成批次，建立正确性和耗时基线；之后再研究
   共享状态编码、问题隔离注意力和批量决策头。直接按因果顺序拼接问题会引入顺序依赖，不能
   自动获得独立并行语义。分别验证问题顺序不影响结果、问题不串扰及状态复用的实际收益。
4. **交付与验收。** 将决策权重、head 配置和校准参数一起保存/导出，补专用数值响应接口。
   在固定准确率和概率质量要求下，与分类器、现有候选标签评分及 LLM JSON 基线比较；改变
   状态长度、问题数和候选数，记录端到端 P50/P95、吞吐、显存及资源成本。只有独立实测通过
   才扩大模型与数据规模，不能复制官方 Jev 的速度数字当作我们的成绩。

验收：动态候选合法、独立问题不串扰、概率可校准、不确定时能拒判；在固定准确率/
校准要求下比较 P50/P95、吞吐和成本。“某一任务接近头部”不得写成“通用能力全面追平”。

### P3：跨过真实的大模型与集群门槛

- 在受支持 Linux GPU 环境验证 Megatron recipes、权重转换和部署兼容，再扩大多机。
- 专家/张量/流水线/上下文并行、检查点分片、通信重叠与故障恢复分别做真机验收。
- FP8、融合算子、KDA/AttnRes/LatentMoE 或原生视觉按证据逐项引入，不强迫本机 Windows
  用户安装完整集群栈，也不先重写一整套 C++ 引擎。
- 模型规模、总 token、设备/节点、数据授权、费用上限与停止条件在正式训练前确认；
  本轮不会自行下载巨型权重、采购 GPU 或启动长时间训练。

验收：同一模型跨小/大规模结果可解释，多机保存恢复可靠，资源预算与实际观测一致，
质量和部署能力通过独立任务评测，再讨论更大规模的头部竞争。

## 5. “对齐头部”的最终验收口径

对每项比较冻结：基准版本/哈希、数据授权与污染检查、提示模板、工具权限、模型版本、
精度/量化、上下文、采样/思考 token 预算、并发和硬件。保留逐题结果、重复实验和误差范围。
不同闭源服务无法核验全部训练条件时，明确比较的是可观察的端到端能力，而非宣称同算力训练胜出。

至少分别报告通用知识、数学/推理、代码、中文、长上下文、工具执行、结构化决策、安全
以及延迟/吞吐/成本；视觉只在真实具备该训练和输入路径后评测。不用一个总分覆盖全部差距。

目标可以高，但结论只能按证据晋级：流程可用 → 同预算小模型有效 → 独立任务接近 →
真实集群可扩展 → 广泛能力有竞争力。CLI 和 import 的简单性始终保留，复杂性封装在经过
验收的后端中，而不是通过省略错误检查或隐藏失败结果来实现。
