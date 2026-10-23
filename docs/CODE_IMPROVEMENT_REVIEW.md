# SaddleLLM 代码完善审查

## 2026 年 10 月 7 日：首轮生命周期修复

本轮落实数据契约、公共评测入口和模型内容追溯，不增加用户权限、角色、审批或运行许可。下文旧审查中的同名条目保留为历史背景，以本节说明和当前代码为准。

- KTO 预检与训练归一化共用严格标签解析，空格与大小写别名一致；未知、缺失或非二元标签给出记录位置，不再静默转成负例。单侧 chosen/rejected 的既有格式仍可推断。
- HF SFT 在展开 assistant 轮次之前按原始对话和明确 case/group 划分；原生 DPO 按相同 prompt 和明确分组合并。独立验证文件保持用户提供的划分；整库 source_id 不被默认当成 case。
- `run_evaluation` 为 CLI 与流水线提供共同的 multiple-choice、completion 和 rubric 入口。旧评测不再在 validation 加载失败时改读 test；多 split JSON 需明确指定。报告保留实际 split、选中记录指纹和模型内容身份。
- 普通导出默认不要求 gate。已有 `--require-gate` 严格选项才要求通过报告与当前本地模型内容匹配，校验失败发生在创建/覆盖导出目录之前。旧报告缺内容身份时仍可普通导出，报告不能冒充已绑定的通过证据。
- 发布 ID 包含实际输出的模型内容，避免相同大小、不同权重得到相同 ID。源模型与合并/转换后的发布模型分别记录身份；`hash_weights` 的公开清单默认行为保留，已计算的摘要复用。
- CLI 补齐已有原生导出后端的 `--format saddle`（别名 `native`），默认 HF 导出用法保持。
- Typed 预测概率使用与离线评测一致的精度，修复极近 logits 被舍入成平局时两接口选择不同候选的问题；没有改变模型权重或训练目标。

这些改动会改变错误标签或原本泄漏的数据划分；旧训练产物不被自动重写，训练效果应在同一独立留出集重新比较。完整训练恢复、DPO 参考缓存、任务采样控制和网页任务调度仍是后续工作。

命令与配置使用示例见 [公共评测与导出说明](WEBUI_AND_SERVING.md#9-训练评测与模型导出的一致性)。

## 2026 年 10 月 3 日的训练框架检查

本轮完成多类型决策模型的实现和 L20 远端调试后，重新检查了训练入口、配置默认值、数据预检、检查点和通用评测。后续改进应以使用者能顺利跑通真实任务、并判断微调是否有效为主线。下文旧日期的审查保留为历史背景，具体已完成情况需结合当前代码判断。

本轮已落实三项使用改进：决策直接命令识别 `dataset.json` 的数据类型；原生小模型和 HF 骨干使用各自的联合训练默认值；提供 CPU Adam 主权重和状态，以降低完整骨干训练的 GPU 内存需求。此前多类型配置省略参数时会继承冻结 Choice 头的默认值，导致冻结骨干、batch size 256、学习率 0.001；现在 HF 多类型默认联合训练、batch size 1、学习率 0.00002、3 个 epoch，小型原生模型使用 batch size 16、学习率 0.001、20 个 epoch。显式参数不会被覆盖。

训练数据检查已包含 SFT/偏好训练的训练与验证泄漏审计；原生 SFT 已按原始对话划分验证集。这些能力不应再次列为“完全未实现”。仍需统一不同后端的来源组切分与监督 token 统计，并提供清晰的丢弃原因，让用户判断模型实际学习了什么。

| 优先级 | 后续工作 | 当前代码证据和用途 | 验收方式 |
| --- | --- | --- | --- |
| P0 | 将真实任务评测与基座对照接入通用 eval 阶段 | `_run_eval_stage` 仍调用 `LLModelEvalute.Evaluator`；选择题、completion、rubric、成对比较已有独立实现。用户微调后应直接看到基座与候选在相同保留集上的差异 | 配置选择评测协议；报告逐条结果、提示变体稳定性、验证集选型和封存测试。验证集通过不等于发布通过 |
| P0 | 统一有效监督与来源组报告 | 数据预检、跨集合审计和对话分组已经存在，但 HF/原生路径的实际 token 化、回答截断和丢弃记录还需统一呈现 | 长提示样本仍保留真实回答监督；同源变体不跨集合；输出有效监督 token、截断率、重复率与任务覆盖 |
| P0 | 固定真实任务评测后再做小规模训练参数对照 | 超参数应区分完整微调、LoRA、评分头和多类型读出。合成数据上的单次奖励收益不能代替业务验证 | 用相同数据和预算对照学习率、目标模块、混合任务比例；至少重复多个种子，同时检查领域效果和通用能力退化 |
| P1 | 完善多类型训练恢复与任务采样 | 当前多类型训练支持加载权重后重新训练，尚未恢复优化器、随机状态和训练步；混合数据按行采样 | 中断恢复与连续训练的步数/参数一致；避免大任务掩盖小任务；每类型、每领域及未见问题单独评测 |
| P1 | 统一实验身份和产物追溯 | 配置、流水线状态、模型哈希、训练摘要和独立评测报告已有基础，但需要一个便于读取的实验入口 | 一份 manifest 可定位代码版本、模型、数据、所选 checkpoint、校准范围与最终评测；训练结束返回可用的下一条命令 |
| P1 | 补充显存预估与吞吐预算 | 本轮共享 L20 上完整 FP32 AdamW 触及显存预算，CPU 优化器降低 GPU 峰值但增加 CPU 内存和传输开销 | 预检区分权重、梯度、优化器与激活；提示可用的 batch/累积/卸载选项；用有效监督 token/s、峰值内存和完整耗时对照 |

实施顺序建议是先补齐通用效果评测和数据监督报告，再优化训练配方、恢复与速度。保留统一的 `doctor → inspect-data → train --dry-run → train → evaluate/compare → export/serve` 使用路线，复杂后端作为明确选项出现。当前多类型模型的能力、命令和远端证据见 [决策模型训练说明](DECISION_MODEL_NOTES.md)。

> 审查日期：2026-09-13
> 范围：文本后训练、通用评测、发布门禁、Web 看板和本地推理主链。本文不把尚未接入主链的研究模块当作可用能力。

## 1. 本轮已修复

### 1.1 统一发布门禁语义

问题：指标门禁使用 `accepted/status`，配对比较使用 `decision=promoted`。`compare-evaluations` 的输出直接交给 `export-model --require-gate` 时会被错误阻断，说明文档中的评测—导出链并未真正互通。

修复：

- 所有新配对比较报告都写出规范字段 `accepted: true/false`；
- `release_gate_accepted()` 兼容既有 `accepted`、`status`、`decision` 和 manifest 内嵌 gate；
- 无明确决策的 JSON 一律按未通过处理，不根据局部指标猜测；
- 导出 manifest 始终补写规范 `gate.accepted`；
- `export-model --require-gate` 已能直接消费新旧配对比较报告。

后续真实评测又发现 validation-only 比较也会被旧逻辑标为
`promoted/accepted=true`。现在 validation 达标只表示可以选择候选，输出
`decision=validation_passed`、`selection_eligible=true`、`accepted=false`；必须同时提交
配对封存 test 且两组都达标，才允许发布晋级。这样 `export --require-gate` 不会把调参证据
误认成最终测试证据。

### 1.2 保持评估、发布和部署职责分离

审查中一度给 `serve-model` 增加 `--require-gate`，试图防止误启动未晋级模型。但这会让部署命令承担评估审批，和已有 `evaluate-model`、`compare-evaluations` 的职责重复，也增加普通用户理解成本。

最终处理：撤回推理门禁参数。`serve-model` 只加载用户指定模型；目录包含 release manifest 时，健康接口和页面展示其门禁状态，但不重新评估或阻断。需要自动控制发布时，在 CI/导出阶段使用可选的 `export-model --require-gate`。模型注册中心、权限和上线审批属于部署平台策略，不写进基础推理命令。

### 1.3 训练曲线不再混入旧运行

问题：`training_metrics.jsonl` 是追加文件。用户复用输出目录并从 step 0 开始新实验时，旧曲线会和新曲线混在一起；正常 resume 又必须保留历史。

修复：训练从 step 0 开始时清空旧 metrics；从 checkpoint 恢复且 global step 大于0时保留历史，并沿用原 `started_at`。

### 1.4 修正流式协议结束和错误语义

问题：流式聊天无论自然结束还是达到长度上限，都返回 `finish_reason=stop`；模型线程异常还可能表现为客户端连接突然断开。

修复：根据实际生成 token 数返回 `stop` 或 `length`；模型生成异常转换为结构化 SSE `server_error` 并以 `[DONE]` 结束；空消息在开始流式响应前即被请求校验拒绝；内置 WebUI 能显示流式错误而不把失败回答写入对话历史。

### 1.5 训练前识别固定模板过拟合风险

问题：字段格式正确、没有精确重复，不代表数据足以支持自然提问。当前护理选择题的 2,455 条训练样本共用一个指令前缀和一种两行答案格式，旧 `inspect-data` 仍返回 `severity=ok`。

修复：SFT 检查增加提示前缀和回答形态集中度。当前护理数据抽样 256 条，两项集中度均为 100%，现在会明确警告模板过拟合，并要求同源改写保留在同一 split。对话数据的共享 system message 不参与用户提示集中度，避免把标准系统角色误报成模板问题。

### 1.6 预训练数据和评估形成真实闭环

修复了长文档先截断、跨文档无EOS、PAD与EOS同号时误屏蔽真实标签、流式训练/验证泄漏、质量阈值未传入、全局batch配置未真正生效，以及有评估参数却没有验证集等问题。预训练现在先清洗去重并划分数据，再分别tokenize和packing；按验证loss选择最佳checkpoint，同时记录PPL、token吞吐、实际全局batch和最低学习率。

`saddlellm train pretrain`已用tiny本地模型真实继续预训练2 step，得到64 input tokens、validation loss 2.5188、PPL 12.4139，并选中step-2。该运行只证明工程链路，不代表护理效果。

### 1.7 中文语料不再被英文词长规则误删

旧质量启发式用空格分词计算平均词长，正常中文段落会被当成一个超长词。现在先识别CJK文本，仅对非CJK文本应用英文平均词长范围，同时保留长度、标点、URL和质量评分检查。该问题若不修，会让使用者以为在训练护理正文，实际有效语料却被过滤。

### 1.8 网络结构采用预设、蓝图和预检三层表达

继续预训练的网络结构完全继承checkpoint；从零预训练可以只选择已登记的`model.config`，也可以通过`model.blueprint`覆盖注意力、FFN、RoPE、上下文、残差、异构层段和训练目标。`validate-config`新增`architecture_analysis`，在分配模型前显示真实参数量、激活参数量、KV cache估算、结构风险和实验建议。训练后的原生checkpoint保存`model_blueprint.json`和`saddle_config.json`，避免配置与权重身份分离。

### 1.9 流式packing不再偷偷物化全部语料

`streaming=true`的token流过去会在packing时通过普通Dataset工厂全部物化，失去流式意义。现在保持为IterableDataset，并对可计数数据记录raw、clean、deduplicated、quality、train sequence和validation sequence数量。任一必需集合在packing后为空时，在创建Trainer前给出可操作的错误。

### 1.10 配置字段和实际Trainer参数保持一致

实际smoke发现`training.gradient_checkpointing`与`training.optimizer`未被解析，导致命令保存了`gradient_checkpointing=false`，训练却仍启用。现已修复并用相同命令复跑，冲突警告消失。预训练还增加tokenizer/model词表和EOS/PAD契约；token ID可能越界时不再等到forward阶段才崩溃。

## 2. 下一批应完善的主线能力

### 部分完成/P0：分组切分、跨 split 泄漏与提示鲁棒性评测

评测侧已支持 `--prompt-variants` 声明式模板，以及已展开数据的
`--case-field/--variant-field`。同一道题可在多种等价提示下运行，报告分模板准确率、最差
模板正确率、答案翻转率、所有模板均正确率和逐题证据；模板由领域项目提供，核心没有护理
文案。护理237题×3模板真实评测已暴露微调模型仍有72.57%的预测翻转率。

尚未完成的是训练侧 `source_id/group_id` 原生分组切分与语义近重复检查，必须继续确保同一题
的自然问法、教师问法和格式变体不会跨 train/validation/test。

已完成的验收：同一个 checkpoint 的标准模板和自然问法结果能由一条 `evaluate` 命令生成
同口径报告。待完成验收：构造同源提示变体跨 split 时预检失败。

### 已完成：直接命令覆盖 DPO/ORPO/KTO 与有序阶段

简洁命令现已支持SFT、DPO、ORPO、KTO，以及逗号分隔的严格有序阶段：

```text
saddlellm train --stage sft  --model ... --data ...
saddlellm train --stage dpo  --model ... --data ...
saddlellm train --stage orpo --model ... --data ...
saddlellm train --stage kto  --model ... --data ...
saddlellm train --stage sft,kto --model ... --data sft.jsonl --kto-data kto.jsonl
```

LoRA版`sft → kto`已用tiny模型真实各跑1 step，KTO正确继续训练上游adapter，而不是重新从基座开始。`--stage sft,ppo,kto`可准确展示线性依赖；因为PPO仍缺reward/value model和rollout接线，实际执行会在任何训练前失败，不会静默跳过。偏好训练用于纠正回答偏好和安全行为，不能替代缺失的护理知识或教师审核。

### P0：统一 experiment manifest

当前配置、数据 manifest、训练 summary、checkpoint、评测、比较门禁和 release manifest 虽然都存在，但仍是多个文件之间的松散约定。需要一个稳定的 `experiment_manifest.json` 串起：

- 配置和源码版本；
- 数据版本、许可、哈希和 split；
- 基座、adapter、checkpoint 选择理由；
- 评测协议、逐题结果和发布门禁；
- 导出 release、重载结果和部署容量报告。

验收：任一 release 能反查到训练数据和评测，任一实验能明确回答“为什么选这个 checkpoint”。

### P0：把新评测器接入 TrainingOrchestrator

当前 orchestrator 的 `eval` stage 仍调用旧 `LLModelEvalute.Evaluator`，而新的 multiple-choice、completion、真实生成和配对比较主要通过独立 CLI 使用。下一步应通过配置选择 evaluator，并在同一运行内完成：

```text
base evaluation → candidate evaluation → paired comparison → release gate → export
```

验收：护理或其他领域项目只提供普通配置、数据和可选评分插件，不写新的流程脚本。

阶段进展：`saddlellm distill --eval-data ...` 已先完成蒸馏专用闭环。它会自动识别选择题或
completion 数据，顺序评估蒸馏前基线和训练后学生，并保存逐条结果、配对比较 JSON 与
Markdown；在线教师只采集待审核数据时明确延后评估。这里没有复用训练集冒充独立效果。
剩余工作是把同一契约下沉到通用 `eval` stage，覆盖普通 SFT、偏好训练和预训练候选。

### P1：领域评分插件协议

通用框架已经能处理 PPL、选择题和配对统计，但病例状态一致性、SBAR、OSCE、安全红队等应通过统一 evaluator 插件接入。插件输出必须保留逐条证据和一票否决项，不能只返回一个平均分。

验收：护理插件位于领域项目；SaddleLLM 核心不出现护理学院、医院或比赛专用字段。

### P1：生产推理后端与可观测性

内置 Transformers 服务定位为功能验证/低并发。生产路径仍需：

- 与执行引擎解耦，接入连续批处理后端；
- 客户端断开后的生成取消；
- 队列长度、首 token 延迟、tokens/s、P50/P95/P99、错误率和显存指标；
- 网关鉴权、限流、审计与30分钟固定口径压测。

验收：不能用 `max_concurrency` 配置值代替实际并发报告。

建议保持 `saddlellm serve` 和 OpenAI 兼容接口不变，只增加 `--engine auto|transformers|vllm`。Windows/单机 smoke 使用内置 Transformers；Linux 生产部署优先 vLLM，利用异步调度、连续批处理和可选前缀缓存。前缀缓存只优化共享长前缀的 prefill，不会让长答案的逐 token 解码凭空加速。

### P1：正确暴露 SFT 与 LoRA 的高价值选项

- 对 conversational 数据支持 `assistant_only_loss`，对 prompt-completion 数据保持 `completion_only_loss`，并在不兼容模板上直接报错；
- 暴露 `packing` 作为可选吞吐优化，但默认关闭，避免短期内改变已验证训练语义；
- LoRA 增加 `target_modules=all-linear`、rsLoRA、DoRA 等可选实验项，全部通过同一验证集和封存集比较，不能默认宣称更好；
- 先修数据覆盖和评测，再做 rank、alpha、学习率小规模扫描；当前 0.6B 模型不应通过单纯增加 step 追求碰运气提升。

官方依据：TRL 支持 conversational/prompt-completion 数据、assistant/completion-only loss 与 packing；PEFT 提供 rsLoRA、DoRA；vLLM 提供 OpenAI 兼容服务及自动前缀缓存；MedHELM 强调按真实临床任务、任务匹配指标和可复现证据评估，而不是只看医学考试或 PPL。

### P1：看板跨运行比较

当前看板擅长查看单次训练，下一步应读取 experiment manifest，按相同数据和协议并排显示基座/候选、checkpoint 曲线和门禁结论。外部 gate 不应靠模糊目录搜索，应由 manifest 明确引用。

### P2：发布工程

- 在干净 Windows/Linux 环境构建 wheel 并运行 CLI 契约测试；
- 固定 post-training 依赖锁和兼容矩阵；
- 为 schema 变更提供版本迁移和向后兼容测试；
- 清理历史模块命名、行宽和 import 风格，但不以大规模格式化干扰功能修改。

## 3. 暂不增加的复杂度

- 不再创建第二套 Gradio/React 前端；
- 不把领域数据下载和护理规则写入 SaddleLLM 核心；
- 不在普通 `model.generate` 上堆参数假装实现生产百并发；
- 不用训练 loss 或 PPL 自动替代医学/领域专家发布决策。

## 4. 当前推荐命令

```powershell
saddlellm train config.yaml
saddlellm dashboard outputs/run --open
saddlellm evaluate ...
saddlellm compare ... --output outputs/run/release_gate.json
saddlellm export <checkpoint> outputs/release --gate-result outputs/run/release_gate.json --require-gate
saddlellm serve outputs/release --open
```

旧的 `evaluate-model`、`compare-evaluations`、`export-model`、`serve-model` 仅作为兼容别名保留。
