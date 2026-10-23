# Saddle 决策模型与 RLCD 学习笔记

更新日期：2026 年 10 月 3 日。

我们希望在 SaddleLLM 中训练直接返回决策与概率的自有模型。这份笔记解释 Jev 的公开训练思路、概率校准的含义和我们的独立研发路线。第一版使用冻结的 Qwen3 文本嵌入骨干、可训练 Choice 评分头和独立温度校准。新版增加独立的多类型决策网络，支持 Choice、Score、Noul、骨干联合训练和实验性 Brier 奖励策略梯度；两套模型分别保存和加载。

## RLCD 是什么

RLCD 全称为 **Reinforcement Learning for Calibrated Decisions**，可译为“面向校准决策的强化学习”。TypeSafe 将其作为 Jev 的训练路线：模型输出可由软件消费的决策和概率，并让概率反映实际结果的不确定性。[官方训练介绍](https://docs.typesafe.ai/introduction/machine-learning-primer)

| 路线 | 优化目标的侧重点 | 常见输出 |
| --- | --- | --- |
| RLHF | 人类对回答的偏好 | 对话文本 |
| RLVR | 可验证的任务结果 | 推理与答案 |
| RLCD | 结构化决策及概率与结果的一致性 | 决策值和概率 |

这张表概括 TypeSafe 对三条路线的定位，不表示每种算法只能产生一种输出，也不表示校准必须通过强化学习实现。[官方训练介绍](https://docs.typesafe.ai/introduction/machine-learning-primer)

## 什么叫概率经过校准

假设我们收集了很多被模型赋予 0.8 概率的判断。在相应的数据分布上，这些判断应当有大约 80% 成立。这是跨样本的统计性质，不是单条判断一定正确，也不是“模型语气很肯定”。[官方校准解释](https://docs.typesafe.ai/introduction/machine-learning-primer)

我们的示例：模型对 100 道题都给出 0.9 的最高选项概率，如果最终只有 60 道答对，它就明显过度自信。单独提高准确率不能证明概率已校准；反过来，均匀输出低概率也不能证明模型具有有效的决策能力。因此应同时评估准确率、概率质量和低置信度时的覆盖率。

Softmax 只是把 logits 转为概率分布；它本身不保证校准。最高概率也不等于 TypeSafe 定义的 `confidence`，后者是由分布计算的统计量。我们应给返回字段明确的数学定义，并在任务外请求上标明校准范围。[官方 confidence 定义](https://docs.typesafe.ai/confidence)

## Jev 是独立网络还是微调模型

官方发布说明介绍了专用模型架构、并行采样方式和 RLCD；技术介绍将 RLCD 与预训练语言模型的后训练路线联系起来。因此，“专用决策架构”与“复用预训练语言能力”可以同时存在。[官方发布说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [官方训练介绍](https://docs.typesafe.ai/introduction/machine-learning-primer)

本次查阅的公开材料没有给出可复现的完整层结构、具体基座权重来源、参数规模、奖励公式或训练更新算法。我们不能确定它由哪个开源模型改造，也不能据此断言全部权重从随机初始化训练。Jev 的商业服务不提供客户数据微调或 LoRA。[模型与定制说明](https://docs.typesafe.ai/models)

## 三类决策的含义

Jev 的公开接口提供三种类型，可在一个请求中对相同状态提出多个独立问题。[官方接口介绍](https://docs.typesafe.ai/introduction)

| 类型 | 判断什么 | 结果 |
| --- | --- | --- |
| Choice | 动态候选中选哪个 | 选项、候选概率与 confidence |
| Score | 按给定等级标准如何评分 | 分数、等级概率与 confidence |
| Noul | 一个陈述是否成立 | 0 到 1 的真假概率 |

问题隔离、多问题并行和共享状态编码是架构层面的要求。首轮 Saddle 基线只实现 Choice。新版使用共享状态编码和独立查询读取，同一批次的问题互不读取对方的表示；三种类型分别训练读出头，Score 还使用有序概率损失。完整接口与训练步骤见本文末尾。实现这些结构不代表获得了 Jev 的通用能力或速度。

## 首轮 Choice 基线如何训练

首轮采用以下独立研发路线，不使用“复现 RLCD”的名义：

```text
状态与问题 → 冻结文本骨干 → 状态向量
候选描述与问题 → 同一冻结骨干 → 候选向量
状态向量 → 可训练低秩残差映射 → 候选相似度与 logits
logits → Softmax → 候选概率
```

使用正确候选标签训练交叉熵损失：`L = -log p(正确选项)`。学习的是状态向量到候选空间的映射和分数尺度，没有自回归解码、词表投影或文本 JSON 解析。冻结骨干保留已有语言表示能力，第一轮只更新小型评分头。

独立温度校准采用 `p = softmax(logits / T)`，只在校准集拟合一个标量 T。正温度不改变最大 logit 的排序；它调整概率的集中程度。温度拟合在校准集上有效，也必须在独立测试集上检查效果；不能因运行了拟合函数就宣称所有任务均已校准。

| 数据集 | 唯一用途 |
| --- | --- |
| train | 更新评分头参数 |
| validation | 通过 NLL 选择评分头检查点 |
| calibration | 拟合温度，不更新网络权重 |
| test | 最后测量准确率、NLL、Brier、ECE 与选择性风险 |

校准与测试数据不参与选超参数。候选集合或问题改变后，已有温度不自动适用；服务返回 `out_of_scope` 并使用未校准概率。当前实验使用 BANKING77 的固定 77 类任务，不能证明任意问题、未见类别或护理领域能力。

## 后续如何研究强化学习决策

以下为我们的研究建议，**不是 TypeSafe 公开的 RLCD 实现**。

可以让决策网络根据状态、问题和候选产生分布，由独立标签或可靠环境反馈评估该分布，再研究使用强化学习更新决策策略。奖励不仅检查选项是否正确，还应约束概率的真实性；具体奖励、采样分布、策略更新、参考约束及训练稳定性都需单独设计和验证。

例如，Brier 分数衡量预测概率与观测结果的平方误差，负 Brier 可作为一个候选奖励设计。但“给交叉熵损失换个名字”不会变成强化学习；用 Brier 直接反向传播也仍可属于监督学习。教师概率、真实结果标签和人类偏好信号必须分别记录来源。

建议先保留监督基线，逐步补齐：

1. 输入状态、问题与候选描述的条件化建模，训练不同候选数量和候选置换。
2. 骨干联合训练，以及 Choice、有序 Score 和 Noul 的类型专用读出与损失。
3. 独立问题批处理基线，然后研究共享状态编码与问题隔离。
4. 独立概率评测与校准，量化拒判覆盖率、域外失效和部署延迟。
5. 有可靠反馈后再研究强化学习，并与监督训练加温度校准做同预算对照。

奖励优化后的效果、API 格式合法性和实际校准效果应分别验收。

## 代码与 L20 实验入口

### 统一命令行与配置

决策训练现已接入内置 Stage Registry、配置校验、流水线状态和产物记录。直接命令与配置文件使用相同执行链：

```bash
# 先检查参数和阶段计划，不加载权重或创建输出目录
saddlellm train --stage decision \
  --model /home/medmodel/models/Qwen3-Embedding-0.6B \
  --data artifacts/decision_model_20261002/data/banking77 \
  --output-dir outputs/decision --dry-run

# 使用同一命令，去掉 --dry-run 即执行训练
saddlellm train --stage decision \
  --model /home/medmodel/models/Qwen3-Embedding-0.6B \
  --data artifacts/decision_model_20261002/data/banking77 \
  --output-dir outputs/decision --epochs 40 --precision bf16

# 配置方式与数据检查
saddlellm validate-config configs/decision_choice.yaml
saddlellm train configs/decision_choice.yaml
```

这些命令需要包含本次改动的源码安装或候选 wheel，以及 `train` 依赖组。模型骨干必须已在本地；训练器不会自动下载模型。`--dry-run` 检查结构与训练参数，文件内容和跨集合泄漏检查由 `validate-config` 默认执行，也在真实训练前执行。示例数据清单位于 [BANKING77 数据描述](../artifacts/decision_model_20261002/data/banking77/dataset.json)，包含 `question`、`choices` 和四个 JSONL 分割。

[decision_choice.yaml](../configs/decision_choice.yaml) 顶层为 `stages: [decision]`。阶段参数位于 `decision` 配置块，骨干可从 `decision.backbone_name_or_path` 或 `model.name_or_path` 读取。当前阶段使用单进程，要求独立运行；自身已包含校准、评测和模型保存，不接入 causal LM 的 `eval`/`export` 阶段，也不接受 LoRA、PPO、RLCD 或精确优化器恢复参数。

默认产物路径为 `OUTPUT/decision/attempt-001/model`，完整报告为同一 attempt 下的 `report.json`。根目录保存 `command_config.json`（直接命令）、`pipeline_state.json` 和 `training_summary.json`。成功阶段可通过配置中的 `pipeline.resume: true` 恢复并跳过；失败重试或重新运行会使用新的 attempt 目录，保留已产生的模型。这是阶段级恢复，不是从中断训练步恢复优化器。

```bash
saddlellm serve-decision outputs/decision/decision/attempt-001/model \
  --device cuda --dtype bfloat16 --host 127.0.0.1 --port 8010
```

服务提供 `/health`、`/v1/decisions/schema` 和 `/v1/decisions`；最后一个接口接收 `{"state":"My card payment was declined"}` 并返回候选 ID 与概率。`--api-key-env NAME` 从环境变量读取认证 token。模型目录不包含骨干，迁移后可使用 `--backbone LOCAL_PATH` 指向相同权重；服务会核对记录的骨干文件哈希。`--dry-run` 可检查服务参数，`--no-web-ui` 关闭内置验证页。

### 已完成的初始实验

| 文件 | 职责 |
| --- | --- |
| [Model.py](../saddlellm/decision/Model.py) | 冻结编码器、动态候选评分头及权重保存重载 |
| [training.py](../saddlellm/decision/training.py) | 四分割数据检查、监督训练、验证集选型和独立校准 |
| [config.py](../saddlellm/decision/config.py) | CLI 和流水线共享的决策配置及静态计划 |
| [dataset.py](../saddlellm/decision/dataset.py) | 无需加载模型的数据格式、哈希和泄漏检查 |
| [Calibration.py](../saddlellm/decision/Calibration.py) | 温度拟合、NLL、Brier、ECE 和选择性风险 |
| [Server.py](../saddlellm/decision/Server.py) | Choice 数值接口、候选缓存与校准范围检查 |
| [test_decision_training.py](../tests/test_decision_training.py) | 泄漏阻断、训练阶段顺序、更新与保存重载验证 |

2026 年 10 月 2 日已在 L20 服务器完成第一轮实验，独立目录为 `/root/saddle-decision-20261002-cfb399f1`。使用服务器现有 Qwen3-Embedding-0.6B，在 GPU 上提取特征，在 CPU 上训练 131,073 个评分头参数；骨干保持冻结。训练、特征提取与最终评测合计约 68.9 秒，不包含环境准备、骨干哈希计算和服务加载。运行参数与源码快照标识见 [remote_run.json](../artifacts/decision_model_20261002/l20_training/remote_run.json)。

数据分割为 train 7,996 条、validation 1,007 条、calibration 989 条和 test 3,080 条。按验证 NLL 选择第 34 轮权重，温度为 1.0391。以下全部为同一官方测试集上的结果：

| 模型 | 准确率 | NLL | Brier | ECE |
| --- | --- | --- | --- | --- |
| 冻结骨干余弦基线 | 62.69% | 1.68634 | 0.60445 | 28.46% |
| 余弦基线加独立温度校准 | 62.69% | 1.36839 | 0.50197 | 2.78% |
| 训练评分头，未做温度校准 | 91.69% | 0.32923 | 0.12707 | 1.35% |
| 训练评分头，加独立温度校准 | 91.69% | 0.32918 | 0.12742 | 1.73% |

训练评分头提高了该任务准确率约 28.99 个百分点。独立温度拟合对已训练模型的测试 NLL 只有很小改善，同时 ECE 与 Brier 略变差；不能将这一步描述为所有概率指标均提升。ECE 使用 15 个等宽最高概率分桶，Brier 为每条样本对全部类别误差平方求和后取平均。完整逐类统计、可靠性分桶与拒判覆盖率见 [report.json](../artifacts/decision_model_20261002/l20_training/report.json)。

已验证权重保存重载、候选置换时概率与 ID 对应一致，以及换问题后校准状态返回 `out_of_scope`。50 条测试样本的真实本机回环 HTTP 请求：预热后 P50 15.83 ms、P95 16.09 ms；首次请求约 431 ms，包含首次候选编码和缓存建立，不包含模型加载。GPU 额外峰值 reserved 约 1,166 MiB。这是共享 L20、短文本、单请求、候选缓存命中的小样本测量，不包含外网延迟，也不是与 Jev 的同质量对照。原始记录见 [inference_validation.json](../artifacts/decision_model_20261002/l20_training/inference_validation.json)。

服务器模型目录为 `/root/saddle-decision-20261002-cfb399f1/run/model`。评分头、校准参数和配置已取回 [本地模型目录](../artifacts/decision_model_20261002/l20_training/model/decision_config.json)，并与远端文件哈希核对。骨干未打包，部署仍需要同版本 Qwen3-Embedding-0.6B。

在服务器源码快照上复跑时使用新的输出目录：

```bash
RUN=/root/saddle-decision-20261002-cfb399f1
PYTHONPATH="$RUN" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /home/medmodel/.venv/bin/python -m saddlellm.decision.training \
  --dataset "$RUN/data/banking77" \
  --backbone /home/medmodel/models/Qwen3-Embedding-0.6B \
  --output "$RUN/run_second" --epochs 40 --encoder-batch-size 16
```

本地相关回归测试为 98 项通过，覆盖模型、数据、校准、训练和服务契约；这不代表通用模型能力或性能验收。

## 多类型决策模型训练

2026 年 10 月 3 日新增 `TypedDecisionModel`，与前述冻结骨干基线分别维护。小型原生 byte Transformer 用于无下载的训练链路验证；真实文本模型可使用本地 Hugging Face AutoModel，并联合更新骨干与三个类型的读出头。

网络在一批请求内只编码一次每个不同的状态。问题和候选分别构成查询，通过交叉注意力读取该状态；不同问题之间没有注意力连接。Choice 的候选顺序不参与打分，候选 ID 只作为返回句柄。Noul 使用专用头返回真假分布和 `probability_true`。Score 使用专用头和有序等级描述，返回等级分布、最可能等级和概率加权的 `expected_score`。

监督目标支持硬标签和软概率分布，采用交叉熵加 Brier；Score 增加 ranked probability score，约束等级的累积概率。骨干可以冻结或联合训练，训练时不缓存会随权重改变的特征。

### 多任务数据格式

数据目录含 `dataset.json` 和四个 JSONL 文件。清单示例：

```json
{"schema_version":2,"task_type":"typed","splits":{"train":"train.jsonl","validation":"validation.jsonl","calibration":"calibration.jsonl","test":"test.jsonl"}}
```

每行分别给出自己的状态、问题与候选。例如：

```json
{"id":"row-1","state_id":"case-1","task_id":"route","state":"用户询问退款。","type":"choice","question":"应转给哪个团队？","choices":[{"id":"sales","text":"销售"},{"id":"support","text":"售后"}],"label":"support"}
{"id":"row-2","state_id":"case-1","task_id":"refund","state":"用户询问退款。","type":"noul","question":"用户是否在询问退款？","label":true}
{"id":"row-3","state_id":"case-2","task_id":"urgency","state":"用户说问题已经解决。","type":"score","question":"判断紧急程度。","levels":[{"id":"low","text":"低","value":0},{"id":"high","text":"高","value":10}],"target_probs":{"low":0.9,"high":0.1}}
```

这些行只是格式示例。数据加载器要求每个分割非空，行 ID 全局唯一，同一状态的规范化文本和 `state_id` 均不能跨分割。一个 `task_id` 对应一组固定的问题与候选含义；不同任务可以使用不同候选数量。Score 的数值必须严格递增。`target_probs` 必须包含全部候选 ID，概率和为 1；Noul 的软目标使用 `false`、`true` 两个键。

软目标应来自可解释的参考分布、重复结果统计或有记录来源的教师。它不自动等于真实世界的不确定性。数据划分应在病例、客户、会话或原始文档等来源组上完成，而不只是为重复样本更换 `state_id`。

### 训练和部署入口

仓库已有 `data/decision_typed_example` 合成数据，可直接检查并训练：

```bash
python -m saddlellm.cli validate-config configs/decision_typed.yaml
python -m saddlellm.cli train configs/decision_typed.yaml
```

直接命令会从本地 `dataset.json` 识别多类型数据，不需要手工拼出 YAML：

```bash
python -m saddlellm.cli train decision --data data/decision_typed_example --output-dir outputs/typed_direct
python -m saddlellm.cli train decision --data data/decision_typed_example \
  --model /home/medmodel/models/Qwen3-Embedding-0.6B --optimizer-device cpu \
  --output-dir outputs/typed_qwen
```

第一条使用小型原生骨干，默认联合训练、batch size 16、学习率 0.001、20 个 epoch。第二条使用本地 HF 骨干，自动读取隐藏层宽度，默认联合训练、batch size 1、学习率 0.00002、3 个 epoch；它没有继承冻结评分头的 batch size 256 和学习率 0.001。用户显式给出的参数保持优先。这些是可用起点，仍需在目标数据的验证集上选择超参数。

不存在可读取数据目录时，可用 `--decision-type typed` 显式指定契约；`--decision-backbone native|hf` 选择骨干，`--decision-method supervised|reinforce_brier` 选择目标。`--dry-run` 不加载张量库或模型权重，真实执行前再校验完整数据。

默认模型路径为 `outputs/decision_typed/decision/attempt-001/model`。重复运行使用新的 attempt 目录。多类型模型目录包含完整骨干权重、读出头、tokenizer（HF 模型）、校准参数和文件哈希；部署不依赖原始骨干目录。

### 多类型训练策略和独立评估

训练配置支持给预训练骨干和新加入的 cross-attention、决策层分别设置学习率。两个字段为 `null` 时均使用原来的 `learning_rate`，因此旧配置保持原有更新方式。`head_warmup_epochs` 指定先只训练新层的轮数，此时骨干保持 eval 且不更新；后续进入联合训练。若 `freeze_backbone: true`，骨干始终冻结。

| decision 字段 | 默认值 | 含义 |
|---|---:|---|
| `query_pooling` | `mean` | 候选文本平均汇聚；`last_token` 使用最后一个有效 token，原文仍保留完整序列 |
| `instance_question_in_query` | `false` | 将记录可选的 `instance_question` 加入候选编码；固定任务问题与校准范围不变 |
| `backbone_learning_rate` | `null` | 骨干学习率，空值沿用 `learning_rate` |
| `head_learning_rate` | `null` | 新 cross-attention 和三个决策层的学习率 |
| `head_warmup_epochs` | `0` | 先只更新新层的轮数 |
| `early_stopping_patience` | `0` | 验证 NLL 连续未改善多少轮后停止；0 禁用，热身阶段不累计 |
| `evaluate_test` | `true` | 开发调参时设为 false，跳过测试集 token 预检、预测和重载校验 |

命令行也支持上述策略。例如以下命令只展示训练计划，不加载模型；这些参数是一个待验证的实验配置，并非保证有效的推荐默认值：

```bash
saddlellm train decision --model /path/to/Qwen3-Embedding-0.6B --data /path/to/typed_data \
  --query-pooling last_token --backbone-learning-rate 2e-5 --head-learning-rate 2e-4 \
  --head-warmup-epochs 1 --early-stopping-patience 2 --no-evaluate-test --dry-run
```

每轮报告包含训练和验证的分类型指标、逐类预测数与召回，以及 Score 的平均星级误差。开发训练关闭测试时，CLI 展示验证指标并注明 `metrics_split: validation`；模型仍会保存，并用验证数据检查重新加载后的结果。

对于每条原文都有不同问题的任务，可以在数据记录中提供 `instance_question`，并设置 `instance_question_in_query: true`，或在直接训练命令中加 `--instance-question-in-query`。例如 BoolQ 的固定任务问题是“判断原文是否支持下面的问题”，而实例问题是该条数据实际问的“这部电影在某年上映吗”。原始 `state` 仍保留完整原文与问题；这个开关额外把实例问题放入候选编码，以便检验输入表达方式的影响。关闭开关或缺少该字段时，候选编码与旧版本完全相同。它不改变标签、数据划分，也不构成效果保证。

独立评估现在可以直接使用 SaddleLLM 命令。它只读取已保存模型，不更新权重或重新拟合温度；输出目录必须是新目录。`--splits validation` 只预测验证集，`--splits test` 用于方案冻结后的最终测试。评估命令会先检查全部四份数据的输入长度，预测仅针对指定划分。

```bash
saddlellm evaluate-decision --model /path/to/saved_model --data /path/to/typed_data \
  --output-dir /path/to/new_evaluation --splits validation --device cuda --batch-size 4
```

结果保存在 `report.json`、`predictions.jsonl`、`logits.jsonl` 和 `token_preflight.json`。用 `--logits-reference /path/to/previous/logits.jsonl` 可核对独立重新加载；不一致时命令返回非零退出码。已看过的测试集可保留为历史回归数据，下一轮最终验收应另锁定未用于调参的留出数据。

```bash
python -m saddlellm.cli serve-decision outputs/decision_typed/decision/attempt-001/model --device cuda --no-web-ui
```

多类型 API 接收一个状态和多个独立问题：

```json
{"state":"Color: red. Intensity: high.","questions":[{"id":"red","type":"Noul","question":"Is the signal red?"},{"id":"intensity","type":"Score","question":"Rate the signal intensity.","levels":[{"id":"low","text":"low","value":0},{"id":"medium","text":"medium","value":1},{"id":"high","text":"high","value":2}]}]}
```

`max_probability` 明确定义为候选最大概率，不等于 Jev 的 confidence 指标。温度仅在独立校准集拟合，并按精确问题、类型与候选含义匹配；换问题后使用原始概率并返回 `out_of_scope`。这个范围检查不识别同一问题下的领域或输入分布漂移。

输入不会静默截断。原生模型的长度按 UTF8 字节加一个起始 token 计算；HF 模型使用其 tokenizer 的 token 长度。长文本应先合理切分，或明确增加配置长度，并重新评估显存和任务效果。

### 实验性 Brier 奖励训练

在配置中设置 `decision.method: reinforce_brier`，或使用模块入口：

```bash
python -m saddlellm.decision.TypedTraining --dataset data/decision_typed_example --output outputs/typed_reinforce --device cuda --method reinforce_brier --epochs 12 --learning-rate 0.001
```

该实验以网络 logits 为高斯策略均值，采样多个 logits 动作，将其 Softmax 分布的负 Brier 作为奖励，用 REINFORCE 和留一基线更新策略。动作采样和奖励均不反向传播，策略梯度来自动作的 log probability；因此不是将监督损失改名。默认保留监督锚点，`supervised_weight=0` 可以进行纯策略梯度对照。当前反馈来自离线标签或参考分布，没有接入在线环境，不声称复现 TypeSafe 的 RLCD。

模型初始化可通过 `initial_model` 指向已训练的多类型模型。它加载模型权重后开始一轮新训练，并重新拟合温度；不恢复优化器或旧训练进度。训练前后可分别对照 NLL、Brier、ECE 和选择性风险，不能仅以奖励升高作为上线依据。

### L20 远端验证结果

最新验证目录为 `/root/saddle-typed-20261003-020439`。远端 135 项回归测试通过，本地扩大到数据预检和 quickstart 后共 156 项通过；模型训练、真实回环 HTTP、认证、域外问题状态、完整模型保存重载均已验证。报告位于 [verification_result.json](../artifacts/typed_decision_20261003/remote_v4/remote_evidence/verification_result.json)。

原生小模型的监督与监督加 REINFORCE 对照使用相同数据、初始化种子、12 个 epoch 和 batch size 16。训练 288 行、验证 72 行、校准 72 行、测试 96 行，其中 90 行是硬标签，6 行是软目标。测试中 24 行使用未见问题措辞。

| 方法 | 硬标签准确率 | NLL | Brier | 最高概率 ECE |
| --- | --- | --- | --- | --- |
| 监督训练加温度校准 | 65.56% | 0.57550 | 0.32382 | 11.09% |
| 监督加 REINFORCE 加温度校准 | 80.00% | 0.37358 | 0.20756 | 10.68% |

NLL、Brier、ECE 使用全部 96 行及其目标分布，准确率仅使用硬标签。这是一个随机种子、少量合成信号数据的结果；不能推出真实任务或通用模型上的强化学习收益。完整报告另列三种类型和未见问题的表现。

真实 Qwen3-Embedding-0.6B 联合训练使用独立小数据集：18 行训练、9 行验证、9 行校准、12 行测试，训练 1 个 epoch。595,776,512 个骨干参数可训练，选中第 1 轮检查点；骨干 q_norm 权重的 L2 更新为 0.0011338，保存后重载的 logits 和预测与原模型一致。单轮验证 NLL 从 0.97115 降到 0.88084。这些数字用于证明训练链路实际更新了骨干，不作为模型质量验收。

共享 GPU 上，普通 GPU AdamW 在第一次更新时超过显存预算。设置 `optimizer_device: cpu` 后，FP32 Adam 主权重和状态放到 CPU，完整骨干仍在 GPU 前向和反向，实测 GPU 峰值分配为 5.11 GiB。CPU 主权重版本与普通 AdamW 在本地小模型上产生相同更新；大型模型需要额外 CPU 内存及数据传输时间。

真实回环 HTTP 每次包含三个问题，首轮预热后的 5 次请求：小型原生模型约 3.43–4.44 ms，Qwen 联合训练模型约 32.93–34.68 ms。首轮记录见 [hf_http_validation.json](../artifacts/typed_decision_20261003/remote_v3/hf_http_validation.json)，最终代码的 Qwen HTTP、CLI 默认值和 tokenizer 一致性检查见 [framework_final_validation.json](../artifacts/typed_decision_20261003/remote_v4/framework_final_validation.json)。这不含模型加载、外网或并发负载，也不是与 Jev 的同质量速度比较。测试服务完成后已停止。

已训练的 Qwen 完整模型留在 `/root/saddle-typed-20261003-020439/hf_joint/model`。本地回收了报告、配置和小模型；HF 证据目录不包含大型权重。在远端源码快照下可以复跑：

```bash
RUN=/root/saddle-typed-20261003-020439
cd "$RUN"
PYTHONPATH="$RUN:$RUN/testdeps" /home/medmodel/.venv/bin/python -m saddlellm.decision.TypedTraining \
  --dataset "$RUN/data/hf_joint_fixture" --output "$RUN/hf_joint_second" \
  --backbone hf --backbone-name-or-path /home/medmodel/models/Qwen3-Embedding-0.6B \
  --hidden-size 1024 --num-heads 8 --max-length 256 --batch-size 1 \
  --epochs 1 --learning-rate 0.00002 --optimizer-device cpu --device cuda
```

下一阶段需要可靠的真实业务标签、跨任务和领域数据，以及独立的模型质量与部署验收。当前实现和测试完成了研发链路，不表示已训练出与 Jev 等价的通用模型。
