# SaddleLLM 简明命令手册

本文面向模型训练、评估和本地部署。推荐统一使用可执行命令 `saddlellm`。

结构化决策训练使用 `saddlellm train --stage decision --model LOCAL_BACKBONE --data DATASET_DIRECTORY --output-dir OUTPUT`，或 `saddlellm train configs/decision_choice.yaml`。训练包括 Choice 评分头、独立温度校准和测试评测；`saddlellm serve-decision MODEL_DIRECTORY` 返回候选概率。当前单进程、冻结骨干，不实现 Score/Noul 或 RLCD；数据格式、参数、恢复和 L20 验证见 [决策模型训练笔记](DECISION_MODEL_NOTES.md)。

## 1. 命令、阶段和参数

```text
saddlellm <命令> [阶段] [参数]
```

- `train`、`evaluate`、`compare`、`export`、`serve` 是命令，表示要执行的动作。
- `pretrain` 是预训练阶段；`sft`、`dpo`、`orpo`、`kto`、`ppo` 是后训练阶段，它们都不是运行模式。
- `auto-posttrain` 是后训练自动选择入口；它不会选择或替代预训练。
- `--stage sft,kto` 表示严格按从左到右的顺序执行，前一阶段输出自动交给后一阶段。
- `--model`、`--data`、`--epochs` 等是参数。
- 不采用 `saddlellm --mode train`，因为一级子命令更短，帮助信息也能按动作独立展示。

旧命令 `evaluate-model`、`compare-evaluations`、`export-model`、`serve-model` 仍兼容，但新脚本推荐使用短名称。

## 2. 最简单的 SFT 训练

LoRA训练一轮：

```powershell
saddlellm train sft `
  --model Qwen/Qwen3-0.6B `
  --data train.jsonl `
  --epochs 1 `
  --output-dir outputs/my-sft
```

正式训练推荐把独立验证集直接交给同一条命令：

```powershell
saddlellm train sft `
  --model Qwen/Qwen3-0.6B `
  --data train.jsonl --validation-data validation.jsonl `
  --method lora --lora-target-modules all-linear `
  --save-steps 50 --eval-steps 50 `
  --output-dir outputs/my-sft
```

也可使用显式阶段参数，功能相同：

```powershell
saddlellm train --stage sft `
  --model Qwen/Qwen3-0.6B `
  --data train.jsonl `
  --epochs 1 `
  --output-dir outputs/my-sft
```

先检查解析结果、不训练：

```powershell
saddlellm train sft --model MODEL --data train.jsonl --dry-run
```

### 2.1 一次检查 train、validation 和 test

```powershell
saddlellm inspect-data train.jsonl --task sft `
  --validation-data validation.jsonl --test-data test.jsonl `
  --group-field case_id --strict-leakage `
  --output outputs/split_audit.json
```

同一病例/文档 group、规范化同题会阻断；近似重复默认警告，加
`--strict-leakage` 后阻断。group id 在报告中默认哈希，题目原文默认不写入报告；只有数据允许
展示时才加 `--include-content-previews`。配置使用独立 `validation_data_path` 时，
`validate-config` 和训练前置检查会自动审计 train/validation。

需要完整报错堆栈：

```powershell
saddlellm train sft --model MODEL --data train.jsonl --debug
```

## 3. 直接训练参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--stage` | 无 | 逗号分隔的有序阶段：`pretrain,sft,dpo,orpo,kto,ppo`；或单独使用`auto-posttrain` |
| `--model` | 按场景 | 继续预训练和后训练必填；从零预训练由结构配置创建 |
| `--data` | 必填 | 第一个或唯一阶段的数据 |
| `--pretrain-data` | 按需 | 多阶段命令中的连续文本预训练语料 |
| `--sft-data` | 按需 | 多阶段命令中的SFT数据 |
| `--dpo-data` | 按需 | DPO的`prompt/chosen/rejected`数据 |
| `--orpo-data` | 按需 | ORPO的`prompt/chosen/rejected`数据 |
| `--kto-data` | 按需 | KTO的`prompt/completion/label`数据 |
| `--ppo-data` | 按需 | PPO提示/rollout数据；当前仅用于规划，执行器尚未接入 |
| `--output-dir` | `outputs/<阶段序列>` | checkpoint、日志和汇总目录 |
| `--pretrain-mode` | `continue` | `continue`继承已有模型；`scratch`从结构配置创建模型 |
| `--model-config` | 无 | 从零预训练使用的已登记网络结构预设 |
| `--tokenizer` | 按场景 | 从零预训练必填；继续预训练默认继承模型tokenizer |
| `--backend` | 按场景 | 继续预训练默认`hf`，从零预训练默认`saddle` |
| `--text-column` | `text` | 预训练JSONL中的正文列 |
| `--min-text-length` | `50` | 预训练文本最小字符数 |
| `--quality-min-score` | `0.3` | 预训练语料质量阈值，范围0到1 |
| `--dedup-method` | `minhash` | `simhash`、`minhash`、`exact`或`none` |
| `--global-batch-size` | 自动 | 预训练目标全局batch；可据此推导梯度累积 |
| `--min-learning-rate` | 初始学习率的10% | 预训练cosine调度最低学习率 |
| `--method` | `lora` | `lora`、`qlora`或`full` |
| `--epochs` | `1` | 训练轮数 |
| `--learning-rate` | 按阶段 | SFT为`2e-4`，DPO/ORPO/KTO为`5e-6`；显式指定时应用到全部阶段 |
| `--batch-size` | 按阶段 | 通常为`1`，KTO默认为`2`且不允许小于2 |
| `--gradient-accumulation-steps` | `4` | 梯度累积步数 |
| `--max-seq-length` | `512` | 最大序列长度 |
| `--max-steps` | `-1` | `-1`表示按epochs训练，否则限制总step |
| `--resume-from-checkpoint [PATH]` | 关闭 | 单阶段精确续训；不写PATH时自动选择输出目录中的最新checkpoint |
| `--warmup-steps` | `10` | warmup步数 |
| `--save-steps` | `100` | checkpoint保存间隔 |
| `--eval-steps` | 有验证集时`50` | 内部验证间隔；`0`表示关闭按step验证 |
| `--logging-steps` | `10` | 指标记录间隔 |
| `--validation-data` | 无 | 独立SFT或偏好验证文件；单阶段使用，与`--validation-split`二选一 |
| `--validation-split` | `0.05` | 未给独立验证文件时，从训练数据随机划出的内部验证比例 |
| `--precision` | `auto` | `auto`、`bf16`、`fp16`或`fp32` |
| `--lora-r` | `8` | LoRA rank |
| `--lora-alpha` | `16` | LoRA alpha |
| `--lora-dropout` | `0.05` | LoRA dropout |
| `--lora-target-modules` | `q_proj,k_proj,v_proj,o_proj` | 逗号分隔的目标层，或PEFT特殊值`all-linear` |
| `--response-template` | `### Answer:` | completion-only回答标记 |
| `--local-files-only` | 关闭 | 只使用本机缓存或本地目录 |
| `--trust-remote-code` | 关闭 | 允许模型仓库自定义代码 |
| `--no-gradient-checkpointing` | 关闭 | 禁用梯度检查点以换取速度 |
| `--seed` | `42` | 随机种子 |

直接训练会把解析后的等价配置保存为：

```text
<output-dir>/command_config.json
```

因此命令行简写仍然可以复现、审计和再次训练。

有验证集时会按最低 `eval_loss` 选择并恢复最佳 checkpoint，结果写出
`validation_dataset_path`、`best_model_checkpoint` 和 `best_metric`。按 step 验证时，
`--save-steps` 必须是 `--eval-steps` 的整数倍，确保每个候选验证点都有可恢复权重。
多轮 messages 会按每个 assistant 回答展开；之前的对话只作上下文，只有当前回答参与损失，
所以 `train_samples` 可以大于 JSONL 行数。

`all-linear` 能减少不同模型架构间手写层名的错误，并提供更大的 LoRA 适配容量；代价是可训练
参数、显存和耗时增加。显存紧张时可改回明确的 Q/K/V/O 层列表或使用 QLoRA。

### 3.1 精确断点续训

```powershell
saddlellm train sft --model MODEL --data DATA `
  --method lora --batch-size 1 --max-steps 300 `
  --output-dir outputs/my-sft --resume-from-checkpoint
```

- 不写 PATH：从该阶段输出目录中解析最新 `checkpoint-*`；也可显式指定路径。
- 增大的是总 `max_steps`，不是“再训练多少步”。
- 模型、数据、方法、有效 batch 和并行规模要保持一致；batch 不一致会在训练前失败。
- 直接参数一次只能恢复一个阶段；多阶段用 YAML 的 `pipeline.resume`/`pipeline.rerun`。
- `summary.json` 中的 `resumed_from_checkpoint` 是实际恢复目录；续训后看
  `latest_train_metrics.loss` 判断最新一步，不要把底层累计 `train_metrics.train_loss`
  误解成当前步 loss。
- 只要划分了验证集，SFT/DPO/ORPO/KTO结束时都会写出最终 `eval_metrics`；
  `--eval-steps 0`只关闭训练中的周期验证，不再导致最终验证缺失。

## 4. 预训练

护理领域继续预训练的最短命令：

```powershell
saddlellm train pretrain `
  --model Qwen/Qwen3-0.6B `
  --data data/nursing_corpus.jsonl `
  --max-steps 1000 --global-batch-size 32 `
  --output-dir outputs/nursing-cpt
```

语料使用连续正文，不是SFT问答：

```json
{"text":"患者入院后，护士首先核对身份并完成初始评估……"}
```

该命令默认是`pretrain_mode: continue`，模型网络结构和tokenizer从`--model`继承。从零预训练涉及模型身份，推荐使用配置文件：

```yaml
stages: [pretrain]
pretrain_mode: scratch
model:
  backend: saddle
  config: qwen-300m
  tokenizer: Qwen/Qwen2.5-0.5B
  blueprint:
    name: nursing-dense-300m
    max_position_embeddings: 8192
    attention: {kind: gqa, num_heads: 16, num_kv_heads: 4, rope_theta: 1000000}
    ffn: {kind: swiglu, intermediate_size: 2816}
```

`model.config`是经过登记的完整预设；`model.blueprint`只写需要明确或覆盖的结构。也可以不用预设、写完整blueprint。可表达的主要维度是：模型宽度/层数/词表、MHA/MQA/GQA/MLA、SwiGLU/MoE、RoPE和上下文、残差拓扑、异构层段、causal LM/MTP目标。

开训前检查最终展开结构：

```powershell
saddlellm validate-config configs/nursing_pretrain_scratch.yaml --skip-data-inspection
saddlellm train configs/nursing_pretrain_scratch.yaml --dry-run
saddlellm train configs/nursing_pretrain_scratch.yaml
```

`validate-config`的`architecture_analysis`返回参数量、激活参数量、KV cache估算、结构风险和实验建议；训练产物保存`model_blueprint.json`和`saddle_config.json`。

## 5. 有序多阶段训练

不确定单个后训练数据文件该用哪种目标时，先运行：

```powershell
saddlellm train --stage auto-posttrain --model MODEL --data DATA --dry-run
```

它只在SFT、DPO、KTO中选择，并输出理由和数据检查结果。混合/未知格式会被拒绝，PPO不会自动选择。
`auto-posttrain`只选择后训练算法，绝不会把连续文本误判成预训练。

已经可真实运行的示例：

```powershell
saddlellm train --stage sft,kto `
  --model Qwen/Qwen3-0.6B `
  --data data/sft.jsonl `
  --kto-data data/kto.jsonl `
  --batch-size 2 `
  --output-dir outputs/sft-kto
```

执行关系固定为：

```text
基础模型 → SFT checkpoint → KTO checkpoint
```

`--stage sft,ppo,kto`同样会被解析为三段线性依赖。当前PPO稳定训练器尚未接入，因此可用`--dry-run`检查顺序；去掉`--dry-run`会在SFT开始前返回`NotImplementedError`，不会跳过PPO后直接跑KTO。真正接入PPO还必须提供reward model、value model、rollout策略和安全验证过的奖励契约。

## 6. 护理模型训练示例

```powershell
saddlellm train sft `
  --model Qwen/Qwen3-0.6B `
  --data E:\DEV\medModel\nursing_project\data\public_nursdata_v5\nursdata_train_sft.jsonl `
  --validation-data E:\DEV\medModel\nursing_project\data\public_nursdata_v5\nursdata_validation_gate.jsonl `
  --method lora --lora-target-modules all-linear `
  --epochs 1 --learning-rate 5e-5 `
  --batch-size 1 --gradient-accumulation-steps 8 `
  --max-seq-length 384 --max-steps 150 `
  --warmup-steps 10 --save-steps 50 --eval-steps 50 `
  --precision bf16 --local-files-only `
  --output-dir E:\DEV\medModel\outputs\nursing-direct-sft
```

训练期间另开终端查看曲线：

```powershell
saddlellm dashboard E:\DEV\medModel\outputs\nursing-direct-sft --open
```

## 7. 复杂训练仍使用配置

下列情况推荐 YAML/JSON：多阶段恢复、多卡、复杂数据混合、大量高级参数，或要串联多个偏好算法。单阶段断点续训可以直接使用上一节的命令行参数。

```powershell
saddlellm validate-config config.yaml
saddlellm train config.yaml --dry-run
saddlellm train config.yaml
```

`train` 支持 `.yaml`、`.yml` 和 `.json`。配置文件和直接参数不能混用，以免实际参数来源不明确。

## 8. 独立评估

评估基座：

```powershell
saddlellm evaluate Qwen/Qwen3-0.6B eval.jsonl `
  --local-files-only `
  --output outputs/base_eval.json `
  --markdown-output outputs/base_eval.md
```

评估LoRA：

```powershell
saddlellm evaluate Qwen/Qwen3-0.6B eval.jsonl `
  --adapter outputs/my-sft/checkpoint-100 `
  --local-files-only --generation-all `
  --output outputs/candidate_eval.json `
  --markdown-output outputs/candidate_eval.md
```

评估不会启动服务，也不会自动导出模型。

### 8.1 多提示模板鲁棒性

同一道客观题只换一种说法就改变答案，是常见的模板过拟合。模板文件是领域项目拥有的
普通 JSON，核心框架不包含任何医疗或护理专用文案：

```json
{
  "variants": [
    {"name": "standard", "template": "{instruction}"},
    {"name": "natural", "template": "请回答下题，只给选项字母。\n{question}\n{options}"},
    {"name": "bare", "template": "{question}\n{options}"}
  ]
}
```

```powershell
saddlellm evaluate MODEL eval.jsonl `
  --prompt-variants configs/prompt_variants.json `
  --local-files-only --batch-size 32 `
  --output outputs/prompt_robustness.json `
  --markdown-output outputs/prompt_robustness.md
```

每个模板可引用记录字段，例如 `{instruction}`、`{question}`、`{options}`。框架先按
`--limit` 选原始病例，再为每例展开所有模板。JSON 和 Markdown 会报告：

- 每种变体准确率以及最差变体准确率；
- `prediction_flip_rate`：同一题在不同问法下预测字母发生变化的比例，越低越稳定；
- `all_variants_correct_rate`：所有问法都答对的题目比例，越高越好；
- 每例预测，供基座/候选做配对比较。

若数据已经自行展开，使用 `--case-field CASE_ID --variant-field VARIANT`，并保证同一
`CASE_ID` 的标准答案一致。蒸馏命令提供对应的 `--eval-prompt-variants`、
`--eval-case-field` 和 `--eval-variant-field`。

### 8.2 开放回答 rubric 与已启动服务评估

领域项目可在每条 JSONL 记录中提供 `instruction` 和 `rubric.checks`。通用检查类型为
`contains_any`、`contains_all`、`not_contains_any`、`regex`、`not_regex`；每项可配置
`weight`、`critical` 和 `dimension`。领域规范、关键词和专家结论不写入核心框架。

```powershell
saddlellm evaluate served-model rubric.jsonl --task rubric `
  --endpoint http://127.0.0.1:8000 `
  --max-new-tokens 128 `
  --output outputs/rubric.json `
  --markdown-output outputs/rubric.md
```

去掉 `--endpoint` 并提供本地模型路径即可直接加载模型评估。需鉴权时用
`--api-key-env ENV_NAME`，环境变量保存密钥。rubric 始终评估全部选中记录，不能与
`--generation-samples/--generation-indices` 混用。空回答固定失败。`compare` 对 rubric
同时比较成对通过率、平均加权分和关键失败数；`--max-critical-failures 0` 可要求候选模型
没有关键失败。规则评估是工程筛查，不替代领域专家盲评。

## 9. 蒸馏与轻量化

### 9.1 缩小网络的 logits 蒸馏

```powershell
saddlellm distill --mode logits `
  --teacher E:\models\Qwen3-4B `
  --data data\nursing_corpus.jsonl `
  --eval-data data\nursing_validation.jsonl `
  --student-layers 12 --student-hidden-size 1024 --student-heads 16 `
  --max-steps 1000 --output-dir outputs\nursing-student
```

这是会创建更小网络的真实蒸馏：teacher 与 student 对相同 token 做 causal KL，再混合真实
标签 loss。结果记录 teacher PPL、学生训练前后 PPL、蒸馏 loss、参数量和缩减比例，并保存
可直接 `serve` 的 Hugging Face 模型。它需要同时加载两个模型；先用 `--dry-run` 检查参数。

带 `--eval-data` 时会自动比较蒸馏前教师与蒸馏后学生。默认 `--eval-task auto`；选择题输出
准确率、能力保留率和 McNemar p，completion 输出前后 PPL 与生成样例。四个产物位于
`输出目录/evaluation/`：`baseline.json`、`candidate.json`、`comparison.json`、`report.md`。
这个数据必须是未参与蒸馏的独立 validation；训练语料上的 PPL 不能证明能力提升。

### 9.2 低显存数据蒸馏

```powershell
# 在线教师只负责产生待审核数据
saddlellm distill --mode data `
  --student Qwen/Qwen3-0.6B `
  --teacher nursing-teacher `
  --teacher-url http://127.0.0.1:8002/v1 `
  --data data\nursing_prompts.jsonl `
  --domain nursing --concurrency 4 `
  --output-dir outputs\nursing-teacher-data

# 护理教师审核修订后，再做 LoRA SFT
saddlellm distill --mode data `
  --student Qwen/Qwen3-0.6B `
  --teacher-data data\nursing_teacher_reviewed.jsonl `
  --eval-data data\nursing_validation.jsonl `
  --method lora --epochs 1 `
  --output-dir outputs\nursing-distilled-sft
```

第一条命令不会自动训练。其输出为普通 `instruction/output` JSONL，同时带
`review_status=pending`。这不是多余门禁：实测 0.6B 教师会生成错误的压疮和急性肺水肿
处置，若立即 SFT 会把危险内容写回学生。审核后的第二条命令才进入统一 SFT Trainer。
第二条命令带 `--eval-data` 后，会用同一测试集和解码参数自动比较原始 student 与训练后的
LoRA/完整模型。第一条在线采集命令没有训练学生，因此评估只标记为等待审核，不制造假分数。

### 9.3 量化与剪枝

```powershell
# 可持久化、可重载的 CPU dynamic INT8
saddlellm quantize outputs\release outputs\release-int8 --local-files-only
saddlellm serve outputs\release-int8 --device cpu --open

# 完全相同的易记别名
saddlellm lightweight outputs\release outputs\release-int8

# 稀疏权重研究产物
saddlellm prune outputs\release outputs\release-pruned `
  --ratio 0.15 --method l1_unstructured
```

`prune` 会报告实际稀疏率，但普通 dense Hugging Face 后端通常不会因为零权重而缩小文件或
加速；它必须配合稀疏 kernel 才可能得到部署收益。CUDA 上推荐使用已有的运行时量化：

```powershell
saddlellm serve outputs\release --device cuda --quantization 8bit --open
```

4bit/8bit 都可能改变答案，压缩命令不替代独立 `evaluate`。

## 10. 比较、导出和服务

2.40 简化部署：`saddlellm serve MODEL` 默认 auto，兼容 Linux CUDA + vLLM 环境走高吞吐连续批处理，
其他环境明确使用 local；`--require-high-throughput` 禁止降级，`--dry-run` 只看计划。
提供 OpenAI 文本 API/SSE、真实用量、官方 SDK 调用和 Python `serve()`；
安装、命令参数、接口边界与源码链路见 [OpenAI 服务手册](OPENAI_SERVING.md)。

比较基座与候选：

```powershell
saddlellm compare `
  --base-validation outputs/base_eval.json `
  --candidate-validation outputs/candidate_eval.json `
  --output outputs/comparison.json `
  --markdown-output outputs/comparison.md
```

上面的 validation-only 比较只用于选 checkpoint。即使指标通过，也会输出
`decision=validation_passed`、`selection_eligible=true`、`accepted=false`。正式发布必须再传
`--base-test` 与 `--candidate-test`；只有 validation 和封存 test 都达到阈值，结果才是
`decision=promoted`、`accepted=true`，才能供 `export --require-gate` 使用。封存 test 解封后
不得再据此修改模型或挑 checkpoint。

整理交付模型：

```powershell
saddlellm export outputs/my-sft/checkpoint-100 outputs/release
```

直接加载checkpoint调试：

```powershell
saddlellm serve outputs/my-sft/checkpoint-100 --device cuda --open
```

运行时量化（需要 CUDA，并安装 `serve,qlora` 依赖）：

```powershell
python -m pip install -e ".[serve,qlora]"
saddlellm serve outputs/my-sft/checkpoint-100 `
  --device cuda --dtype bfloat16 --quantization 8bit --open
```

`--quantization` 支持 `none`（默认）、本地 bitsandbytes 的 `8bit/4bit` 以及实验性 Saddle 原生 `int8`（W8A16，CUDA/Triton）。
它在加载时压缩模型，不生成新的量化目录；本地 LoRA adapter 可挂载于相应基座，原生 int8 不支持独立 adapter。
量化可能改变输出，必须用 `evaluate` 或业务题集复测后再选精度，不能把“服务启动成功”等同于“效果通过”。

加载release：

```powershell
saddlellm serve outputs/release --device cuda --open
```

`serve` 只加载模型并提供Web对话和OpenAI兼容API，不执行评估。

## 11. 推荐命令链

```text
saddlellm train ...
    ↓ checkpoint
saddlellm evaluate ...
    ↓ evaluation.json
saddlellm compare ...
    ↓ comparison.json
saddlellm export ...
    ↓ release目录
saddlellm quantize ...            # 可选：CPU持久化INT8
    ↓ INT8目录
saddlellm serve ...
```

每条命令只做一件事。某一步失败时停在该步骤，修复后可单独重跑，不必把整个流程重新执行。

模型相关动作应全部从SaddleLLM进入：`train --method lora/qlora`负责轻量微调，`distill`负责学生模型或教师数据蒸馏，`quantize/prune`负责显式压缩产物，`evaluate/compare`负责效果证据，`export`负责交付目录，`serve --quantization`负责GPU推理时4/8bit轻量化和部署API。业务Web只调用这个API或启动该命令，不应再各自实现Transformers/PEFT模型加载。

## 12. 查看帮助

```powershell
saddlellm --help
saddlellm train --help
saddlellm distill --help
saddlellm quantize --help
saddlellm prune --help
saddlellm evaluate --help
saddlellm compare --help
saddlellm export --help
saddlellm serve --help
```
