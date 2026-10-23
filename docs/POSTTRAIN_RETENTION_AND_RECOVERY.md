# 后训练能力保持与压缩恢复

这套流程将 SFT 回放、参考模型 KL、统一能力回归和剪枝后恢复接到现有训练与 CLI。新增功能默认关闭，不改变已有配置的训练目标。示例命令从仓库根目录运行，需要先准备自己的模型与独立数据。

## SFT 数据回放与参考模型约束

```powershell
python -m saddlellm.cli train --stage sft --model outputs/base --data examples/retention/business.train.jsonl --validation-data examples/retention/business.validation.jsonl --replay-data examples/retention/replay.train.jsonl --replay-ratio 0.25 --retention-kl-weight 0.1 --method lora --max-steps 100 --output-dir outputs/retention-sft --local-files-only
```

`--replay-ratio` 表示回放样本占最终训练样本的比例，不是相对主任务数据的比例。按 `ceil(N*r/(1-r))` 计算回放数量，整数取整后的实际比例记入报告。也可用 `--replay-count` 指定数量，两者不能同时启用。固定 seed 后无放回采样；经过过滤的独立样本不足时明确报错。比例按格式化/分词后的训练样本计算，多轮对话可能产生多个样本。

原始主任务的自动训练/验证拆分会按分组、提示与内容的传递关联划分；手动指定的主任务验证集保持用户分配，由用户保证独立。回放在划分后加入，并排除与验证集共享分组、提示或内容的记录，以及相同 token 输入。HF 路径使用实际 TRL 完成 EOS 处理、分词和截断后的输入检查碰撞，避免原文不同但截断后完全相同的样本混入。保存的 `replay_manifest.json` 记录数据指纹、过滤统计、seed 与实际混合比例。业务评测集和通用回归集也应由用户在数据准备时排除，训练器无法推断未提供的外部评测数据。

`--retention-kl-weight` 默认 0，启用后目标为 `CE + weight * T² * KL(reference || student)`。`--reference-model` 默认采用 SFT 起始检查点，`--retention-temperature` 默认 1。KL 只覆盖移位后的有效监督 token，忽略 prompt 和 padding；验证 loss 保持原来的任务 CE。参考模型冻结、使用 eval 模式，不接收梯度；要求 token ID 语义一致，不能用不同 tokenizer 的模型直接做 logits KL。

HF 和原生 Saddle SFT 都支持回放与 KL。KL 目前支持单设备/DDP，每个进程持有自己的参考模型，统计报告标明 local-rank；FSDP、DeepSpeed 和 Megatron 组合会明确拒绝。KL 需要额外的教师显存与完整 logits，原生分块 CE 在相应训练调用中会切换到完整 logits。LoRA 本身不提供防遗忘保证。

完整 YAML 示例是 [posttrain_retention.yaml](../configs/posttrain_retention.yaml)。先验证、预演，再实际训练：

```powershell
python -m saddlellm.cli validate-config configs/posttrain_retention.yaml
python -m saddlellm.cli train configs/posttrain_retention.yaml --dry-run
python -m saddlellm.cli train configs/posttrain_retention.yaml
```

## 固定基线的多任务回归

```powershell
python -m saddlellm.cli regression examples/retention/suite.yaml --baseline outputs/base --candidate outputs/base --candidate-adapter outputs/retention-sft/sft_checkpoints --output outputs/retention-regression.json
```

上例对应前面的 LoRA 训练：`sft_checkpoints` 是保存的 adapter，父目录只是运行日志目录。完整模型可直接用 `--candidate MODEL_DIRECTORY`；具体产物路径以训练命令输出为准。

suite 支持 `multiple-choice`、`completion` 和 `rubric`。每个任务单独指定数据、split、推理设置，以及需要比较的指标。套件内的数据路径相对 suite 文件解析。`max_drop` 是指标的绝对退化预算，例如 accuracy 的 0.02 表示两个百分点；`higher_is_better: false` 可用于 NLL/PPL。可同时配置候选模型的绝对 `min`/`max`。

报告记录每项 baseline/candidate/delta/drop、完整原始评测、数据和设置指纹、模型内容身份。缺失或非有限指标、数据/设置/模型身份变化均为 invalid，不会通过。completion 的 NLL/PPL 比较还要求分词语义与模板一致；签名绑定原始文件，只规范化经过验证的等价运行时截断/填充缓存，实际词表、模板或处理器变化仍会拒绝。当前 gate 使用可绑定内容身份的本地检查点；请先将远程模型下载成固定版本的本地目录。

CLI 通过返回 0，退化超标或证据无效返回 1；`--dry-run` 只检查计划，不加载模型。Python 可用 `compare_regression_suite(suite, baseline_reports, candidate_reports)` 重比 suite 生成的原始报告，不需要重新推理。多次用同一验证集选择配置后，最终还应使用独立测试集。

在训练 YAML 的 `eval.regression_suite` 配置该套件，即可让 eval 阶段比较起始模型与最新训练产物。`eval.regression_baseline` 可显式指定起始基线，`eval.regression_fail_on_rejection` 默认 true。套件中的任务规则承担 gate 职责，不与旧的 `eval.gate.enabled` 同时配置。即使设置评测失败后继续，未通过的 suite 仍会阻止导出；通过后也会核对导出模型内容身份。dry-run 的导出计划会标记 gate 尚待执行。

## 剪枝后的恢复训练

先准备已完成后训练的高精度完整检查点；LoRA 模型应先按需要合并导出。下面用 `outputs/posttrained-float` 表示该检查点。剪枝的 `--target-modules` 应与模型结构对应。

```powershell
python -m saddlellm.cli prune outputs/posttrained-float outputs/pruned --ratio 0.15 --local-files-only
python -m saddlellm.cli recover outputs/pruned examples/retention/business.train.jsonl outputs/recovered --teacher outputs/posttrained-float --validation-data examples/retention/business.validation.jsonl --max-steps 100 --distill-weight 0.5 --local-files-only --suite examples/retention/suite.yaml
```

恢复会加载现有学生权重，目标为 `(1-w)*CE + w*T²*KL(teacher || student)`；不提供教师时使用 CE。教师应选择想要保留其行为的压缩前模型，token ID 必须兼容。训练按组划分验证集，显式验证文件与训练数据有共享组/提示时拒绝。`--validation-split` 默认 0.1；提供 `--validation-data` 时默认置 0，不能再同时启用切分比例。

分词后再次使用统一切分审计，同时检查完整有效输入和监督答案之前的提示。即使原文不同、答案不同，只要实际有效提示相同，也会在优化前拒绝；padding 不参与身份比较。成功报告的 `data.split_audit` 保存审计计数和散列，不包含原始提示。

剪枝掩码在梯度与每次优化更新后执行，防止被剪权重重新长出来；精确掩码随剪枝和恢复检查点保存，支持重复恢复。旧剪枝检查点只有 manifest 和实体零值时，会保守地冻结声明目标权重中的所有零值，并在报告中说明掩码推断方式。`recovery_report.json` 保存实际步数、训练损失、恢复前后验证 CE、稀疏率与数据来源。`--suite` 会继续生成 `regression_report.json`，相对 `--baseline`、`--teacher` 或输入模型（依次回退）验收；未通过返回非零。

新检查点若已声明精确掩码，文件缺失、参数清单不完整或掩码与实体权重不一致都会拒绝；不会改用零值推断来掩盖损坏。`ModelPruner.save()` 使用当前原始参数乘掩码保存，支持优化后立即落盘；加载后再次剪枝会保留之前的掩码，天然零值仍可在恢复训练中更新。

恢复目前是单设备、本地数据驻内存的训练，支持 HF 和原生检查点。输出必须使用新的目录，不覆盖输入或教师。命令级 `--output` 报告放在检查点目录外。

## 量化边界与验收

量化现在可通过 `apply_quantization(..., config={"ptq": ..., "qat": ...})` 选择 AWQ/GPTQ，或专用的 `portable-int8-weight-only` QAT 路线；默认均关闭。配置、依赖、导出格式与加载边界见 [函数式量化入口](QUANTIZATION_FUNCTION_API.md)。既有 CLI 默认入口仍为 CPU 动态 INT8。`recover` 继续拒绝打包的量化产物，需先恢复量化前的浮点/剪枝检查点，再选择目标量化路线。

```powershell
python -m saddlellm.cli quantize outputs/recovered outputs/recovered-int8 --local-files-only
python -m saddlellm.cli regression examples/retention/suite.yaml --baseline outputs/posttrained-float --candidate outputs/recovered-int8 --output outputs/final-compression-regression.json
```

普通恢复微调、QLoRA 和 QAT 是不同流程。真正的 QAT 需要与目标后端匹配的 fake quant 数值、转换和部署内核；不能把这个恢复命令标记成 QAT。

建议保留高精度基线，在剪枝恢复后、最终量化后分别运行 suite，并在目标硬件验证显存、首 token 延迟、解码吞吐与并发。权重变零不代表已有稀疏内核，INT8 模型体积变小也不代表所有负载都加速。附带的几条样例仅验证接线，不能证明模型的通用能力或生产发布质量。
