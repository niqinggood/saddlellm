# 路线三：评测、导出与重新部署

接续 [LoRA 教程](TRAIN.md)。以下 `./my-adapter` 必须替换成训练 summary 中的实际 `results.sft.model_path`。
使用训练环境运行评测/导出；部署可使用独立 `[serve]` 环境。

## 1. 同一验证集比较

```text
saddlellm evaluate ./models/Qwen3-0.6B examples/quickstart/sft.validation.jsonl --task completion --local-files-only --generation-all --max-new-tokens 64 --output outputs/base-validation.json
saddlellm evaluate ./models/Qwen3-0.6B examples/quickstart/sft.validation.jsonl --adapter ./my-adapter --task completion --local-files-only --generation-all --max-new-tokens 64 --output outputs/candidate-validation.json
saddlellm compare --base-validation outputs/base-validation.json --candidate-validation outputs/candidate-validation.json --output outputs/validation-comparison.json --markdown-output outputs/validation-comparison.md
```

检查 PPL 与逐条实际回答。两次评测应使用同一设备、精度、模板和生成参数。
validation-only 比较用于选择候选，`accepted=false` 是预期行为，并非命令坏了；差的候选可能导致 compare 非零退出。
这 4 条样例不能证明通用模型质量，更不能替代业务评测。

## 2. 本地功能试用导出

```text
saddlellm export ./my-adapter outputs/first-release --local-files-only
saddlellm serve outputs/first-release --backend local --local-files-only --no-chat-history --open
```

导出默认把 LoRA 合并到基座，可能需要额外内存。上面没有使用质量门禁，仅用于跑通交付链，不能把导出成功解释为质量验收通过。
服务启动后重新发送一条聊天请求，确认模型和 tokenizer 都能重载。

## 3. 正式模型发布

在训练开始前另行准备封存测试集；选择好 checkpoint 后，对基座和候选分别评测该测试集，再把 `--base-test`、`--candidate-test` 加入 compare。
只有满足设定指标、得到 `accepted=true` 的报告，才应传给：

```text
saddlellm export ./my-adapter outputs/approved-release --gate-result outputs/release-gate.json --require-gate --local-files-only
```

评测不通过应保留失败记录，不要通过放宽阈值伪造通过。质量验收之外仍需做目标硬件上的服务容量与稳定性测试。
