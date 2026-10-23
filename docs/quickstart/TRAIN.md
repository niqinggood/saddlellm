# 路线二：第一次 LoRA 微调

前提：在独立环境安装 `[posttrain]`，准备本地 `./models/Qwen3-0.6B`。源码中的 `examples/quickstart` 附带 8 条训练和 4 条验证数据。
如果只收到 wheel，请同时取得该样例目录；示例不是生产数据，也不能用 5 步训练宣称效果提升。

每行是一个 JSON 对象，基本格式为：

```json
{"instruction":"把 hello 改成大写。","output":"HELLO"}
```

## 检查依赖、数据和训练计划

```text
saddlellm doctor --profile train
saddlellm inspect-data examples/quickstart/sft.train.jsonl --task sft --validation-data examples/quickstart/sft.validation.jsonl
saddlellm train sft --model ./models/Qwen3-0.6B --data examples/quickstart/sft.train.jsonl --validation-data examples/quickstart/sft.validation.jsonl --method lora --lora-target-modules all-linear --max-steps 5 --batch-size 1 --gradient-accumulation-steps 1 --max-seq-length 256 --save-steps 5 --eval-steps 5 --local-files-only --output-dir outputs/first-lora --dry-run
```

检查数据报告后处理错误；样例很小，集中度/规模告警不应被当成生产数据合格证明。
删除末尾 `--dry-run` 后执行相同命令，才会真正训练。程序会按现有设备选择精度；CPU 排错可显式使用 `--precision fp32`，训练速度可能很慢。
不要反复覆盖正式实验目录；每次实验使用新目录。需要恢复训练时使用已有的 `--resume-from-checkpoint`，不要把重跑当作续训。

## 找到产物

查看 `outputs/first-lora/training_summary.json` 中 `results.sft.model_path`，以该字段为准找到最终 adapter 目录。
其中应包含 `adapter_config.json` 和 adapter 权重；日志/checkpoint 目录不一定就是最终可部署目录。
LoRA adapter 不是完整模型，仍依赖配置中引用的基座模型。移动机器时也需要准备基座或先合并导出。

下一步按 [评测与部署路线](EVALUATE_DEPLOY.md) 对比基座和候选。训练 loss 下降不等于实际回答变好。
