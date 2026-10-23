# 通过函数配置选择 AWQ/GPTQ 或 QAT

统一入口是 `saddlellm.apply_quantization`，也可从 `saddlellm.compression` 导入。函数接受模型路径或内存中的模型，使用 `config.ptq.enabled` 与 `config.qat.enabled` 决定是否执行对应流程。默认均为 false。

```python
from saddlellm import apply_quantization

result = apply_quantization(
    model, tokenizer,
    config={"ptq": {"enabled": False}, "qat": {"enabled": False}},
)
assert result.model is model
```

关闭时不加载模型、训练运行时或可选量化库，不保存文件，也不改变输入模型。启用时会修改传入的内存模型；需要保留原模型时，传检查点路径，或由调用方保留副本。`dry_run=True` 只验证配置和输出路径，不执行训练或校准。

`result.model` 和 `result.tokenizer` 是处理结果；`result.report` 是可序列化执行报告，`result.output_dir` 是产物目录。`result.to_dict()` 只返回报告，不序列化模型对象。

## AWQ / GPTQ：训练后校准量化

AWQ 根据激活统计保护重要通道，GPTQ 根据校准数据估计逐层量化误差并进行补偿。这里两种算法都通过 LLM Compressor 执行，输出统一为 **compressed-tensors**，不冒充 legacy AutoAWQ/AutoGPTQ 文件布局。

```python
result = apply_quantization(
    "outputs/posttrained-float",
    output_dir="outputs/model-awq",
    config={
        "ptq": {
            "enabled": True,
            "method": "awq",        # 换成 "gptq" 选择 GPTQ
            "bits": 4,
            "group_size": 128,
            "calibration_path": "data/calibration.jsonl",
            "max_samples": 128,
            "max_seq_length": 512,
            "seed": 42,
            "device": "auto",
            "local_files_only": True,
        },
        "qat": {"enabled": False},
    },
)
```

校准数据支持本地 JSON/JSONL/TXT，或 `calibration_data` 内存序列；路径字段和内存字段不能同时设置。数据可以是 `text` 或常见 instruction/output/messages 格式。样本选择、去重与来源指纹记录在报告中。使用与部署任务、语言和长度分布接近的数据，另留独立回归集。

`group_size` 必须整除目标投影的输入维度；小模型可改为 8/16 等。`targets` 默认选择标准 `Linear`，`ignore` 默认包含 `lm_head`。GPTQ 可设置 `dampening_frac`、`block_size`、`actorder`；AWQ 的模型族必须有后端支持的映射。不是所有自定义或原生模型都能直接使用这一 HF 校准后端。不支持的结构、设备、依赖或实际导出格式会明确报错，不会回退成普通 INT8 并报告成功。

为兼容固定的 Transformers 版本，默认 `pipeline="basic"`，通过真实前向收集校准统计，不依赖 FX 符号跟踪。它可能需要更多校准内存。`sequential` 等模式仅应在相应模型与库版本经过验证后选择；不能把 tiny-model 验证外推为大模型的内存可行性结论。

## QAT：明确数值格式的量化感知训练

目前实现的 profile 是 **`portable-int8-weight-only`**：逐输出通道对称 INT8 权重，训练时使用与最终转换相同的 fake quant 公式和 STE；激活、偏置与其他参数保留 FP32。转换后保存实际 INT8 权重与 scale，重载时按相同公式反量化后做浮点矩阵计算。

这是可训练、转换和重载的参考后端，**没有 INT8 优化内核的速度承诺**，也不是 GPU INT4、CPU dynamic INT8 或 QLoRA。它支持标准 HF 投影、GPT2 `Conv1D` 和原生 Saddle 模型；自定义算子不能仅因为继承 `Linear` 就被替换。

```python
result = apply_quantization(
    "outputs/posttrained-float",
    output_dir="outputs/model-qat",
    config={
        "ptq": {"enabled": False},
        "qat": {
            "enabled": True,
            "backend": "portable-int8-weight-only",
            "train_data": "data/recovery.train.jsonl",
            "validation_data": "data/recovery.validation.jsonl",
            "validation_split": 0,
            "max_steps": 100,
            "batch_size": 1,
            "learning_rate": 2e-5,
            "max_seq_length": 512,
            "device": "cpu",
            "local_files_only": True,
        },
    },
)
```

可选 `teacher_checkpoint`、`distill_weight` 与 `temperature` 加入教师蒸馏；未提供教师时使用 CE。QAT 会再次更新参数，训练数据也应覆盖需要保留的通用/历史任务，并用独立能力回归检查遗忘。复用恢复训练的独立验证划分与泄漏检查，并保留剪枝掩码。输入必须是可训练的浮点完整模型；PEFT adapter 应先合并，已经打包的量化产物不能重复训练。

两组配置当前是不同数值/导出格式的替代路线，**同时启用会在加载模型前报错**。不会暗中按“QAT 后再跑 AWQ”串接不同量化规则。需要新 profile 时，应一并实现其 fake quant、转换、存储与加载规则，再加入入口配置。

## 类型化配置、加载与评测

同样支持 dataclass：

```python
from saddlellm import PTQConfig, QATConfig, QuantizationConfig

config = QuantizationConfig(
    ptq=PTQConfig(enabled=False),
    qat=QATConfig(enabled=False),
)
```

使用统一加载器重载：

```python
from saddlellm.models.ModelLoader import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(
    "outputs/model-qat", device="cpu", dtype="float32", local_files_only=True,
)
```

加载器按 manifest 的 `method/backend/format` 分流。QAT 使用自己的 INT8 权重格式；AWQ/GPTQ 使用 compressed-tensors 的官方解压 API 构建本地 HF 模型，便于评测和普通生成。固定版本组合的 HF 自动量化加载存在 meta tensor 兼容问题，请使用上述 SaddleLLM 加载器，或 `saddlellm.compression.load_calibrated_ptq`；不要直接调用 HF `AutoModelForCausalLM.from_pretrained` 加载本次 PTQ 产物。这条**解压加载**路径不承诺低比特内核的显存或吞吐收益。当前 SaddleLLM 的 vLLM 路由仍不自动接收这些带 manifest 的产物，不能据此声称完成 vLLM 压缩部署验收。

可继续用既有 `regression` suite 比较最终产物，验证业务与通用能力。保存成功、权重变小或一次 tiny-model 测试通过，都不能代替真实模型和目标硬件验收。

QAT 与恢复训练复用统一切分审计：原始记录分组后，还会按实际分词、截断和 padding 处理后的输入及提示检查训练/验证重叠。相同提示搭配不同答案也会拒绝，检查发生在替换 fake-quant 模块和优化之前。成功报告的 `data.split_audit` 记录审计结果。

## 可选依赖与 CLI

默认路径不需要安装这些额外依赖。QAT 使用 `saddlellm[qat]`；AWQ/GPTQ 使用 `saddlellm[ptq]`。当前兼容组合固定为 Torch 2.6、Transformers 4.57.6、LLM Compressor 0.6.0 和 compressed-tensors 0.10.2；Windows 对应 Triton 3.2。建议为 PTQ 使用独立环境，避免改变已经验证的训练/服务环境。

CLI 是上述函数的薄封装，JSON/YAML 使用相同的 `ptq` / `qat` 两组配置：

```powershell
python -m saddlellm.cli quantize outputs/posttrained-float outputs/model-awq --config configs/quantization_awq.yaml --dry-run
python -m saddlellm.cli quantize outputs/posttrained-float outputs/model-awq --config configs/quantization_awq.yaml
python -m saddlellm.cli quantize outputs/posttrained-float outputs/model-gptq --method gptq --calibration-data data/calibration.jsonl --local-files-only
python -m saddlellm.cli quantize outputs/posttrained-float outputs/model-qat --config configs/quantization_qat.yaml
```

现有 `quantize` 默认 CPU dynamic INT8 入口保留。旧的顶层 `quantize_model` 类别名也保持兼容；新配置式函数请使用 `apply_quantization`。

## 接入训练流水线

训练配置支持可选 `quantize` 阶段，顶层 `quantization` 中的 `ptq` / `qat` 与函数入口使用相同配置。例如在已有 SFT 配置中增加：

```yaml
stages: [sft, quantize, eval, export]
quantization:
  merge_lora: true
  ptq:
    enabled: true
    method: awq
    calibration_path: data/calibration.jsonl
    local_files_only: true
  qat:
    enabled: false
```

两组开关默认关闭，关闭时阶段记为 `skipped`，不生成量化模型，后续仍使用训练结果。开启时必须显式列入 `quantize` 阶段，并排在模型训练之后、评测和导出之前。当前只支持单进程；Megatron 流水线尚未适配这个阶段。

默认从最近完成的语言模型训练阶段取检查点；也可设置 `quantization.model_path`。输出默认为运行目录下的 `quantized`，可用 `quantization.output_dir` 修改。PEFT adapter 在 `merge_lora: true` 时先以 FP32 合并，保存在量化目录的同级 `<量化目录名>-float`，保留为可恢复的浮点来源；禁止直接训练或校准未合并 adapter。

后续评测和默认导出使用最终量化检查点。`eval.adapter` 不得再叠加 adapter。发布门禁绑定该检查点的文件身份；导出保留整数权重、manifest 和 tokenizer 内容，不把量化结果重新保存成普通浮点权重。原生 QAT 使用 `export.format: saddle`，HF QAT/PTQ 使用 `hf`；不要为导出请求 dtype 转换。

`training.dry_run` / `preflight_only` 只规划，不合并 adapter、不训练、不校准，也不产生量化权重。断点恢复可跳过已完成量化；主动重跑量化仍要求新的空输出目录。若合并已完成而量化尚未产生输出就失败，重试仅在 adapter、基座和合并模型的内容指纹均未变化时复用浮点检查点；没有可验证记录时要求新目录。显式 DAG 的 `pipeline.dependencies` 必须包含训练→量化→评测→导出的依赖，确保重跑上游时下游结果失效。

完整 QAT 配置见 `configs/posttrain_quantization_qat.yaml`：

```powershell
python -m saddlellm.cli train configs/posttrain_quantization_qat.yaml --dry-run
```
