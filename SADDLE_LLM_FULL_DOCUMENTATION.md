# SaddleLLM 完整文档

版本：`2.30.0`

仓库路径：`E:\saddle-ml-llm\build-saddle-llm`

本文档面向学习、二次开发和实际使用。它不是 API 自动生成文档，而是按训练工厂的业务逻辑解释整个代码包如何组织、每条链路怎么跑、各模块承担什么职责。

## 1. 项目定位

SaddleLLM 当前定位是：

```text
传统文本 LLM 训练工厂
```

它重点解决的是：

- 从领域数据到训练计划的组织问题
- 从模型结构想法到可运行实验的落地问题
- 从预训练、继续预训练、SFT、偏好训练、RL scaling 到评估报告的串联问题
- 从单卡、小规模实验到 DeepSpeed/Megatron/Colossal 风格后端规划的衔接问题

它不主张重复造完整 DeepSpeed、Megatron-LM、TorchTitan 这类训练基础设施，而是把这些系统当作执行后端或参考对象。SaddleLLM 更应该做模型结构、训练工厂、实验规划、数据治理和领域模型工作流。

## 2. 当前能力概览

### 2.1 传统文本 LLM 主线

已经具备：

- 领域训练工厂
- 数据 manifest
- 数据混合规划
- tokenizer 训练与评估
- scratch pretraining 实验规划
- 模型结构 blueprint
- 自研模块化 decoder-only LLM
- Dense GQA / MLA / MoE / MTP / 长上下文结构实验
- 成本估算：参数量、active 参数、KV cache、训练 FLOPs
- SFT 数据统一
- DPO/ORPO/KTO 数据统一
- SFT/Preference 配置生成
- backend parallel plan
- 训练报告和模型指标

### 2.2 多模态入口

多模态目前只是轻量入口，不是当前主线。

已有：

- 图文数据标准化
- vision backbone registry
- projector
- `MultimodalForCausalLM`
- `mllm_sft` planning stage

但没有完整 VLM Trainer，不建议近期继续扩张。

### 2.3 后端规划

已有规划能力：

- single
- torch DDP
- torch FSDP
- DeepSpeed ZeRO-2
- DeepSpeed ZeRO-3
- hybrid parallel
- MoE hybrid parallel

注意：规划不等于完全执行。Hybrid TP/PP/CP/EP 更适合作为 Megatron/Colossal adapter 的配置来源。

## 3. 推荐学习路线

建议按下面顺序阅读：

```text
WORKLOG.md
→ SADDLE_LLM_CODE_GUIDE.md
→ saddle_llm/TrainingFactory.py
→ saddlellm/training/TrainingOrchestrator.py
→ saddle_llm/TrainingOrchestrator.py
→ saddle_llm/ModelBlueprint.py
→ saddle_llm/SaddleModeling.py
→ saddle_llm/ModelExperimentPlanner.py
→ saddle_llm/PostTrainingData.py
→ saddle_llm/PeftSFTTrainer.py
→ saddle_llm/Preference.py
→ saddle_llm/DataPipeline.py
→ saddle_llm/DatasetManifest.py
→ saddle_llm/FactoryBackendPlanner.py
→ saddle_llm/BackendAdapters.py
```

阅读逻辑：

1. 先看 `TrainingFactory.py`，理解最高层用户入口。
2. 再看 `TrainingOrchestrator.py`，理解统一配置和 stage 如何执行。
3. 再看 `ModelBlueprint.py` 和 `SaddleModeling.py`，理解模型结构。
4. 最后看数据、后训练、后端、监控等辅助模块。

## 4. 安装与环境

### 4.1 本地开发安装

在仓库根目录执行：

```powershell
pip install -e ".[posttrain]"
```

检查版本：

```powershell
python tools/check_release_consistency.py
```

当前应返回：

```text
2.33
```

### 4.2 基础导入检查

```python
import saddle_llm

print(saddle_llm.__version__)
```

### 4.3 建议环境

基础功能需要：

- Python 3.10+
- PyTorch
- Transformers
- Datasets
- TRL
- PEFT

可选能力：

- DeepSpeed
- Accelerate
- bitsandbytes
- xFormers
- TransformerEngine
- flash-attn
- sentencepiece
- tokenizers

## 5. 顶层目录说明

仓库根目录主要文件：

```text
build/                         构建输出
dist/                          打包输出
external_research/             外部框架调研代码
outputs/                       训练/实验输出
saddlellm/                     核心 Python 包
saddle_llm/                    旧包名兼容层
pyproject.toml                 发布元数据与依赖分组
setup.py                       构建前端发现兼容层
WORKLOG.md                     开发日志
SADDLE_LLM_CODE_GUIDE.md       代码学习指南
SADDLE_LLM_FULL_DOCUMENTATION.md 当前完整文档
```

## 6. 核心概念

### 6.1 Factory

Factory 是最高层入口，代表一个领域模型训练工厂。

对应文件：

```text
saddle_llm/TrainingFactory.py
```

它负责生成：

- workspace
- 数据计划
- backend plan
- pretrain experiments
- model architecture experiments
- post-training plan
- report

### 6.2 统一配置

训练入口只接受一份普通 YAML/JSON。顶层 `stages` 是非空列表，公共参数和各阶段参数直接写在同一份配置中。

对应文件：

```text
saddlellm/training/TrainingOrchestrator.py
```

CLI 读取并校验后，直接交给 `TrainingOrchestrator`，不经过格式转换器。

### 6.3 Orchestrator

Orchestrator 是执行器。

对应文件：

```text
saddle_llm/TrainingOrchestrator.py
```

它按 `stages` 执行：

- tokenizer
- pretrain
- sft
- preference
- rlhf
- mllm_sft
- eval

### 6.4 Blueprint

Blueprint 是模型结构设计。

对应文件：

```text
saddle_llm/ModelBlueprint.py
```

它描述：

- attention
- FFN
- MoE
- MLA
- MTP
- 长上下文
- 多模态入口

### 6.5 Saddle Model

Saddle 模型是框架内自研的 decoder-only LLM。

对应文件：

```text
saddle_llm/SaddleModeling.py
```

它是模型结构实验的核心载体。

### 6.6 Manifest

Manifest 是数据清单。

对应文件：

```text
saddle_llm/DatasetManifest.py
```

它可以吸收 LLaMA-Factory 风格 `dataset_info.json`。

### 6.7 PostTrainingDataAdapter

传统 LLM 后训练数据统一入口。

对应文件：

```text
saddle_llm/PostTrainingData.py
```

它统一处理：

- SFT
- DPO
- ORPO
- KTO
- RL-style records

## 7. 快速开始

### 7.1 创建一个领域训练工厂

```python
from saddle_llm import LLMTrainingFactory, FactoryConfig

factory = LLMTrainingFactory(FactoryConfig(
    root_dir="./llm_factory_medical",
    domain="medical",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    global_batch_size=64,
    num_gpus=1,
    gpu_memory_gb=24.0,
))

factory.create_workspace()
```

输出目录大致如下：

```text
llm_factory_medical/
  data/
    raw/
    processed/
  eval/
  tokenizer/
  experiments/
  models/
  reports/
  configs/
  FACTORY_BLUEPRINT.md
  factory_blueprint.json
  factory_config.json
```

### 7.2 保存总规划

```python
factory.save_plan()
```

生成：

```text
configs/factory_plan.json
```

### 7.3 生成 SFT 后训练计划

```python
plan = factory.create_post_training_plan(
    data_path="./data/sft.jsonl",
    stage="sft",
    method="qlora",
    save=True,
)
```

生成内容包括：

```text
experiments/post_training_sft/
  sft_normalized.jsonl
  sft_normalized.jsonl.report.json
  config.json
  post_training_plan.json
```

### 7.4 生成 DPO 训练计划

```python
plan = factory.create_post_training_plan(
    data_path="./data/preference.jsonl",
    stage="dpo",
    save=True,
)
```

因为 `stage="dpo"` 会自动识别为：

```text
stage = preference
method = dpo
```

### 7.5 生成模型结构实验

```python
bundle = factory.create_model_experiments(
    max_steps=100,
    seq_length=1024,
    include_moe=True,
    include_mla=True,
    include_mtp=True,
    include_long_context=False,
)
```

输出：

```text
experiments/model_architectures/
  experiment_config.json
  blueprint_comparison.json
  experiment_summary.json
  manifest.json
  runs/
    000-xxx/
      blueprint.json
      config.json
      backend_plan.json
```

### 7.6 生成 scratch pretraining 实验

```python
bundle = factory.create_pretrain_experiments(
    corpus_sources=[
        {"type": "local", "path": "./data/pretrain/*.jsonl"}
    ],
    dry_run=True,
)
```

### 7.7 执行训练配置

```python
from saddle_llm import TrainingOrchestrator

orchestrator = TrainingOrchestrator.from_yaml("./configs/train.yaml")
orchestrator.run()
```

或者：

```python
orchestrator = TrainingOrchestrator.from_dict(config)
orchestrator.run()
```

## 8. 完整工作流

### 8.1 数据准备

推荐准备三类数据：

```text
pretrain      领域语料，纯文本或文档
sft           instruction / output / messages
preference    prompt / chosen / rejected
```

建议目录：

```text
data/
  pretrain/
  sft/
  preference/
  eval/
```

### 8.2 数据 manifest

如果你有 LLaMA-Factory `dataset_info.json`：

```python
manifest_payload = factory.import_dataset_manifest(
    manifest_path="./dataset_info.json",
    dataset_dir="./data",
    role="sft",
)
```

内部会转成 SaddleLLM 的 `DatasetManifest`。

### 8.3 Tokenizer

如果你从零训练小模型，建议训练领域 tokenizer：

```python
from saddle_llm import TokenizerTrainer

trainer = TokenizerTrainer(vocab_size=32000)
trainer.fit(["./data/pretrain"])
trainer.save("./tokenizer")
```

如果是继续预训练或后训练，可以直接使用 base model tokenizer。

### 8.4 模型结构实验

先用模型结构实验比较：

- dense GQA
- MLA latent
- MoE top-2
- Dense + MTP

```python
bundle = factory.create_model_experiments(
    include_moe=True,
    include_mla=True,
    include_mtp=True,
    max_steps=100,
)
```

重点看：

```text
experiment_summary.json
```

其中包含：

- recommended smoke order
- recommended research order
- lowest cost order
- training peta FLOPs
- KV cache GB
- feasibility

### 8.5 后训练

SFT：

```python
factory.create_post_training_plan(
    data_path="./data/sft.jsonl",
    stage="sft",
    method="qlora",
)
```

DPO：

```python
factory.create_post_training_plan(
    data_path="./data/preference.jsonl",
    stage="dpo",
)
```

KTO：

```python
factory.create_post_training_plan(
    data_path="./data/kto.jsonl",
    stage="kto",
)
```

### 8.6 评估和报告

训练完成后：

```python
report = factory.report(output_dir="./outputs/run1")
```

或：

```python
from saddle_llm import PretrainReport

result = PretrainReport.generate("./outputs/run1")
```

## 9. 数据格式

### 9.1 SFT Alpaca 格式

```json
{"instruction": "解释什么是风险控制", "input": "", "output": "风险控制是识别、评估和缓释风险的过程。"}
```

### 9.2 SFT messages 格式

```json
{
  "messages": [
    {"role": "user", "content": "什么是半导体良率？"},
    {"role": "assistant", "content": "半导体良率是合格芯片数量占总芯片数量的比例。"}
  ]
}
```

### 9.3 Prompt/Completion 格式

```json
{"prompt": "解释 DNA 甲基化", "completion": "DNA 甲基化是一种表观遗传修饰。"}
```

### 9.4 Preference DPO 格式

```json
{
  "prompt": "如何处理模型幻觉？",
  "chosen": "应通过检索增强、拒答策略和事实核查降低幻觉。",
  "rejected": "模型不会产生幻觉。"
}
```

### 9.5 KTO 格式

```json
{
  "prompt": "解释 VaR",
  "completion": "VaR 是一定置信水平下的潜在最大损失估计。",
  "label": true
}
```

### 9.6 后训练数据规范化

```python
from saddle_llm import normalize_post_training_file

report = normalize_post_training_file(
    input_path="./data/raw_sft.jsonl",
    output_path="./data/sft_normalized.jsonl",
    task="sft",
)
```

DPO：

```python
report = normalize_post_training_file(
    input_path="./data/raw_pref.jsonl",
    output_path="./data/preference_normalized.jsonl",
    task="dpo",
)
```

## 10. 模型结构设计

### 10.1 Dense GQA

```python
from saddle_llm import ModelBlueprintLab

bp = ModelBlueprintLab.dense_gqa(
    name="risk-dense-gqa",
    hidden_size=512,
    layers=8,
    heads=8,
    kv_heads=2,
    seq_length=2048,
)

model = bp.build_model()
print(bp.analyze())
```

### 10.2 DeepSeek-style MoE

```python
bp = ModelBlueprintLab.deepseek_style_moe(
    name="bio-moe",
    hidden_size=1024,
    layers=16,
    heads=16,
    kv_heads=4,
    experts=8,
    experts_per_token=2,
)

model = bp.build_model()
```

### 10.3 Long Context

```python
bp = ModelBlueprintLab.minimax_style_long_context(
    name="research-long-context",
    seq_length=32768,
)
```

### 10.4 GLM-style Reasoning

```python
bp = ModelBlueprintLab.glm_style_reasoning(
    name="risk-reasoning",
)
```

### 10.5 自定义 Blueprint

```python
from saddle_llm import ModelBlueprint, AttentionBlueprint, FFNBlueprint, ObjectiveBlueprint

bp = ModelBlueprint(
    name="custom-gqa-mtp",
    family="qwen",
    hidden_size=768,
    num_layers=12,
    attention=AttentionBlueprint(
        kind="gqa",
        backend="sdpa",
        num_heads=12,
        num_kv_heads=3,
    ),
    ffn=FFNBlueprint(
        kind="swiglu",
        intermediate_size=3072,
    ),
    objective=ObjectiveBlueprint(
        multi_token_prediction=True,
        mtp_extra_tokens=2,
    ),
)

model = bp.build_model()
```

## 11. SaddleForCausalLM

核心模型类：

```text
saddle_llm/SaddleModeling.py
```

主要能力：

- decoder-only causal LM
- RMSNorm
- RoPE
- GQA/MQA/MHA
- experimental MLA
- MoE
- MTP
- KV cache
- latent MLA cache
- `generate()`
- `save_pretrained()`
- `from_pretrained_saddle()`

简单 forward：

```python
import torch
from saddle_llm import ModelBlueprintLab

bp = ModelBlueprintLab.dense_gqa(hidden_size=128, layers=2, heads=4, kv_heads=2, vocab_size=1000)
model = bp.build_model()

input_ids = torch.randint(0, 1000, (2, 32))
out = model(input_ids=input_ids, labels=input_ids)

print(out.loss)
print(out.logits.shape)
```

生成：

```python
tokens = model.generate(input_ids, max_new_tokens=32)
```

## 12. 后训练执行

### 12.1 SFT

```python
from saddlellm.training.PeftSFTTrainer import SFTTrainConfig, train_sft

train_sft(SFTTrainConfig(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    dataset_path="./data/sft_normalized.jsonl",
    output_path="./outputs/sft",
    use_lora=True,
    use_qlora=True,
    lora_r=16,
    lora_alpha=32,
    learning_rate=2e-4,
    batch_size=1,
    num_epochs=3,
))
```

### 12.2 DPO

```python
from saddle_llm import PreferenceTrainConfig, train_preference

train_preference(PreferenceTrainConfig(
    model_path="./outputs/sft",
    dataset_path="./data/preference_normalized.jsonl",
    output_path="./outputs/dpo",
    method="dpo",
    beta=0.1,
))
```

### 12.3 ORPO

```python
train_preference(PreferenceTrainConfig(
    model_path="./outputs/sft",
    dataset_path="./data/preference_normalized.jsonl",
    output_path="./outputs/orpo",
    method="orpo",
))
```

### 12.4 KTO

```python
train_preference(PreferenceTrainConfig(
    model_path="./outputs/sft",
    dataset_path="./data/kto_normalized.jsonl",
    output_path="./outputs/kto",
    method="kto",
))
```

## 13. 后端规划

### 13.1 生成 backend plan

```python
plan = factory.backend_plan(
    model_params=7_000_000_000,
    num_gpus=8,
    gpu_memory_gb=80,
    seq_length=4096,
    stage="pretrain",
)
```

返回内容包括：

- backend
- DP/TP/PP/CP/EP
- micro batch
- gradient accumulation
- memory estimate
- Megatron-style args
- Colossal plugin spec
- Orchestrator distributed config

### 13.2 生成 launch plan

```python
from saddle_llm import BackendAdapterRegistry

plan = BackendAdapterRegistry.create_launch_plan(
    config_path="./configs/config.json",
    backend="deepspeed_zero2",
    num_gpus=8,
)
```

保存脚本：

```python
BackendAdapterRegistry.save_launch_script(plan, "./run_train.ps1")
```

## 14. 评估与报告

### 14.1 Benchmark

```python
from saddle_llm import BenchmarkRunner

runner = BenchmarkRunner()
```

### 14.2 Pretrain Report

```python
from saddle_llm import PretrainReport

result = PretrainReport.generate("./outputs/pretrain_run")
print(result.to_dict())
```

### 14.3 Model Metrics

`ModelMetricsCallback` 会在 Saddle 模型训练时记录：

- architecture
- attention backend
- MLA cache mode
- MoE router metrics
- KV cache profile
- MTP loss

## 15. 监控和诊断

### 15.1 环境检查

```python
from saddle_llm import check_environment

result = check_environment()
print(result)
```

### 15.2 训练报告卡

```python
from saddle_llm import TrainingReportCard
```

### 15.3 MFU 监控

```python
from saddle_llm import TrainingMonitor
```

## 16. 多模态边界

多模态模块包括：

- `MultimodalData.py`
- `MultimodalModeling.py`
- `MultimodalProjector.py`
- `VisionBackbones.py`

当前能做：

- 标准化图文数据
- 构建 LLaVA-style wrapper
- tiny patch encoder smoke test
- projector
- 多模态训练计划

当前不建议投入太深：

- 没有完整 VLM Trainer
- 没有成熟 image collator
- 没有完整 VQA/OCR benchmark pipeline
- 当前项目主线仍是传统文本 LLM

## 17. 模块参考

### 17.1 工厂和编排

| 文件 | 作用 |
|---|---|
| `TrainingFactory.py` | 最高层训练工厂入口 |
| `TrainingOrchestrator.py` | 读取统一配置并按 stage 执行训练 |
| `cli.py` | 命令行入口 |
| `easy.py` | 简易 API |
| `DomainBuilder.py` | 领域模型训练方案 |
| `TrainingStrategyAdvisor.py` | 训练路线可行性顾问 |

### 17.2 模型结构

| 文件 | 作用 |
|---|---|
| `SaddleModeling.py` | 自研 decoder-only LLM |
| `ModelBlueprint.py` | 模型结构蓝图 |
| `ModelRegistry.py` | 预置模型规格 |
| `ModelExperimentPlanner.py` | 模型结构实验生成 |
| `ModelMetrics.py` | 模型结构指标 |
| `OperatorBackends.py` | attention backend registry |
| `Architecture.py` | 架构支持矩阵 |

### 17.3 数据

| 文件 | 作用 |
|---|---|
| `DataPipeline.py` | 预训练数据处理 |
| `DatasetManifest.py` | 数据 manifest |
| `DataStrategy.py` | 数据配比和污染检测 |
| `DataCatalog.py` | 数据集目录 |
| `DataBalance.py` | 数据平衡 |
| `TextProcess.py` | 文本清洗 |
| `PostTrainingData.py` | 后训练数据统一 |

### 17.4 预训练

| 文件 | 作用 |
|---|---|
| `TokenizerTrainer.py` | tokenizer 训练和评估 |
| `DensePretrainer.py` | dense 预训练器 |
| `PretrainExperimentRunner.py` | 预训练实验规划 |
| `PretrainEvalSuite.py` | 预训练评估 |
| `PretrainReport.py` | 训练报告 |
| `PretrainStability.py` | 稳定性监控 |
| `ScalingLawAnalyzer.py` | 缩放法则 |
| `pretrain.py` | 早期预训练脚本 |

### 17.5 后训练

| 文件 | 作用 |
|---|---|
| `PeftSFTTrainer.py` | SFT/LoRA/QLoRA |
| `Preference.py` | DPO/ORPO/KTO |
| `GRPOTrainer.py` | GRPO |
| `RLScaling.py` | RL scaling |
| `RLHFTrainer.py` | 早期 RLHF |
| `FrontierAlign.py` | CAI/PRM/Rejection Sampling |
| `AdvancedTechniques.py` | EMA/Evol-Instruct/ModelSoup |
| `ExclusiveTechniques.py` | MTP/MoE/RLAIF 等实验技术 |

### 17.6 蒸馏、压缩和部署

| 文件 | 作用 |
|---|---|
| `UniversalDistiller.py` | 通用蒸馏 |
| `RapidDistill.py` | 快速 API 蒸馏 |
| `DistillationTrainer.py` | 蒸馏 trainer |
| `distill.py` | 蒸馏脚本 |
| `ModelQuantizer.py` | 量化 |
| `quantize.py` | 量化脚本 |
| `ModelPruner.py` | 剪枝 |
| `prune.py` | 剪枝脚本 |
| `Lightweight.py` | 轻量化工具 |
| `Deploy.py` | 部署 |
| `SafeGenerate.py` | 安全生成 |

### 17.7 评估、监控和工具

| 文件 | 作用 |
|---|---|
| `BenchmarkRunner.py` | benchmark |
| `LLModelEvalute.py` | 模型评估 |
| `ExperimentTracker.py` | 实验追踪 |
| `TrainingMonitor.py` | MFU/吞吐监控 |
| `monitordashbord.py` | dashboard |
| `TrainerUtils.py` | 环境检查/诊断/早停 |
| `CurriculumScheduler.py` | 课程学习 |

### 17.8 加载和兼容

| 文件 | 作用 |
|---|---|
| `LLMLoader.py` | 模型加载 |
| `load_model.py` | LoRA/Unsloth/Transformer 加载函数 |
| `LoRATuner.py` | LoRA 调参 |
| `TRLFullFineTuner.py` | TRL 全参微调旧入口 |
| `SwiftFullFineTuner.py` | ms-swift 风格微调 |
| `UnslothFineTuner.py` | Unsloth 微调 |
| `UnslothSFTTrainer.py` | Unsloth SFT |

## 18. 当前限制

### 18.1 传统 LLM

当前传统 LLM 主线已经能做规划、数据规范化、结构实验、部分训练执行，但仍需注意：

- 大规模预训练执行仍依赖外部训练栈
- Megatron/Colossal 目前更多是规划/适配方向
- 自研 `SaddleForCausalLM` 适合研究和小规模实验
- 训练 7B+ 需要严格依赖 DeepSpeed/FSDP/Megatron 类后端

### 18.2 多模态

多模态只是入口，不是完整训练系统。

### 18.3 从头训练

低成本从头训练 MiniMax、GLM、DeepSeek 级别模型不现实。

现实路线：

```text
小模型 scratch 实验
→ 继续预训练强 base model
→ SFT
→ Preference
→ RL/GRPO on verifiable tasks
→ 蒸馏/压缩/部署
```

## 19. 推荐下一步开发方向

### 19.1 文本 LLM 主线

优先级最高：

1. 完整打通 `create_post_training_plan()` 到实际 SFT/DPO 执行。
2. 增强 eval suite，尤其领域评估。
3. 增强 `SaddleForCausalLM` checkpoint 兼容和 HF 转换。
4. 增强 MoE router 训练稳定性。
5. 增强 MLA 和长上下文真实 benchmark。

### 19.2 后端适配

建议：

1. FSDP adapter
2. DeepSpeed ZeRO-2/3 adapter
3. Megatron-style config export
4. TorchTitan/Nanotron 风格配置对照

### 19.3 数据和报告

建议：

1. 领域数据质量报告
2. SFT/preference 数据分布报告
3. contamination scan 更细化
4. 训练结果对比 dashboard

## 20. 一句话总结

SaddleLLM 当前最有价值的方向不是做一个新的 DeepSpeed 或 Megatron，而是做：

```text
模型结构可实验
数据流程可治理
训练配置可复现
后端执行可适配
领域模型可工厂化生产
```
