# SaddleLLM 代码学习指南

本文档用于快速理解 `saddle_llm` 代码包的模块职责、学习顺序和常见用法。

当前框架可以理解为：

```text
SaddleLLM = 传统文本 LLM 训练工厂
          + 模型结构实验
          + 预训练/后训练 recipe
          + 数据规范化
          + 后端规划
          + 轻量多模态入口
```

当前主线是传统文本 LLM，多模态模块保留为轻量入口，不作为近期重点。

## 推荐学习顺序

建议按下面顺序阅读：

```text
WORKLOG.md
→ saddle_llm/TrainingFactory.py
→ saddle_llm/TrainingRecipe.py
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

这样能先理解训练工厂，再理解模型结构，然后看数据、后训练和后端适配。

## 最重要的入口

### `WORKLOG.md`

开发日志。建议从最新版本往前看，能看到框架能力是如何一步步补起来的。

重点看：

- `2.30.0 Traditional LLM Post-Training Data Unification`
- `2.29.0 Multimodal Factory and Recipe Planning`
- `2.28.0 Multimodal / VLM Foundation`
- `2.27.0 Operator Backend Registry`
- `2.26.0 Model Economics and Experiment Ranking`
- `2.25.0 Model Experiment Factory`

### `saddle_llm/__init__.py`

包导出入口。里面能看到当前主要公开 API。

常用导入方式：

```python
from saddle_llm import (
    LLMTrainingFactory,
    FactoryConfig,
    TrainingRecipe,
    ModelBlueprintLab,
    PostTrainingDataAdapter,
)
```

## 训练工厂层

### `TrainingFactory.py`

最高层工厂 API，建议优先阅读。

主要类：

- `FactoryConfig`
- `FactoryStatus`
- `LLMTrainingFactory`

主要职责：

- 创建训练工作区
- 生成训练总规划
- 导入数据 manifest
- 生成后端并行计划
- 生成模型结构实验
- 生成预训练实验
- 生成后训练计划
- 生成多模态计划
- 汇总训练报告

常用方式：

```python
from saddle_llm import LLMTrainingFactory, FactoryConfig

factory = LLMTrainingFactory(FactoryConfig(
    root_dir="./llm_factory_medical",
    domain="medical",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    global_batch_size=64,
))

factory.create_workspace()
```

生成传统 LLM SFT 后训练计划：

```python
plan = factory.create_post_training_plan(
    data_path="./data/sft.jsonl",
    stage="sft",
    method="lora",
    save=True,
)
```

生成 DPO 偏好训练计划：

```python
plan = factory.create_post_training_plan(
    data_path="./data/preference.jsonl",
    stage="dpo",
    save=True,
)
```

生成模型结构实验：

```python
bundle = factory.create_model_experiments(
    max_steps=100,
    include_moe=True,
    include_mla=True,
    include_mtp=True,
)
```

生成预训练实验：

```python
bundle = factory.create_pretrain_experiments(
    corpus_sources=[
        {"type": "local", "path": "./data/pretrain/*.jsonl"}
    ],
    dry_run=True,
)
```

### `TrainingRecipe.py`

统一训练 recipe schema。

主要类：

- `RecipeModelConfig`
- `RecipeMethodConfig`
- `RecipeDataConfig`
- `RecipeMultimodalConfig`
- `RecipeBackendConfig`
- `RecipeTrainingConfig`
- `RecipeEvalConfig`
- `RecipeLoggingConfig`
- `TrainingRecipe`

主要职责：

- 用统一 schema 表达训练任务
- 把用户友好的 recipe 编译成 `TrainingOrchestrator` 配置
- 支持 `pretrain`、`sft`、`preference`、`rlhf`、`mllm_sft`、`eval`

示例：

```python
from saddle_llm import TrainingRecipe

recipe = TrainingRecipe.template(
    stage="sft",
    domain="medical",
    output_dir="./outputs/medical_sft",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

compiled = recipe.compile()
recipe.save("./configs/sft_recipe.yaml")
recipe.save_compiled("./configs/sft_orchestrator.json")
```

### `TrainingOrchestrator.py`

训练执行编排器。

主要职责：

- 读取 YAML/JSON/dict 配置
- 按 stages 顺序执行训练流程
- 支持 tokenizer、pretrain、sft、preference、rlhf、mllm_sft、eval
- 将训练摘要写入 output 目录

常用方式：

```python
from saddle_llm import TrainingOrchestrator

orchestrator = TrainingOrchestrator.from_yaml("./configs/train.yaml")
orchestrator.run()
```

也可以单阶段运行：

```python
orchestrator.run_stage("sft")
```

### `cli.py`

命令行入口。

典型用途：

```powershell
python -m saddle_llm.cli init --root ./llm_factory_medical --stage sft
python -m saddle_llm.cli train ./configs/train.yaml
python -m saddle_llm.cli report ./outputs/run1
```

### `DomainBuilder.py`

领域模型方案生成器。

支持方向：

- medical
- biology
- research
- risk
- semiconductor

主要用途：

- 生成领域训练策略
- 生成数据配比
- 生成 eval suite
- 生成低成本训练路线

### `TrainingStrategyAdvisor.py`

训练可行性顾问。

核心判断：

- 从头训练是否现实
- 继续预训练是否适合
- SFT/Preference/RL/Distill 该怎么组合
- 当前预算下推荐路线是什么

## 模型结构层

### `SaddleModeling.py`

传统文本 LLM 的核心实现。

主要类：

- `SaddleModelConfig`
- `SaddleCausalLMOutput`
- `SaddleRMSNorm`
- `SaddleRotaryEmbedding`
- `SaddleAttention`
- `SaddleSwiGLU`
- `SaddleMoE`
- `SaddleDecoderLayer`
- `SaddleForCausalLM`

支持能力：

- MHA / MQA / GQA
- experimental MLA
- SDPA/eager/flash/xformers/TE attention backend dispatch
- SwiGLU
- MoE
- shared expert
- aux-loss-free router
- MTP
- KV cache
- MLA latent cache
- generation
- save/load
- `inputs_embeds`

构建模型：

```python
from saddle_llm import ModelBlueprintLab

bp = ModelBlueprintLab.dense_gqa(
    hidden_size=512,
    layers=8,
    heads=8,
    kv_heads=2,
)

model = bp.build_model()
```

### `ModelBlueprint.py`

模型结构蓝图。

主要类：

- `AttentionBlueprint`
- `FFNBlueprint`
- `ObjectiveBlueprint`
- `VisionBlueprint`
- `ProjectorBlueprint`
- `ModelBlueprint`
- `ModelBlueprintLab`

主要职责：

- 用结构化配置描述模型
- 估算总参数、激活参数、KV cache、训练 FLOPs
- 生成风险报告和训练假设
- 构建真实 `SaddleForCausalLM`
- 生成 dense、MoE、MLA、MTP、long-context、VLM preset

示例：

```python
from saddle_llm import ModelBlueprintLab

bp = ModelBlueprintLab.deepseek_style_moe()

print(bp.analyze())

model = bp.build_model()
```

常见 preset：

```python
ModelBlueprintLab.dense_gqa()
ModelBlueprintLab.deepseek_style_moe()
ModelBlueprintLab.minimax_style_long_context()
ModelBlueprintLab.glm_style_reasoning()
ModelBlueprintLab.llava_style_vlm()
```

### `ModelRegistry.py`

预置模型规格注册表。

主要类：

- `ModelSpec`
- `ModelRegistry`

主要职责：

- 定义从 tiny 到 1.5B 等模型规格
- 根据 spec 创建 HF 模型或 Saddle 模型
- 提供参数量估算

示例：

```python
from saddle_llm import ModelRegistry

spec = ModelRegistry.get("qwen-tiny-160m")
model = ModelRegistry.create_saddle_model(spec)
```

### `ModelExperimentPlanner.py`

模型结构实验规划器。

主要类：

- `ModelExperimentConfig`
- `ModelExperimentRun`
- `ModelExperimentBundle`
- `ModelExperimentPlanner`

主要职责：

- 生成 Dense GQA、MLA、MoE、MTP 等结构对比实验
- 生成每个实验的 blueprint、recipe、orchestrator config、backend plan
- 计算实验成本、KV cache、FLOPs、优先级
- 输出 `experiment_summary.json`

示例：

```python
from saddle_llm import ModelExperimentPlanner, ModelExperimentConfig

planner = ModelExperimentPlanner(ModelExperimentConfig(
    output_dir="./experiments/model_arch",
    domain="medical",
    max_steps=100,
    seq_length=1024,
))

bundle = planner.run(dry_run=True)
```

### `ModelMetrics.py`

模型结构指标抽取。

主要类：

- `ModelMetricSnapshot`
- `ModelMetricsExtractor`
- `ModelMetricsCallback`

主要职责：

- 提取 attention/backend/MLA/MoE/MTP 等结构信息
- 记录 router metrics
- 记录 KV cache profile
- 训练中写入 `model_metrics.jsonl`

### `OperatorBackends.py`

attention/operator backend registry。

主要类：

- `AttentionBackendSpec`
- `AttentionBackendRegistry`

支持 backend：

- `sdpa`
- `flash`
- `eager`
- `xformers`
- `transformer_engine`

查看当前环境支持：

```python
from saddle_llm import list_attention_backends

print(list_attention_backends())
```

### `Architecture.py`

模型架构能力矩阵。

主要类：

- `ArchitectureSupport`
- `ArchitectureRegistry`

主要职责：

- 判断 Llama/Qwen/DeepSeek/GLM/MiniMax/Mamba/RWKV 等架构支持哪些阶段
- 支持 pretrain、continue pretrain、SFT、preference、GRPO 等能力判断

## 预训练层

### `TokenizerTrainer.py`

分词器训练器。

主要类：

- `TokenizerConfig`
- `TokenizerEvalResult`
- `TokenizerTrainer`

功能：

- 从零训练 BPE/SentencePiece tokenizer
- 评估 compression ratio、UNK rate、roundtrip、领域词切分

### `DensePretrainer.py`

dense 模型预训练器。

主要类：

- `DensePretrainConfig`
- `DensePretrainer`

功能：

- dense causal LM 预训练
- 稳定性控制
- 初始化策略
- flash attention 检测
- checkpoint 管理

初始化函数：

```python
init_weights_llama_style()
init_weights_deepseek_style()
init_weights_small_embed()
```

### `PretrainExperimentRunner.py`

低成本 scratch pretraining 实验生成器。

主要类：

- `PretrainExperimentConfig`
- `ExperimentRunManifest`
- `ExperimentBundle`
- `PretrainExperimentRunner`

功能：

- 生成多个小模型/不同 token budget 的预训练实验
- 输出 manifest 和 orchestrator config
- 支持 dry-run

### `PretrainEvalSuite.py`

预训练评估任务套件。

主要类：

- `EvalTaskSpec`
- `EvalSuiteResult`
- `PretrainEvalSuite`

功能：

- 管理 perplexity、domain QA、基础任务等 eval spec
- 保存评估配置

### `PretrainReport.py`

训练报告聚合器。

主要类：

- `ReportFinding`
- `PretrainReportResult`
- `PretrainReport`

功能：

- 汇总 tokenizer eval、stability、model metrics、eval result
- 输出风险发现和建议

### `PretrainStability.py`

预训练稳定性工具。

主要类：

- `StabilityConfig`
- `StabilityEvent`
- `PretrainStabilityCallback`
- `CheckpointInspector`

功能：

- loss spike 检测
- loss stagnation 检测
- checkpoint 检查

### `ScalingLawAnalyzer.py`

缩放法则分析器。

主要类：

- `ScalingRun`
- `PretrainRunPlan`
- `ScalingLawAnalyzer`

功能：

- Chinchilla-style 参数/token 估算
- 训练预算规划
- 推算不同模型大小所需 token

### `pretrain.py`

早期预训练脚本。

功能：

- 加载预训练模型
- 从零创建模型
- 加载数据
- 执行基础训练

现在更推荐使用 `TrainingFactory`、`TrainingRecipe`、`TrainingOrchestrator`。

## 数据层

### `DataPipeline.py`

预训练数据处理管线。

主要类：

- `PipelineConfig`
- `DataPipeline`
- `DataMixConfig`
- `DataMixer`

功能：

- 读取 local/huggingface/wikitext/wikipedia/c4
- 文本过滤
- 去重
- packing
- train/val split
- 数据混合

### `DatasetManifest.py`

数据 manifest 工具。

主要类：

- `DatasetManifestEntry`
- `DatasetManifest`

功能：

- 吸收 LLaMA-Factory `dataset_info.json`
- 标准化数据源
- 支持 Alpaca/ShareGPT/preference/pretrain
- 输出 SaddleLLM sources

示例：

```python
from saddle_llm import DatasetManifest

manifest = DatasetManifest.from_llamafactory_json("./dataset_info.json")
manifest.save("./configs/dataset_manifest.json")
```

### `DataStrategy.py`

数据策略工具。

主要类：

- `DataBucketPlan`
- `DataMixPlan`
- `DataMixPlanner`
- `ContaminationDetector`
- `ContaminationReport`

功能：

- 领域数据 bucket 规划
- 数据混合权重
- 训练/评估污染检测

### `DataCatalog.py`

公开数据集目录。

主要类：

- `DatasetEntry`
- `DataCatalog`
- `DataQualityClassifier`

功能：

- 记录常见数据集信息
- 给出推荐用途
- 做简单数据质量分类

### `DataBalance.py`

数据平衡工具。

主要类：

- `DatasetConfig`
- `BalancedPretrainer`

功能：

- 多数据源混合
- 控制不同数据源采样比例

### `TextProcess.py`

文本处理工具集。

主要类：

- `TextCleaner`
- `TextProcessor`
- `DataSourceHandler`

功能：

- 文本清洗
- 格式转换
- 数据源加载
- 一些旧版处理逻辑

### `PostTrainingData.py`

传统 LLM 后训练数据统一入口。

主要类：

- `NormalizationReport`
- `PostTrainingDataAdapter`

支持格式：

- `text`
- `messages`
- `conversations`
- Alpaca: `instruction/input/output`
- prompt/completion
- preference: `prompt/chosen/rejected`
- KTO: `prompt/completion/label`

示例：

```python
from saddle_llm import PostTrainingDataAdapter

record = {
    "instruction": "解释什么是风险控制",
    "output": "风险控制是识别、评估和缓释风险的过程。"
}

item = PostTrainingDataAdapter.normalize_record(record, task="sft")
```

离线标准化：

```python
from saddle_llm import normalize_post_training_file

report = normalize_post_training_file(
    input_path="./data/raw_sft.jsonl",
    output_path="./data/sft_normalized.jsonl",
    task="sft",
)
```

## 后训练层

### `PeftSFTTrainer.py`

稳定 SFT/LoRA/QLoRA 入口。

主要类：

- `SFTTrainConfig`

主要函数：

- `train_sft()`
- `train_model()`
- `generate_response()`

功能：

- 加载 HF 模型
- LoRA/QLoRA 配置
- TRL SFTTrainer 兼容
- SFT 数据格式化
- 现在数据格式统一走 `PostTrainingDataAdapter`

示例：

```python
from saddle_llm.PeftSFTTrainer import SFTTrainConfig, train_sft

train_sft(SFTTrainConfig(
    model_path="Qwen/Qwen2.5-7B-Instruct",
    dataset_path="./data/sft_normalized.jsonl",
    output_path="./outputs/sft",
    use_lora=True,
    use_qlora=True,
))
```

### `Preference.py`

DPO/ORPO/KTO 偏好训练入口。

主要类：

- `PreferenceTrainConfig`
- `PreferenceTrainer`

主要函数：

- `train_preference()`

功能：

- 支持 DPO
- 支持 ORPO
- 支持 KTO
- 自动选择 TRL trainer
- 数据格式统一走 `PostTrainingDataAdapter`

示例：

```python
from saddle_llm import PreferenceTrainConfig, train_preference

train_preference(PreferenceTrainConfig(
    model_path="./outputs/sft",
    dataset_path="./data/preference.jsonl",
    output_path="./outputs/dpo",
    method="dpo",
))
```

### `GRPOTrainer.py`

GRPO 训练器。

主要类：

- `GRPOConfig`
- `GRPOTrainer`

功能：

- Group Relative Policy Optimization
- 面向数学、代码、规则验证、长推理任务

### `RLScaling.py`

RL scaling 工具。

主要类：

- `RolloutSample`
- `RLScalingConfig`
- `RewardManager`
- `RolloutBuffer`
- `RLScalingTrainer`

功能：

- rollout 生成
- reward 打分
- accepted/rejected 样本过滤
- 导出 SFT 数据
- 对接 GRPO

### `RLHFTrainer.py`

早期 RLHF 封装。

功能：

- PPO/DPO 风格训练入口
- 目前更建议使用 `Preference.py` 和 `RLScaling.py`

### `FrontierAlign.py`

前沿对齐方法。

主要类：

- `ConstitutionalAI`
- `ProcessRewardModel`
- `RejectionSampling`
- `FrontierAlignmentPipeline`

功能：

- Constitutional AI
- Process Reward Model
- rejection sampling
- 对齐 pipeline

### `AdvancedTechniques.py`

高级效果提升方法。

主要类：

- `EMA`
- `EMACallback`
- `SelfConsistency`
- `EvolInstruct`
- `ModelSoup`
- `InstructionBacktranslation`

功能：

- EMA
- 自一致性
- 指令进化
- 模型汤
- instruction backtranslation

### `ExclusiveTechniques.py`

实验性模型/对齐技术。

主要类：

- `MultiTokenPredictionHead`
- `AuxLossFreeRouter`
- `AuxLossFreeMoELayer`
- `IterativeAlignment`
- `TestTimeCompute`
- `RLAIF`
- `SLURPMixer`

功能：

- MTP
- aux-loss-free MoE
- test-time compute
- RLAIF
- DeepSeek/OpenAI/Anthropic 风格技术探索

## 蒸馏/压缩/部署

### `UniversalDistiller.py`

通用蒸馏系统。

主要类：

- `TeacherInterface`
- `DistillConfig`
- `DataDistiller`
- `MultiTeacherDistiller`
- `CapabilityDistiller`
- `AutoDistiller`

功能：

- 多 teacher 蒸馏
- 能力蒸馏
- 自动生成蒸馏数据

### `RapidDistill.py`

快速蒸馏系统。

主要类：

- `TeacherProfile`
- `DownloadableTeacher`
- `ModelDownloader`
- `LocalTeacherPool`
- `SmartRouter`
- `ParallelAPICaller`
- `RapidDistill`

功能：

- 调 API 蒸馏
- 本地 teacher pool
- 多 teacher 并行生成

### `DistillationTrainer.py` / `distill.py`

传统蒸馏训练入口。

功能：

- teacher/student 蒸馏
- 旧式命令行蒸馏脚本

### `ModelQuantizer.py` / `quantize.py`

量化工具。

功能：

- 模型量化
- 校准数据准备
- 量化后测试

### `ModelPruner.py` / `prune.py`

剪枝工具。

功能：

- 模型剪枝
- 剪枝后评估
- 剪枝训练

### `Lightweight.py`

轻量化工具箱。

功能：

- 统一量化、剪枝、轻量化相关能力
- 面向部署压缩

### `Deploy.py`

部署工具。

功能：

- 导出模型
- 启动 API 服务
- 生成部署配置

### `SafeGenerate.py`

安全生成和反重复。

主要类：

- `SafeGenerate`
- `AntiRepeatGenerator`
- `OutputSafetyFilter`

功能：

- 过滤 toxic 输出
- 检测重复生成
- 安全生成 wrapper

## 后端/分布式/执行计划

### `FactoryBackendPlanner.py`

训练后端和并行规划。

主要类：

- `TrainingBackendSpec`
- `ParallelismPlan`
- `FactoryBackendPlanner`

支持 backend：

- single
- torch_ddp
- torch_fsdp
- deepspeed_zero2
- deepspeed_zero3
- hybrid_parallel
- moe_hybrid_parallel

功能：

- 估算显存
- 推荐 DP/TP/PP/CP/EP/SP
- 输出 Megatron-style args
- 输出 Colossal plugin spec
- 输出 TrainingOrchestrator distributed config

### `BackendAdapters.py`

后端启动计划生成。

主要类：

- `LaunchPlan`
- `BackendAdapterRegistry`

功能：

- 生成 single/accelerate/deepspeed/torchrun 命令
- 保存 launch script
- 根据 config 推断 backend

### `DistributedConfig.py`

分布式配置。

主要类：

- `DistributedConfig`

功能：

- 统一 DDP/FSDP/DeepSpeed ZeRO 配置
- 输出 deepspeed config

## 评估/监控/实验管理

### `BenchmarkRunner.py`

benchmark 运行器。

主要类：

- `EvalResult`
- `BenchmarkRunner`

功能：

- 跨模型评测
- 多任务对比

### `LLModelEvalute.py`

通用评估工具。

主要类：

- `Evaluator`

功能：

- 模型评估
- 命令行评测入口

### `ExperimentTracker.py`

实验追踪。

主要类：

- `TrackingConfig`
- `ExperimentTracker`

功能：

- WandB
- TensorBoard
- 本地日志

### `TrainingMonitor.py`

训练效率监控。

主要类：

- `MFUConfig`
- `TrainingMonitor`

功能：

- MFU
- tokens/s
- step time
- 显存/吞吐监控

### `monitordashbord.py`

监控 dashboard。

主要类：

- `MetricData`
- `SystemStatus`
- `AlertData`
- `MonitorServer`

功能：

- WebSocket metric
- dashboard endpoint
- alerts

### `TrainerUtils.py`

训练工具集。

主要类：

- `EnvCheckResult`
- `TrainingDiagnostics`
- `TrainingReportCard`
- `EarlyStopping`

主要函数：

- `check_environment()`
- `auto_train_best_small_model()`

功能：

- 环境检查
- 训练诊断
- 报告卡
- early stopping

### `CurriculumScheduler.py`

课程学习调度器。

主要类：

- `StageConfig`
- `CurriculumScheduler`

功能：

- 分阶段训练
- 按 step 切换数据/难度/阶段

## 加载/微调兼容层

### `LLMLoader.py`

模型加载器。

主要类：

- `LLMConfig`
- `LLMLoader`

功能：

- 加载 HF/本地模型
- 处理 device
- 处理 quantization
- 处理 flash attention

### `load_model.py`

加载函数集合。

主要函数：

- `unsloth_load_lora_model`
- `transfomer_load_lora`
- `src_mode_transformer_load`

### `LoRATuner.py`

LoRA 调参器。

主要类：

- `LoRATuner`

功能：

- LoRA 微调封装

### `TRLFullFineTuner.py`

TRL full fine-tuning 旧入口。

主要函数：

- `train_model`
- `generate_response`

### `SwiftFullFineTuner.py`

ms-swift 风格 full fine-tune 封装。

主要类：

- `SwiftFullFineTuner`

### `UnslothFineTuner.py`

Unsloth 相关微调入口。

功能：

- Unsloth 模型训练
- 生成训练数据
- checkpoint eval

### `UnslothSFTTrainer.py`

Unsloth SFT 入口。

主要函数：

- `UnslotSFTTrainer`

## 多模态入口，当前非主线

这些模块已经有基础能力，但近期主线是传统文本 LLM。

### `MultimodalData.py`

图文数据标准化。

主要类：

- `MultimodalAsset`
- `MultimodalSample`
- `MultimodalNormalizationReport`
- `MultimodalDataAdapter`

### `MultimodalModeling.py`

LLaVA-style wrapper。

主要类：

- `MultimodalConfig`
- `MultimodalForCausalLM`

### `MultimodalProjector.py`

视觉特征到 LLM hidden size 的 projector。

支持：

- linear
- mlp
- gated_mlp
- resampler

### `VisionBackbones.py`

vision backbone registry。

支持：

- tiny_patch
- clip
- siglip
- dinov2

### `SaddleMultimodalModeling.py`

兼容 shim。

新代码不建议直接用这个文件，使用 `MultimodalModeling.py`。

## UI/部署/其他工具

### `ChatUI.py`

聊天 UI。

主要类：

- `ChatUI`

功能：

- 启动 Web Chat UI
- 方便人工体验模型

### `DataCatalog.py`

数据目录和质量分类，前面数据层也提到过。

### `easy.py`

简易 API。

主要函数：

- `create`
- `train`
- `train_tokenizer`
- `distill`
- `chat`
- `improve`
- `evaluate`
- `list_models`
- `check`
- `auto`

适合快速试用，不建议作为复杂项目主入口。

## 常见任务怎么做

### 1. 创建一个领域训练工厂

```python
from saddle_llm import LLMTrainingFactory, FactoryConfig

factory = LLMTrainingFactory(FactoryConfig(
    root_dir="./llm_factory_risk",
    domain="risk",
    base_model="Qwen/Qwen2.5-7B-Instruct",
))

factory.create_workspace()
factory.save_plan()
```

### 2. 导入 LLaMA-Factory 数据清单

```python
payload = factory.import_dataset_manifest(
    manifest_path="./dataset_info.json",
    dataset_dir="./data",
    role="sft",
)
```

### 3. 生成后端并行计划

```python
plan = factory.backend_plan(
    model_params=7_000_000_000,
    num_gpus=8,
    gpu_memory_gb=80,
    seq_length=4096,
    stage="sft",
)
```

### 4. 生成 SFT 训练计划

```python
plan = factory.create_post_training_plan(
    data_path="./data/sft.jsonl",
    stage="sft",
    method="qlora",
    save=True,
)
```

### 5. 生成 DPO 训练计划

```python
plan = factory.create_post_training_plan(
    data_path="./data/preference.jsonl",
    stage="dpo",
    save=True,
)
```

### 6. 生成模型结构实验

```python
bundle = factory.create_model_experiments(
    include_moe=True,
    include_mla=True,
    include_mtp=True,
    include_long_context=False,
    max_steps=100,
)
```

### 7. 构建一个自定义模型

```python
from saddle_llm import ModelBlueprint, AttentionBlueprint, FFNBlueprint

bp = ModelBlueprint(
    name="my-dense-gqa",
    family="qwen",
    hidden_size=512,
    num_layers=8,
    attention=AttentionBlueprint(kind="gqa", num_heads=8, num_kv_heads=2),
    ffn=FFNBlueprint(kind="swiglu", intermediate_size=2048),
)

model = bp.build_model()
```

### 8. 规范化 SFT 数据

```python
from saddle_llm import normalize_post_training_file

report = normalize_post_training_file(
    input_path="./data/raw_sft.jsonl",
    output_path="./data/sft_normalized.jsonl",
    task="sft",
)
```

### 9. 规范化 preference 数据

```python
from saddle_llm import normalize_post_training_file

report = normalize_post_training_file(
    input_path="./data/raw_pref.jsonl",
    output_path="./data/preference_normalized.jsonl",
    task="dpo",
)
```

### 10. 执行 orchestrator config

```python
from saddle_llm import TrainingOrchestrator

orchestrator = TrainingOrchestrator.from_yaml("./configs/train.yaml")
orchestrator.run()
```

## 当前框架定位

当前最重要的方向是：

```text
传统文本 LLM 训练工厂
```

重点不是重复造 DeepSpeed/Megatron，而是：

- 定义模型结构
- 生成结构实验
- 规划训练成本
- 管理领域数据
- 统一后训练数据
- 生成可执行 recipe
- 对接已有训练后端

多模态已经留了入口，但不作为当前重点。

