# SaddleLLM Worklog

Date: 2026-07-01 — 2026-08-14
Repo: SaddleLLM repository root
Current package version: `2.32`

## Goal

Upgrade `saddle_llm` from a fine-tuning-oriented toolkit into a low-cost domain LLM training framework that can support:

- small-scale scratch pretraining experiments
- domain continued pretraining
- SFT, preference training, RL scaling, and distillation
- data strategy, contamination checks, evaluation suites, and run reports

## Major Changes

### Architecture and model support

- Added an architecture support matrix in `saddle_llm/Architecture.py`.
- Added support-level categories:
  - `native`
  - `hf_auto`
  - `custom_required`
  - `unsupported`
- Registered model families including Llama, GPT-2, GPT-NeoX, Mixtral, Qwen, GLM, DeepSeek, MiniMax, Mamba, RWKV, and multimodal families.
- Added MiniMax detection and conservative support notes.
- Promoted Qwen to native scratch support through `Qwen2Config`.

### Qwen scratch pretraining

- Added Qwen-style scratch model specs in `saddle_llm/ModelRegistry.py`:
  - `qwen-tiny-160m`
  - `qwen-300m`
  - `qwen-700m`
  - `qwen-1.5b`
- Added `ModelRegistry.create_model()` branch for `Qwen2Config`.
- Verified Qwen scratch instantiation with a reduced vocab override.

### Data pipeline

- Improved `DataPipeline` robustness:
  - config validation
  - local glob collection
  - weighted source interleaving
  - exact/minhash/simhash dedup modes
  - quality filtering fallback paths
  - safer packing labels and attention masks
- Added `source_weights` support through `TrainingOrchestrator`.

### Tokenizer training and evaluation

- Added `TokenizerEvalResult`.
- Added `TokenizerTrainer.evaluate()`.
- Metrics include:
  - compression ratio
  - tokens per char
  - UNK rate
  - roundtrip exact rate
  - domain-term fragmentation
- Fixed BPE post-processor initialization order so special token IDs are set after training.

### Domain model builder

- Added `DomainBuilder` workflows for:
  - medical
  - biology
  - research
  - risk
  - semiconductor
- Added domain recipes with recommended stages, data buckets, risk controls, and notes.
- Added helpers:
  - `training_strategy()`
  - `low_cost_recipe()`
  - `data_mix_plan()`
  - `contamination_report()`
  - `eval_suite()`
  - `pretrain_experiments()`

### SFT and preference training

- Reworked SFT config/API around `SFTTrainConfig` and `train_model`.
- Added better dataset formatting support:
  - `text`
  - `messages`
  - `instruction/output`
  - prompt/completion style records
- Added preference training module with:
  - DPO
  - ORPO
  - KTO
- Added `PreferenceConfig` and `preference` stage support to `TrainingOrchestrator`.

### RL scaling

- Added `RLScaling.py`.
- Main components:
  - `RLScalingConfig`
  - `RewardManager`
  - `RolloutBuffer`
  - `RolloutSample`
  - `RLScalingTrainer`
- Supports rollout generation, reward scoring, filtering, and SFT/GRPO bridge export.

### Distillation and strategy advisor

- Added `TrainingStrategyAdvisor`.
- It analyzes whether a target family is realistic for:
  - scratch pretraining
  - continued pretraining
  - SFT
  - preference training
  - RL/GRPO
  - distillation
- Encodes the practical conclusion:
  - low-cost scratch MiniMax/GLM/DeepSeek-scale training is not realistic
  - low-cost domain modeling should use checkpoint adaptation, distillation, and post-training
  - scratch should start with small dense/Qwen/Llama models

### Data strategy

- Added `DataStrategy.py`.
- Main components:
  - `DataBucketPlan`
  - `DataMixPlan`
  - `DataMixPlanner`
  - `ContaminationDetector`
  - `ContaminationReport`
- Supports domain/general data mix planning.
- Supports benchmark/eval contamination scanning with n-gram shingles.

### Scaling-law experiment planning

- Added `PretrainRunPlan`.
- Added `ScalingLawAnalyzer.plan_pretrain_grid()`.
- Added `PretrainRunPlan.to_orchestrator_config()`.
- Supports generating small-model scratch pretraining plans for 160M/300M/700M/1.5B-scale experiments.

### Training stability

- Added `PretrainStability.py`.
- Main components:
  - `StabilityConfig`
  - `StabilityEvent`
  - `PretrainStabilityCallback`
  - `CheckpointInspector`
- Detects:
  - NaN/Inf loss
  - loss spikes
  - loss stagnation
  - increasing loss trend
- Writes:
  - `stability_events.jsonl`
  - `stability_summary.json`
- Integrated into `TrainingOrchestrator` pretrain stage.

### Distributed templates

- Extended `DistributedConfig`.
- Added:
  - `recommend_for_model()`
  - `save_accelerate_config()`
  - `save_template_bundle()`
- Generates conservative distributed strategies and configs for DDP, FSDP, and DeepSpeed ZeRO.

### Evaluation suites

- Added `PretrainEvalSuite.py`.
- Main components:
  - `EvalTaskSpec`
  - `EvalSuiteResult`
  - `PretrainEvalSuite`
- Supports fixed pretraining eval suites for:
  - heldout general perplexity
  - domain heldout perplexity
  - domain QA/generation metrics
  - contamination references
  - run-to-run comparisons

### Pretraining report

- Added `PretrainReport.py`.
- Aggregates:
  - `summary.json`
  - `tokenizer/tokenizer_eval.json`
  - `stability/stability_summary.json`
  - `eval_suite/suite_results.json`
  - `data_mix_plan.json`
  - `contamination_report.json`
- Produces:
  - `pretrain_report.json`
  - `pretrain_report.md`
- Adds findings, score, and grade.
- Integrated into `TrainingOrchestrator._print_summary()`.

### Experiment runner

- Added `PretrainExperimentRunner.py`.
- Main components:
  - `PretrainExperimentConfig`
  - `ExperimentRunManifest`
  - `ExperimentBundle`
  - `PretrainExperimentRunner`
- Generates a complete dry-run experiment bundle:
  - `experiment_config.json`
  - `data_mix_plan.json`
  - `eval_suite.json`
  - per-run `orchestrator_config.json`
  - per-run `run_plan.json`
  - `manifest.json`
- Supports real execution through `run(dry_run=False)`.

### Training factory facade

- Added `TrainingFactory.py`.
- Main components:
  - `FactoryConfig`
  - `FactoryStatus`
  - `LLMTrainingFactory`
- Provides a project-level factory facade for:
  - workspace creation
  - factory blueprint generation
  - domain training plan generation
  - scratch pretraining experiment bundle generation
  - report aggregation
  - factory status inspection
- This is now the top-level entry point for building a repeatable LLM training factory.

## Current Capability

The framework can now support a low-cost domain model training loop:

```text
data strategy
-> contamination scan
-> tokenizer train/eval
-> scratch pretrain plan
-> TrainingOrchestrator config
-> stability monitoring
-> eval suite
-> pretrain report
-> compare runs
```

Recommended first real training target:

```text
model: qwen-tiny-160m
domain: semiconductor or medical
token_multipliers: [0.01, 0.05]
mode: dry-run first, then one short real run
```

## Validation Performed

Commands repeatedly used during development:

```powershell
python -m compileall -q saddle_llm
python -m compileall -q saddle_llm build\lib\saddle_llm
python setup.py --version
python setup.py build_py
```

Current verified version:

```text
2.24.0
```

Additional targeted checks passed:

- Qwen2Config scratch model instantiation
- Tokenizer small-corpus train/evaluate flow
- Scaling-law grid generation
- Data mix normalization
- ContaminationDetector sample scan
- TrainingOrchestrator `source_weights` parsing
- PretrainEvalSuite generation and comparison
- PretrainReport JSON/Markdown generation
- PretrainExperimentRunner dry-run bundle generation
- DomainModelBuilder experiment bundle generation
- LLMTrainingFactory workspace/plan/dry-run/status generation
- DatasetManifest import from LLaMA-Factory-style `dataset_info.json`
- FactoryBackendPlanner backend/parallelism recommendation and export helpers
- TrainingRecipe compile path from user-facing recipe YAML/JSON to TrainingOrchestrator config
- BackendAdapter launch-plan/script generation for single, Accelerate, DeepSpeed/FSDP/DDP, and hybrid handoff
- ModelBlueprint architecture lab for dense GQA, DeepSeek-style MoE/MLA/MTP, MiniMax-style long context, and GLM-style reasoning blueprints
- Fixed ModelSpec parameter estimation so FFN parameters are counted per layer
- Modular SaddleModeling layer with reusable model blocks and trainable SaddleForCausalLM
- Saddle model backend integrated into recipes and TrainingOrchestrator scratch pretraining
- KV-cache generation and experimental MTP draft/verify decoding for SaddleForCausalLM
- Attention backend selection, MLA latent cache, and MoE router metrics
- Model metrics extractor/callback and PretrainReport integration

## Known Boundaries

- Native DeepSeek-V3/MiniMax-style architecture is not implemented.
- DeepSeek MLA, full large MoE routing stack, and MiniMax Lightning Attention remain future work.
- Low-cost scratch training should target 100M to 1.5B-class experiments first.
- Megatron/Colossal hybrid-parallel plans are exported as control-plane specs; direct runtime execution still needs a matching backend integration.
- DeepSpeed and Megatron-LM are treated as execution backends, not as the core product direction.
- Real capability gains still depend heavily on data quality, compute budget, eval set design, and post-training.
- No long real pretraining run has been executed yet in this worklog.

## 2.19.0 Model-First Direction

The strategy was clarified: SaddleLLM should not mainly clone AI infrastructure.  DeepSpeed, Megatron-LM, Accelerate, and Colossal-AI are backend/runtime layers.  SaddleLLM should use them when useful, while focusing on model definition and model experimentation.

Implemented:

- `TrainingRecipe.py`
  - User-facing recipe schema for `pretrain`, `sft`, `preference`, `rlhf`, and `eval`.
  - Compiles recipes into existing `TrainingOrchestrator` configs.
  - Auto-resolves dataset manifests and backend plans.
- `BackendAdapters.py`
  - Generates launch plans and PowerShell launch scripts.
  - Normalizes `ddp/fsdp/deepspeed_zero*` strategy names to backend plans.
- `cli.py`
  - Added command surface:
    - `saddle-llm init`
    - `saddle-llm import-data`
    - `saddle-llm plan`
    - `saddle-llm train`
    - `saddle-llm report`
- `ModelBlueprint.py`
  - Added model architecture blueprints and design-time analysis.
  - Supports Dense GQA, DeepSeek-style MoE + MLA + MTP, MiniMax-style long context, and GLM-style reasoning templates.
  - Reports total params, active params, Chinchilla tokens, FLOPs/token, KV cache cost, architecture axes, risks, and hypotheses.
- `TrainingFactory.py`
  - Added `model_blueprints()` to generate domain-specific model architecture comparisons.

Validated:

```powershell
python -m compileall -q saddle_llm build\lib\saddle_llm
python setup.py --version
python setup.py build_py
python -m saddle_llm.cli init --domain semiconductor --root-dir build\tmp_cli_factory --stage sft --num-gpus 2 --gpu-memory-gb 24
python -m saddle_llm.cli plan build\tmp_cli_factory\configs\sft_recipe.yaml
python -m saddle_llm.cli train build\tmp_cli_factory\configs\sft_recipe.yaml --dry-run
```

## 2.20.0 Modular Model Layer

Implemented `SaddleModeling.py` as the first real model-structure layer:

- `SaddleModelConfig`
  - Native config for SaddleLLM models.
  - Can be created from `ModelBlueprint` or `ModelSpec`.
- Modular blocks:
  - `SaddleRMSNorm`
  - `SaddleRotaryEmbedding`
  - `SaddleAttention`
  - `SaddleSwiGLU`
  - `SaddleMoE`
  - `SaddleDecoderLayer`
  - `SaddleForCausalLM`
- Attention support:
  - MHA/GQA/MQA-style KV grouping.
  - Experimental MLA-style low-rank Q/KV projection path.
  - Explicit causal + padding mask compatibility.
- FFN support:
  - Dense SwiGLU.
  - Sparse top-k MoE.
  - Shared expert option.
  - Aux-loss-free router bias option.
  - Router aux loss option.
- Objective support:
  - Standard causal LM loss.
  - Optional multi-token prediction heads and losses.
- Integration:
  - `ModelBlueprint.build_model()`
  - `ModelBlueprint.to_saddle_config()`
  - `ModelRegistry.create_saddle_model()`
  - Top-level exports from `saddle_llm`.

Validated tiny dense and MoE models:

```powershell
python -m compileall -q saddle_llm
python setup.py --version
```

Targeted checks covered:

- Dense GQA forward/backward with MTP.
- MoE + experimental MLA forward/backward with expert load stats.
- Attention mask compatibility.
- `save_pretrained()` / `from_pretrained_saddle()`.
- `ModelRegistry.create_saddle_model()` with a tiny `ModelSpec`.

## 2.21.0 Saddle Model Training Backend

Implemented the first end-to-end bridge from model structure to training:

- `SaddleForCausalLM` now returns `SaddleCausalLMOutput`, a Transformers-compatible `ModelOutput`.
- Added Trainer compatibility methods:
  - `gradient_checkpointing_enable()`
  - `gradient_checkpointing_disable()`
  - `enable_input_require_grads()`
  - `get_input_embeddings()`
  - `set_input_embeddings()`
  - `get_output_embeddings()`
  - `set_output_embeddings()`
  - `resize_token_embeddings()`
- `save_pretrained()` accepts Trainer-style kwargs and writes Saddle-native `saddle_config.json` + `pytorch_model.bin`.
- `TrainingRecipe` supports:

```yaml
model:
  backend: saddle
```

- `TrainingOrchestrator` supports `model_backend="saddle"` for scratch pretraining and calls `ModelRegistry.create_saddle_model()`.
- Final orchestrator save writes both Trainer output and Saddle-native model artifacts.

Validated:

- `SaddleCausalLMOutput` attribute and dict-style access.
- `TrainingRecipe` compile with `model.backend=saddle`.
- Hugging Face `Trainer` 1-step CPU smoke train with `SaddleForCausalLM`.
- Native save/load after Trainer smoke training.
- `TrainingOrchestrator.from_dict()` parses `model.backend=saddle`.

## 2.22.0 Inference Path for Modular Models

Implemented generation-oriented model features:

- `SaddleAttention`
  - Added `past_key_value` and `use_cache`.
  - Rotary embeddings now support positional offsets for cached decoding.
  - Standard GQA/MQA/MHA and experimental MLA paths both support cache.
  - Causal and padding masks are merged explicitly for cached and uncached paths.
- `SaddleDecoderLayer`
  - Propagates per-layer KV cache.
- `SaddleForCausalLM`
  - Forward accepts and returns `past_key_values`.
  - Added `prepare_inputs_for_generation()`.
  - `generate()` now uses cache by default.
  - Added `draft_mtp_tokens()` for MTP-head draft proposals.
  - Added experimental `generate_mtp_speculative()` with main-head verification.

Validated:

- Dense GQA full-forward vs cached incremental logits alignment.
- Experimental MLA + MoE full-forward vs cached incremental logits alignment.
- Greedy cached `generate()`.
- MTP draft token shape.
- Experimental MTP speculative generation shape.

## 2.23.0 Model Architecture Experiment Controls

Implemented model-first controls for comparing architecture choices:

- `SaddleModelConfig`
  - Added `attention_backend`: `sdpa`, `eager`, `auto`, or `flash`.
  - Added `mla_cache_mode`: `kv` or `latent`.
- `AttentionBlueprint`
  - Added matching `backend` and `mla_cache_mode` fields so blueprints can drive these choices.
- `SaddleAttention`
  - Added explicit attention backend dispatch.
  - `sdpa/auto/flash` use PyTorch scaled dot-product attention.
  - `eager` uses explicit matmul + softmax attention for debugging and numerical comparisons.
  - Experimental MLA latent cache stores compressed latent KV instead of decompressed K/V, then reconstructs K/V for attention.
- `SaddleMoE`
  - Added router metrics:
    - `router_entropy`
    - `load_balance`
    - `tokens_per_expert`
    - `dropped_tokens`
- `SaddleForCausalLM`
  - Added `router_metrics()` aggregation across MoE layers.

Validated:

- SDPA vs eager logits alignment.
- MLA latent cache full-forward vs cached incremental logits alignment.
- MoE expert stats include entropy, load balance, tokens per expert, and dropped token count.
- `router_metrics()` aggregates layer-level MoE load balance.

## 2.24.0 Model Experiment Metrics Loop

Implemented model-structure metric extraction and reporting:

- `ModelMetrics.py`
  - Added `ModelMetricSnapshot`.
  - Added `ModelMetricsExtractor`.
  - Added `ModelMetricsCallback` for HuggingFace Trainer.
- Metrics captured:
  - architecture fields: hidden size, layers, heads, attention kind/backend, MLA cache mode, FFN kind, MoE settings, MTP settings
  - router metrics from modular MoE models
  - cache profile including bytes/token/layer and bytes/token total
  - loss-like log fields such as MTP losses
- `TrainingOrchestrator`
  - Automatically attaches `ModelMetricsCallback` for `model.backend=saddle` pretraining.
  - Writes metrics to `model_metrics/model_metrics.jsonl`.
  - Writes latest summary to `model_metrics/model_metrics_summary.json`.
- `PretrainReport`
  - Discovers `model_metrics/model_metrics_summary.json`.
  - Adds findings for:
    - low MoE load balance
    - MLA latent cache enabled
    - MTP objective enabled

Validated:

- Direct `ModelMetricsExtractor.snapshot()` on Saddle MoE + MLA + MTP model.
- Trainer 1-step smoke run with `ModelMetricsCallback`.
- `PretrainReport.generate()` loading model metrics and producing model findings.

## 2.25.0 Model Experiment Factory

Implemented controlled architecture experiment generation for model-first R&D:

- `ModelExperimentPlanner.py`
  - Added `ModelExperimentConfig`, `ModelExperimentRun`, and `ModelExperimentBundle`.
  - Generates comparable runs for Dense GQA, MLA latent cache, MoE top-2, MTP, and optional long-context variants.
  - Writes per-run `blueprint.json`, `recipe.yaml`, `orchestrator_config.json`, and `backend_plan.json`.
  - Writes top-level `experiment_config.json`, `blueprint_comparison.json`, and `manifest.json`.
- `TrainingRecipe` / `TrainingOrchestrator`
  - `RecipeModelConfig.backend="saddle"` now compiles through to orchestrator configs.
  - Added `model_blueprint` support so an experiment can instantiate the exact modular Saddle model structure, not only a registry fallback.
  - Scratch pretraining with `model.backend=saddle` now builds `SaddleForCausalLM` from the supplied blueprint.
- `TrainingFactory`
  - Added `create_model_experiments()` high-level entry point for LLM training factory workflows.
  - Factory output now places model architecture experiments under `experiments/model_architectures`.

Validated:

- Dry-run `ModelExperimentPlanner` bundle generation.
- Orchestrator reconstruction of a blueprint-backed MLA latent Saddle model.
- Factory `create_model_experiments()` entry point and exported run metadata.
- `python -m compileall -q saddle_llm`.
- `python setup.py build_py`.

## 2.26.0 Model Economics and Experiment Ranking

Implemented cost-aware model experiment planning:

- `ModelBlueprint`
  - Added `estimate_training_flops()`.
  - Added `estimate_kv_cache_gb()`.
  - Added `cost_profile()` with token budget, training FLOPs, peta-FLOPs, KV cache GB, bf16 parameter memory, and active/total ratio.
  - Included the cost profile in `analyze()` output.
- `ModelExperimentPlanner`
  - Adds per-run `experiment_cost_profile`.
  - Adds per-run `score` with feasibility, smoke priority, research priority, memory headroom, innovation points, cost penalty, and cache penalty.
  - Adds per-run backend plan metadata to the exported manifest.
  - Writes `experiment_summary.json` with:
    - recommended smoke order
    - recommended research order
    - lowest-cost order
    - backend warnings
    - next actions

Validated:

- Dry-run model experiment bundle generation with Dense GQA, MLA, MoE, and MTP runs.
- Generated `experiment_summary.json` and verified ordered recommendations.
- Direct `ModelBlueprint.cost_profile()` smoke validation.
- `python -m compileall -q saddle_llm`.

## 2.27.0 Operator Backend Registry

Implemented the first operator-backend plug-in layer for model-structure R&D:

- `OperatorBackends.py`
  - Added `AttentionBackendSpec`.
  - Added `AttentionBackendRegistry`.
  - Added `list_attention_backends()`.
  - Registered `sdpa`, `flash`, `eager`, `xformers`, and `transformer_engine` capability specs.
  - Added safe fallback behavior for optional backends.
- `SaddleModeling`
  - `SaddleAttention._attention()` now dispatches through `AttentionBackendRegistry`.
  - The model layer no longer owns backend-specific kernel branching.
- `ModelBlueprint`
  - Attention backend choices now include `auto`, `sdpa`, `flash`, `eager`, `xformers`, and `transformer_engine`.
- `ModelExperimentPlanner`
  - `experiment_summary.json` now records operator backend capability snapshots.

Validated:

- `sdpa` vs `eager` attention output alignment.
- `xformers` backend fallback on CPU.
- Blueprint-backed `SaddleForCausalLM` forward pass with `attention.backend="xformers"`.
- Experiment summary includes registered operator backend capabilities.
- `python -m compileall -q saddle_llm`.

## 2.28.0 Multimodal / VLM Foundation

Implemented the first low-cost LLaVA-style multimodal modeling layer:

- `MultimodalData.py`
  - Added `MultimodalAsset`, `MultimodalSample`, and `MultimodalDataAdapter`.
  - Normalizes `messages + images`, LLaVA/ShareGPT conversations, VQA, OCR, and document-style records.
  - Adds `normalize_multimodal_file()` for offline jsonl normalization and reports.
- `VisionBackbones.py`
  - Added `VisionBackboneConfig`.
  - Added `VisionBackboneRegistry`.
  - Added `TinyPatchVisionBackbone` for no-download smoke tests and projector warmup.
  - Added HuggingFace vision backbone wrapper hooks for CLIP/SigLIP/DINOv2 style encoders.
- `MultimodalProjector.py`
  - Added linear, MLP, gated MLP, and resampler projectors.
- `MultimodalModeling.py`
  - Added clean public API `MultimodalConfig` and `MultimodalForCausalLM`.
  - Kept `SaddleMultimodalConfig` and `SaddleMultimodalForCausalLM` as compatibility aliases only.
  - Supports image-token replacement and image-token prepending.
- `SaddleModeling`
  - `SaddleForCausalLM.forward()` now accepts `inputs_embeds`, enabling multimodal wrappers to feed projected visual tokens directly.
- `ModelBlueprint`
  - Added `VisionBlueprint` and `ProjectorBlueprint`.
  - Added `build_multimodal_model()`.
  - Added `ModelBlueprintLab.llava_style_vlm()` preset.
- `PostTrainingData.py`
  - Added a shared post-training data normalizer for SFT, preference, KTO, and RL-style records.

Validated:

- Built a tiny LLaVA-style `MultimodalForCausalLM` from blueprint.
- Ran forward pass with image patch tokens, projected visual embeddings, labels, and loss.
- Normalized a LLaVA/ShareGPT-style image conversation sample.
- Verified vision backbone registry exposes `tiny_patch`, `clip`, `siglip`, and `dinov2`.
- `python -m compileall -q saddle_llm`.

## 2.29.0 Multimodal Factory and Recipe Planning

Implemented the first factory-level VLM/MLLM planning loop:

- `MultimodalModeling.py`
  - Moved the primary implementation to the clean module name.
  - `SaddleMultimodalModeling.py` is now only a compatibility shim.
  - `save_pretrained()` writes `multimodal_config.json`; loading remains backward-compatible with `saddle_multimodal_config.json`.
- `TrainingRecipe`
  - Added `RecipeMultimodalConfig`.
  - Added `mllm_sft` and `vision_alignment` recipe support.
  - Compiled recipes now include a `multimodal` plan block.
- `TrainingOrchestrator`
  - Added `MultimodalTrainingConfig`.
  - Added `mllm_sft` / `vision_alignment` stage recognition.
  - Added `_run_multimodal_stage()` for planning and data materialization.
  - The stage writes `multimodal/multimodal_training_plan.json`.
  - When data is provided, it writes `multimodal/mllm_sft_normalized.jsonl`.
- `TrainingFactory`
  - Added `create_multimodal_plan()`.
  - Generates a LLaVA-style VLM blueprint, recipe, orchestrator config, and backend/vision capability context.

Validated:

- `vision_alignment` recipe compiles to `mllm_sft + eval`.
- Factory creates multimodal plan artifacts.
- Orchestrator `mllm_sft` stage normalizes a sample image QA jsonl and writes a plan.
- `python -m compileall -q saddle_llm`.

## 2.30.0 Traditional LLM Post-Training Data Unification

Returned focus to text-only LLM post-training:

- `PostTrainingData.py`
  - Shared normalizer is now the canonical path for SFT, DPO/ORPO, KTO, and RL-style records.
  - Supports Alpaca, ShareGPT/messages, prompt/completion, chosen/rejected, KTO labels, and simple text records.
- `PeftSFTTrainer`
  - `_format_sft_batch()` now uses `PostTrainingDataAdapter`.
  - SFT formatting rules are no longer duplicated inside the trainer.
- `Preference`
  - `_normalize_preference_batch()` now uses `PostTrainingDataAdapter`.
  - DPO/ORPO/KTO records share the same schema handling.
- `TrainingFactory`
  - Added `create_post_training_plan()`.
  - Generates normalized jsonl, recipe YAML, orchestrator config, and `post_training_plan.json` for traditional text LLM SFT/preference work.

Validated:

- SFT batch formatting through the shared adapter.
- DPO and KTO batch normalization through the shared adapter.
- Factory SFT post-training plan generation with normalized data and compiled orchestrator config.
- `python -m compileall -q saddle_llm`.

## 2.18.0 External Framework Study

Official shallow clones were placed under `external_research/`:

- `LLaMA-Factory`: studied dataset registry, training stage, fine-tuning method, and chat-template organization.
- `ColossalAI`: studied Booster/plugin abstractions covering DDP, FSDP, ZeRO, Gemini, hybrid parallel, and MoE hybrid parallel.
- `Megatron-LM`: studied explicit TP/PP/CP/EP/SP parallel-state dimensions and Megatron-style launch arguments.

Implemented SaddleLLM-native equivalents without copying external source code:

- `DatasetManifest.py`
  - Normalizes LLaMA-Factory-style `dataset_info.json`.
  - Supports local files, Hugging Face hub entries, ShareGPT/Alpaca column mappings, ranking/preference flags, roles, and weights.
  - Exports `to_saddle_sources()` and normalized `source_weights()` for `DataPipeline` and pretraining experiment configs.
- `FactoryBackendPlanner.py`
  - Registers backend specs for `single`, `torch_ddp`, `torch_fsdp`, `deepspeed_zero2`, `deepspeed_zero3`, `hybrid_parallel`, and `moe_hybrid_parallel`.
  - Recommends backend, DP/TP/PP/CP/EP/SP dimensions, micro batch, global batch, and gradient accumulation.
  - Exports SaddleLLM orchestrator distributed config, Megatron-style args, and Colossal-style plugin specs.
- `TrainingFactory.py`
  - Added `num_gpus`, `gpu_memory_gb`, and `prefer_backend` to `FactoryConfig`.
  - Added `import_dataset_manifest()` and `backend_plan()` high-level factory methods.
  - Included dataset manifest and backend parallel plan in the factory blueprint/plan.

## Recommended Next Steps

1. Prepare a small but clean domain corpus.
2. Run `PretrainExperimentRunner` in dry-run mode.
3. Inspect generated:
   - data mix plan
   - eval suite
   - orchestrator configs
   - run plans
4. Train `qwen-tiny-160m` for a short smoke run.
5. Review:
   - tokenizer eval
   - stability summary
   - eval suite output
   - pretrain report
6. Only then scale to `qwen-300m`.

## Example Entry Point

```python
from saddle_llm import PretrainExperimentRunner, PretrainExperimentConfig

runner = PretrainExperimentRunner(PretrainExperimentConfig(
    domain="semiconductor",
    output_dir="./experiments/semiconductor_scratch",
    model_names=["qwen-tiny-160m", "qwen-300m"],
    token_multipliers=[0.01, 0.05],
    corpus_sources=[
        {"type": "local", "path": "data/semiconductor/**/*.jsonl"},
    ],
    tokenizer_path="./tokenizer",
    global_batch_size=64,
    seq_length=1024,
))

bundle = runner.run(dry_run=True)
```

## Public-policy RLVR: Mid-Scale Experiment Run (2026-08-06/07)

Location: local standalone experiment directory (not published with this repository; `saddlellm.PublicPolicyRLVR` / `PublicPolicySFT` / `GRPOTrainer` provide the framework).

Environment: Qwen/Qwen3-0.6B (revision pinned), RTX 4090 Laptop 16GB, torch 2.6.0+cu124, python 3.11. Config: 256 train samples (balanced), 100 eval samples per benchmark, 50 GRPO steps, G=4, LoRA r8/a16, lr 5e-5, beta 0.02, seed 42, reward = 0.9*exact_answer_match + 0.1*format_validity, `train_on_informative_only=true`.

Grid: `base / rl / sft / sft_rl` x `a_to_a / a_to_b` — all 8 runs completed (3 runs with SFT were finished on 08-07 via `resume_experiments.py` after an interrupted first attempt).

Results (post-training; Benchmark A = 7-class stance, Benchmark B = multi-label themes):

| Run | A acc | A macro F1 | B exact-match | Notes |
|---|---|---|---|---|
| base | 73.0% | 55.2% | 22.0% | untrained reference |
| rl (GRPO) | 73→81 (+8.0) | 55.2→66.3 (+11.1) | 22→17 (-5.0) | genuine improvement; format stable (99.5→98.5%) |
| sft | 83.0% | ~54% | 8–16% | best A accuracy, worst negative transfer |
| sft_rl | 81→73 (-8.0) | 54.5→34.6 (-19.9) | 21→39 ("+18") | collapse / reward hacking; B gain is artifact (invalid rate 1.5→13.5%) |

Key findings:

- Pure GRPO (RLVR) is the only method that genuinely improves the target task: +8.0pp accuracy and +11.1pp macro F1 on A with mild negative transfer (-5pp) on B.
- SFT alone maximizes A accuracy but destroys B (multi-label) capability; the SFT-tuned model has low answer diversity, and GRPO on top of it collapses the policy to a few answer modes (A consistency -> 99%, macro F1 -20pp). The apparent B improvement (+18pp EM) is driven by degenerate unparseable outputs (invalid rate x9) being scored by verifier fallback.
- `direction` is a no-op in `run_experiments.py` (`_execute_run` always trains on `train_a` with the A reward and evaluates both benchmarks), so `a_to_a` and `a_to_b` are duplicate runs — only 4 unique experiments.
- SFT results are not reproducible across runs: pre-eval on the same benchmark differed between SFT runs (B EM 8% / 16% / 21%). Data sampling (`AASBDataModule.select`, stable bucket by example_id + seed) and eval are deterministic, so the variance source is the SFT training pipeline itself (or a pre-fix run) — unresolved.

Artifacts:

- `outputs/experiments_midscale/` — 8 x `run_result.json` + rebuilt `experiment_summary.json` (8/8 completed)
- `outputs/experiment_results.xlsx` — all 8 runs + KEY FINDINGS now computed from JSON (was hardcoded)
- `resume_experiments.py` + `outputs/experiments_midscale/resume_run.log` — runner and log for the 3 finished runs

Next steps:

1. Fix `direction` so it actually changes the training target, or document the pairs as duplicates.
2. Investigate the SFT reproducibility issue before drawing SFT conclusions.
3. Steps sweep for base+RL (25/50/100) to study the A-gain vs B-negative-transfer tradeoff; consider early stopping on B.

## Public-policy RLVR: Bug Fixes + Verification (2026-08-07)

Four confirmed bugs were fixed (see previous entry for how they were diagnosed); all results in the previous section are now INVALIDATED and require a rerun.

Fixes:

1. **SFT non-reproducibility** (`saddlellm/PublicPolicySFT.py`): LoRA init draws from the global torch RNG inside `SFTTrainer.__init__` before trl's internal `set_seed`, so every run produced a different adapter. Fixed by explicit seeding (`random`/`np`/`torch`/`cuda`) at the top of `train_public_policy_sft`. Verified: two identical-config runs now produce **byte-identical** adapters (max-abs diff = 0.0, same train_loss).
2. **B parser prose false positives** (`saddlellm/PublicPolicyRLVR.py::extract_answer`): task B used to scan ANY text for single letters A–I, so prose like "I think this proposal is a good step..." parsed as ('A','I'). Now requires an explicit `<answer>` tag with comma/space-separated single letters or bare UNCLEAR; free text → invalid. 16/16 unit tests pass, including the exact prose case.
3. **`direction` was a no-op** (`run_experiments.py::_execute_run` + `prepare_pilot_data`): every run trained on `train_a` and evaluated BOTH benchmarks regardless of direction. Now direction selects the source split (a→train_a, b→train_b) and evaluates ONLY the target benchmark; SFT gained a `source_task` field; `--directions` CLI flag added.
4. **GRPO ratio ≡ 1** (`saddlellm/GRPOTrainer.py`): `old_log_probs = policy_log_probs.detach()` from the same forward made the ratio identically 1 and the clip dead code. Now rollout log probs are scored with a no-grad forward over the sampled token stream at rollout time (raw model distribution; `outputs.scores` is unusable in transformers 5.14 — it contains top_p-warped logits). The loss forward reuses the rollout-time padded prompt so both sides of the ratio share the same RoPE positions; residual mismatch is bf16 batch-shape rounding (~0.1–0.2 nat, same regime as trl). Note: with one update per rollout, ratio ≈ 1 is expected in standard online GRPO; the clip is now structurally live for any future change to update cadence.

Verification:

- extract_answer unit tests: 16/16 PASS
- SFT determinism (2 identical runs, in-process): adapters IDENTICAL
- GRPO log-prob alignment: rollout vs batched re-score = 0.0 (exact); vs single-sequence loss path ≤ 0.21 nat (bf16 noise)
- Pipeline smoke (`verify_fix_smoke.py`, 1-step runs): rl a_to_a, rl b_to_a, sft b_to_b all completed; each run's result contains ONLY its target benchmark; GRPO + SFT-on-B + strict parser all exercised end-to-end
- `export_results.py` updated for single-benchmark run schema (blank fills for the non-evaluated benchmark); runs cleanly

Artifacts: `verify_sft_determinism.py`, `verify_grpo_logprobs.py`, `verify_fix_smoke.py` (+ `outputs/verify_fix_smoke.log`, `outputs/experiments_fixverify/`, `outputs/sft_determinism_test/run{1,2}`) — verification scripts/artifacts, pending cleanup.

Next: rerun the full 8-run grid (old numbers invalidated); consider adding b_to_a/b_to_b directions and a steps sweep.

## Public-policy RLVR: Full Grid Rerun (v2) — Results (2026-08-07)

Full 8-run grid rerun on the fixed pipeline (256 train / 100 eval / 50 steps, LoRA r8/a16, seed 42, Qwen3-0.6B). All 8 runs completed without errors (~2h35m total). Output: `outputs/experiments_midscale_v2/`, Excel: `outputs/experiment_results.xlsx` (regenerated).

Results (target benchmark only per run; SFT runs report pre=SFT-adapter, post=after-RL):

| Run | A: acc / macro-F1 | B: EM / example-F1 / invalid |
|---|---|---|
| base a_to_a | 0.730 / 0.552 (no-op, by design) | — |
| base a_to_b | — | 0.220 / 0.459 / 0.020 |
| rl a_to_a | 0.730→0.390 (**−34.0pp**), F1 −23.9pp | — |
| rl a_to_b | — | EM −5.0pp, F1 −2.0pp, invalid 0.020→0.025 |
| sft a_to_a | 0.810→0.810 (SFT lifts base 0.73→0.81) | — |
| sft a_to_b | — | 0.210→0.210 (no transfer) |
| sft_rl a_to_a | 0.810→0.510 (**−30.0pp**), F1 −30.1pp | — |
| sft_rl a_to_b | — | EM +3.0pp, F1 +8.6pp, **invalid 0.015→0.080 (×5.3)** |

Key findings:

1. **The v1 "RLVR significantly improves A" conclusion is REVERSED.** With correct rollout log probs (Fix 4), 50 steps of GRPO on A *collapses* accuracy 0.730→0.390 (F1 0.552→0.314). The v1 gain was an artifact of the buggy ratio≈1 training signal; the v2 number is the honest one.
2. Cross-task RL (a_to_b) shows mild negative transfer (EM −5pp, F1 −2pp) — consistent with v1.
3. SFT+RL collapse is reproduced on A (−30pp acc, F1 0.545→0.244) and the apparent B gain (+8.6pp F1) is again driven by reward hacking: invalid (unparseable) output rate ×5.3.
4. SFT alone on the source task transfers to neither task (B: zero delta; the A numbers above are SFT's own lift, 0.73→0.81).

Interpretation: at this scale (0.6B, 256 prompts, G=4, lr=5e-5, 50 steps) GRPO over-optimizes the verifiable reward and destroys the answer distribution — the reward signal (0.9 exact-match + 0.1 format) is too dense/easy to hack for these tasks. Next candidates: lower lr (1e-5), fewer steps (25), larger G (8), KL penalty up (beta 0.04), or reward shaping that penalizes format degradation.

Artifacts: `outputs/midscale_v2.log` (full run log), `outputs/experiments_midscale_v2/` (8 runs), `outputs/experiment_results.xlsx` (v2, regenerated by `export_results.py`).

Pending cleanup (verification-era files): `verify_sft_determinism.py`, `verify_fix_smoke.py`, `verify_grpo_logprobs.py`, `outputs/verify_fix_smoke.log`, `outputs/experiments_fixverify/`, `outputs/sft_determinism_test/`.

User-facing summary: `outputs/v2_results_report.md` (Chinese, plain-language report with charts) + `outputs/v2_chart_{A_accuracy,B_accuracy,B_invalid}.png` (generated by `make_report_charts.py`). Charts show: A accuracy pre/post per method (only SFT lifts, RL collapses), B exact-match pre/post (negative transfer), B invalid-rate (reward hacking evidence: 1.5%→8%). Report includes interpretation ("RL teaches the model bad behavior at this scale") and next-step candidates (fewer steps, lower lr, larger G, stronger KL, stricter reward).

---

## 原生世界模型、WorldAgent 与后训练发布闭环（2026-08-09）

工作目录：SaddleLLM 项目根目录

当前包版本：`2.32`

### 本阶段目标

本阶段包含两条相互独立但可以协同的主线：

1. 不直接搬用 `reference/` 中的第三方代码，借鉴相关架构思想，构建 SaddleLLM 自己的轻量世界模型训练、推理、空间规划和反馈框架。
2. 把文本大模型的后训练流程整理成可配置、可验证、可发布的闭环，而不是只完成一次 SFT/DPO 训练。

目标工作流如下：

```text
文本模型：
数据检查 -> SFT -> DPO/ORPO/KTO -> 标准评测 -> 发布门禁 -> 导出 -> 推理服务

空间世界模型：
图片/地图 -> 感知 -> WorldState -> 路线规划 -> 世界模型模拟 -> 执行反馈 -> replay 数据
```

### 一、SaddleLLM 原生世界模型框架

已形成不依赖 `reference/` 运行代码的原生实现：

- `_WorldModel.py`
  - 连续高斯 RSSM。
  - 观测编码器、观测解码器、确定性/随机隐状态。
  - 奖励、持续概率以及可选空间预测头。
  - 支持向量观测和 CHW 图像/局部地图观测。
- `_CategoricalWorldModel.py`
  - 离散潜变量世界模型后端。
  - symlog/symexp 与分类式潜状态支持。
- `WorldModelData.py`
  - 统一离线轨迹协议。
  - 支持整段 trajectory 和逐 transition 输入。
  - 校验 `observations = actions + 1`、动作类型、奖励、终止状态及辅助目标。
- `_WorldModelTrainer.py`
  - YAML/JSON 配置训练入口。
  - 训练/验证切分、窗口采样、checkpoint 和指标输出。
- `WorldModelBackends.py`
  - 原生后端注册、创建和 checkpoint 加载。
- `WorldModelInference.py`
  - 状态编码、历史过滤、开放环 rollout、候选动作评分和轻量规划。

主要命令：

```powershell
saddle-llm world-model-backends
saddle-llm train-world-model configs\world_model.yaml --dry-run
saddle-llm train-world-model configs\world_model.yaml
saddle-llm infer-world-model outputs\world_model_example data\world_model_inference_example.json
```

### 二、空间世界模型与可视化链路

空间能力不是让语言模型直接猜路线，而是把不同职责拆开：

- `SpatialPerception.py`
  - `TopDownMapExtractor` 将干净俯视图/户型图转成 FREE/BLOCKED/UNKNOWN 占用栅格。
  - `QwenVLSpatialAnalyzer` 是可选语义适配器，只负责房间、门、障碍、危险物、地标和连通关系。
  - 明确要求透视图标记遮挡和画外区域为 unknown，不虚构隐藏几何。
- `SpatialPlanner.py`
  - A*、多路线搜索和路线多样性约束。
  - 输出长度、代价、转弯数、最小/平均净空和几何风险。
- `SpatialWorldModel.py`
  - 连接感知、占用图、路线规划和 RSSM 评分。
  - 新增 `plan_grid()`，使同一 `WorldState` 可以被重复规划，无需重复解码图片。
  - 支持 `spatial_features_v1` 向量观测和 `spatial_sequence_v2` 局部张量观测。
- `SpatialWorldModelData.py`
  - 从地图生成可直接训练的世界模型轨迹。
  - v2 数据包括局部占用、ego motion 和 collision 等监督信息。
- `SpatialWorldModelControl.py`
  - 带硬几何安全屏障的闭环 CEM/MPC。
  - 每一步重新规划，并在学习动作不安全时回退到几何专家动作。
- `SpatialWorldModelEvaluation.py`
  - 多步占用、运动和碰撞预测评测。
- `SpatialVisualization.py`
  - JSON、HTML、PNG 和占用 mask 产物。
- `SpatialAPI.py`
  - FastAPI 服务、图片上传、模型能力发现、任务产物和静态工作台。

主要命令：

```powershell
saddle-llm build-spatial-world-data data\spatial_floorplan_example.pbm `
  --output data\spatial_world_model_example.jsonl `
  --episodes 64 --routes-per-pair 2

saddle-llm train-world-model configs\spatial_world_model.yaml --dry-run
saddle-llm train-world-model configs\spatial_world_model.yaml
saddle-llm evaluate-spatial-world-model `
  outputs\spatial_world_model_example `
  data\spatial_world_model_example.jsonl
saddle-llm spatial-studio --world-model outputs\spatial_world_model_example
```

### 三、WorldAgent 统一运行时

新增 `saddlellm/WorldAgent.py`，将已有组件整理成稳定的四阶段运行协议：

```text
analyze -> plan -> simulate -> feedback
```

#### 稳定协议

- `saddle.world-state.v1`
  - 原始 observation、图片哈希、结构化语义、占用图、置信度和限制。
  - 占用栅格使用 `rle-v1` 压缩持久化。
- `saddle.world-plan.v1`
  - 起终点、Top-K 路线、推荐路线、排序模式、解释和限制。
- `saddle.world-simulation.v1`
  - 硬几何验证、可选学习预测、碰撞信息、开放环奖励和持续概率。
- `saddle.world-feedback.v1`
  - 实际路径、执行结果、计划偏差、用户说明和 replay 状态。

#### 运行能力

- `WorldAgentRuntime.analyze_image()`
  - 图片只分析一次，生成可重复使用的 `WorldState`。
  - 可以选择纯几何或 Qwen-VL 语义模式。
- `WorldAgentRuntime.plan()`
  - 基于已保存状态规划多条路线。
  - 可以使用原生 RSSM 对几何候选路线重新排序。
- `WorldAgentRuntime.simulate()`
  - `geometry`：确定性占用栅格碰撞检查。
  - `world_model`：真实调用 checkpoint 做潜空间开放环想象。
  - `auto`：有兼容 checkpoint 时使用学习预测，否则明确回退到几何验证。
- `WorldAgentRuntime.record_feedback()`
  - 比较计划路线和真实执行路径。
  - 只有提供至少两个真实轨迹点时才写入训练 replay，避免把计划路线误当现实数据。

#### 记忆与训练回放

新增 `WorldAgentStore`，默认目录：

```text
outputs/world_agent/
  observations/
  states/
  plans/
  simulations/
  feedback/
  events.jsonl
  replay/spatial_feedback.jsonl
```

`spatial_feedback.jsonl` 符合原生 `WorldModelData` 轨迹要求，可以在检查后用于下一轮世界模型训练。当前实现是“记录并导出 replay”，不会在收到一次反馈后静默在线更新模型。

### 四、WorldAgent API、CLI 与 SpatialStudio 接入

新增 `saddlellm/WorldAgentAPI.py`：

- `POST /v1/world/analyze`
- `POST /v1/world/plan`
- `POST /v1/world/simulate`
- `POST /v1/world/feedback`
- `GET /v1/world/health`
- `GET /v1/world/capabilities`
- `GET /v1/world/memory`
- 状态、计划、模拟和反馈查询接口

API 可以独立启动：

```powershell
saddle-llm world-agent-api --config configs\world_agent.yaml
```

也已挂载到原 `spatial-studio` 服务中，并与工作台共享 Qwen-VL/RSSM 的惰性加载实例，避免同一进程重复加载大模型。

新增一键运行命令：

```powershell
saddle-llm world-agent-run data\spatial_floorplan_example.pbm `
  --start 2,2 --goal 21,13 `
  --config configs\world_agent.yaml `
  --simulation geometry `
  --output build\world_agent_demo.json
```

新增简化配置：`configs/world_agent.yaml`。

配置中的 Qwen-VL 和 RSSM checkpoint 均为可选项；无模型时，几何分析和规划仍可以独立运行。

### 五、文本大模型后训练发布闭环

本阶段同时补齐了传统 LLM 后训练的发布链路。

#### 可配置流程

- `FlowConfigAdapter` 支持字段较少的 Flow 配置。
- 数据检查、SFT、偏好训练、评测、导出可以按需组合。
- `flow: full` 会编译为 SFT、偏好训练、评测和导出流程。
- DPO、ORPO、KTO 已接入统一偏好数据和训练入口。
- 原生 GRPO 仍可独立使用；极简 `sft+grpo` flow 尚未宣称稳定接入。

#### 评测发布门禁

新增 `ReleaseGate.py`：

- 支持最小值和最大值规则。
- 支持嵌套评测指标。
- 支持 required 指标和 `require_all`。
- 输出 `release_gate.json`。
- 门禁拒绝后默认停止导出，不把未达标 checkpoint 当作正式发布版本。

#### 模型导出

新增 `ModelExporter.py`：

- 导出 Hugging Face 标准目录。
- 支持 LoRA adapter 合并。
- 支持 safe serialization。
- 写出 `release_manifest.json`、文件清单、可选哈希和发布说明。
- 规范化 tokenizer 元数据，支持通用 `PreTrainedTokenizerFast` 回退。

命令：

```powershell
saddle-llm export-model CHECKPOINT OUTPUT_DIR --require-gate
```

#### OpenAI-compatible 推理服务

新增 `InferenceServer.py`：

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- Bearer token、并发上限、输入/输出 token 上限。
- 支持完整 checkpoint、已导出 release 和 LoRA adapter 加载。

命令：

```powershell
saddle-llm serve-model outputs\qwen-domain-v1\release
```

当前稳定实现是非流式单机服务；SSE、动态批处理和 KV Cache 调度仍属于后续工作。

#### PEFT 可选后端兼容

新增 `PostTrainingCompatibility` 兼容处理：

- 检测系统环境中不可用的 AutoAWQ 可选包。
- 仅关闭 PEFT 的 AWQ dispatcher，不影响普通 LoRA。
- 在 doctor 中报告经过清理的诊断和建议。
- 已验证真实 LoRA 训练、adapter 加载和 merge 后标准 Transformers 加载。

### 六、README 与配置更新

README 已增加：

- 简化后训练 flow。
- 发布门禁、模型导出和 OpenAI-compatible 服务。
- 世界模型数据、训练、评测和 SpatialStudio。
- WorldAgent 架构、四阶段 API、CLI、Python 示例、产物目录和边界。

新增/更新的主要配置：

- `configs/posttrain_release_flow.yaml`
- `configs/world_agent.yaml`
- `configs/spatial_world_model.yaml`
- `configs/spatial_world_model_v2.yaml`

### 七、实际验证结果

#### WorldAgent 示例运行

输入：`data/spatial_floorplan_example.pbm`

起点/终点：`2,2 -> 21,13`

结果：

- 成功构建 `WorldState`。
- 生成 3 条候选路线。
- 选择 `route-1`。
- 几何模拟 `route_valid = true`。
- `reached_goal = true`。
- 结果写入 `build/world_agent_demo.json`。

#### 专项测试

```text
tests/test_spatial_api.py + tests/test_world_agent.py
13 passed
```

覆盖内容：

- RLE 世界状态保存与恢复。
- 几何路线规划和模拟。
- RSSM 路线重排和开放环想象。
- 透视图默认安全拒绝。
- analyze/plan/simulate/feedback 完整 HTTP 链路。
- 反馈 replay 可被原生世界模型数据加载器读取。
- SpatialStudio 内的 `/v1/world` 路由挂载。
- CLI 和简化配置。

#### 全量回归

```text
72 passed, 2 skipped
exit code: 0
```

还完成了以下人工回归：

- 实际 SFT + DPO + eval + release gate + export 冒烟流程。
- 真实 LoRA adapter 合并。
- 合并后的模型通过标准 `AutoModel` / `AutoTokenizer` 重新加载。
- OpenAI-compatible Chat Completions 通过 TestClient 调用。

Windows 测试结束后仍可能打印 Triton AMD 后端探测引起的 access-violation 诊断噪声；本次 pytest 退出码为 0。该问题来自当前继承系统 site-packages 的 Windows 环境，不是本阶段功能测试失败，正式训练建议使用干净隔离环境。

### 八、当前边界

- 单张透视照片无法证明被遮挡空间的结构。
- Qwen-VL 是视觉语义适配器，不等于世界模型本身。
- 几何路线可行不等于现实环境绝对安全。
- 世界模型分数依赖训练数据和 checkpoint，当前 `confidence.calibrated = false`。
- 真正机器人行走还需要深度、标定、定位/SLAM、动态障碍感知和控制器。
- feedback 当前产生离线 replay，不执行未经授权的在线自动训练。
- WorldAgent 的规则解释层可追溯，但尚未接入可配置语言模型解释器。
- 文本推理 API 当前没有 SSE 流式输出和动态连续批处理。

### 九、下一步建议

优先级建议：

1. 使用多张地图、动态障碍和真实执行反馈训练 `spatial_sequence_v2` checkpoint。
2. 建立独立 held-out 地图评测，加入碰撞率、到达率、路径效率和校准误差门禁。
3. 将 SpatialStudio 前端逐步切换到分阶段 WorldAgent API，增加状态历史、模拟时间线和反馈录入界面。
4. 接入深度/SLAM observation adapter，让世界状态支持连续更新，而不是只处理单张静态图。
5. 增加世界模型 checkpoint 对比、数据版本和 replay 采样比例控制。
6. 为文本后训练补齐极简 GRPO/RLVR flow、checkpoint 横向评测和 SSE 推理。

### 十、本阶段关键文件

```text
saddlellm/_WorldModel.py
saddlellm/_CategoricalWorldModel.py
saddlellm/_WorldModelTrainer.py
saddlellm/WorldModelData.py
saddlellm/WorldModelBackends.py
saddlellm/WorldModelInference.py
saddlellm/SpatialPerception.py
saddlellm/SpatialPlanner.py
saddlellm/SpatialWorldModel.py
saddlellm/SpatialWorldModelData.py
saddlellm/SpatialWorldModelControl.py
saddlellm/SpatialWorldModelEvaluation.py
saddlellm/SpatialVisualization.py
saddlellm/SpatialAPI.py
saddlellm/WorldAgent.py
saddlellm/WorldAgentAPI.py
saddlellm/ReleaseGate.py
saddlellm/ModelExporter.py
saddlellm/InferenceServer.py
saddlellm/TokenizerLoader.py
saddlellm/PostTrainingCompatibility.py
configs/world_agent.yaml
configs/posttrain_release_flow.yaml
tests/test_world_agent.py
tests/test_spatial_api.py
tests/test_release_pipeline.py
README.md
```

---

## 模型积木、原生生命周期与分布式预训练 P0（2026-08-13）

### 一、本阶段目标

本阶段把原生 Saddle LLM 从“可以描述网络结构”推进到以下状态：

- 同一个网络既能通过 JSON/YAML 配置生成，也能通过 Python 积木 API 组合。
- Attention、FFN/MoE、残差拓扑和初始化策略可以按层配置。
- 原生模型具备保存、加载、继续预训练、精确恢复、评测、推理和发布闭环。
- 分布式能力不再只停留在 Planner；先完成可靠的 DDP 预训练底座，并为 FSDP 接好配置与启动入口。
- 尚未实现的 TP/PP/CP/EP 必须 fail-closed，不能静默退化为普通数据并行。

### 二、可配置、可编程的模型积木

完善 `ModelBlueprint` 体系：

- 新增 `LayerBlueprint` 和 `ResidualBlueprint`。
- `layers[].repeat` 可以展开为独立 Decoder Layer 实例。
- 每层可以覆盖 MHA/MQA/GQA/MLA、SwiGLU/Dense/MoE 和残差策略。
- 配置与 Python API 使用同一套规范化、校验和构建路径。
- 支持 JSON、YAML、`pathlib.Path` 读取及规范化 round trip。
- 对未知字段、非法 repeat、非法 head/cache/rope 配置提前报错。

新增代码式组合入口：

- `ModelComponentRegistry`
- `SaddleModelBuilder`
- 内置 Attention、FFN、Residual preset
- 支持自定义 registry/preset

残差网络支持：

- `serial/sequential` 与 `parallel` 拓扑。
- Attention/FFN 固定缩放和可学习标量门控。
- residual dropout。
- `standard`、`depth_scaled`、`zero` 输出投影初始化。
- 默认残差策略不增加 state-dict 参数，保持旧 checkpoint 严格兼容。

异构层已经真实接入运行时，不再被压平为全局同构配置；GQA/MLA、Dense/MoE 混合层的 forward、backward、KV cache 和保存恢复均已验证。

### 三、原生 Saddle checkpoint 生命周期

新增统一模型加载入口 `ModelLoader.py`：

- `is_saddle_checkpoint()`
- `load_causal_lm()`
- `load_model_and_tokenizer()`
- 自动区分本地 Saddle checkpoint 与 Hugging Face 模型。
- 原生 `.bin` 加载使用 `weights_only=True`，并要求安全版本的 PyTorch。

新增 `SaddleTrainer`：

- Trainer checkpoint 保存 `pytorch_model.bin`、`saddle_config.json`、模型蓝图和 tokenizer。
- 保留 Transformers Trainer 对 optimizer、scheduler、scaler、RNG 和 `trainer_state.json` 的管理。
- `resume_from_checkpoint` 表示精确恢复；`pretrain_mode=continue` 表示仅加载权重开始新 run，两种语义明确分离。
- 单进程中断恢复已经验证与不中断训练逐参数一致。

下游链路更新：

- Evaluator 和 InferenceServer 走统一 Loader。
- 原生 `generate()` 兼容常用生成参数和 EOS 停止。
- ModelExporter 支持 `format=saddle`，并验证模型、配置和真实 tokenizer 文件齐全。
- 原生 checkpoint 请求 HF 导出或 LoRA merge 时明确拒绝，不伪装成无损转换。

### 四、分布式预训练 P0

新增 `DistributedRuntime.py`，统一描述：

- `rank`
- `local_rank`
- `world_size`
- `local_world_size`
- 当前进程是否为 main process

并在运行前校验配置进程数与 Launcher 的 `WORLD_SIZE` 一致。

#### DDP

- `ddp_backend=auto` 在 CUDA/Linux 环境选择 NCCL，在 CPU/Windows 测试环境选择 Gloo。
- 支持 `num_gpus`、`num_nodes` 和最小多节点启动参数。
- 非 rank 0 不创建输出目录、训练日志、summary 或最终模型副作用。
- stability/model metrics callback 仅由 world process zero 写文件。
- 分布式模式暂时只允许 `pretrain`；eval/export 等下游阶段要求训练后单进程独立运行。

#### 激活重算

原先 `gradient_checkpointing_enable()` 只设置标记但没有重算。本阶段改为：

- 训练期逐个异构 `SaddleDecoderLayer` 调用 non-reentrant checkpoint。
- 保留 aux loss、MoE stats 和异构层输出语义。
- 训练期 activation checkpoint 与 KV cache 冲突时明确报错。
- Aux-loss-free MoE 在 backward 重算时不会重复更新 router bias。

#### MoE DDP 安全

- 某个 rank 没有 token 命中某专家时，不执行空专家完整前向。
- 通过每个专家参数的轻量零梯度连接，保证所有 rank 的 autograd 参数集合稳定。
- expert load 通过 `all_reduce(SUM)` 汇总。
- router bias、负载统计、load balance 和辅助损失使用全局负载。
- 可微 router probability 保留本地梯度，同时使用跨 rank 全局均值作为前向值。

#### FSDP、DeepSpeed 与混合并行

- FSDP 自动 wrap 类改为原生 `SaddleDecoderLayer`，不再硬编码 `LlamaDecoderLayer`。
- 生成的 Accelerate 启动命令会显式携带 FSDP/DeepSpeed 选项。
- DeepSpeed 未安装时提前报错；外部配置路径和 CPU offload 字段不再被 Orchestrator 丢弃。
- 当前原生精确恢复仅开放 single/DDP；FSDP/DeepSpeed 分片 checkpoint 恢复仍明确拒绝。
- TP/PP/CP/SP/EP 仍只有规划模型，没有 runtime collective，因此 Planner 和 Launcher 现在都会明确抛出 `NotImplementedError`，不再映射为 ZeRO-3 或普通 DDP。

### 五、分布式 checkpoint 契约

`saddle_checkpoint.json` 新增并行信息：

```json
{
  "parallelism": {
    "strategy": "ddp",
    "world_size": 2,
    "rank": 0,
    "state_dict_type": "full"
  }
}
```

精确 DDP 恢复会校验：

- checkpoint 保存时的 world size 与当前配置一致。
- 每个 rank 都存在对应的 `rng_state_<rank>.pth`。
- 模型、optimizer、scheduler 和 Trainer state 文件齐全。
- weights-only 的 `final_model` 不能冒充 Trainer 精确恢复目录。

### 六、配置与文档

分布式配置已贯通：

- `TrainingRecipe`
- `FlowConfigAdapter`
- `TrainingOrchestrator`
- `TrainingConfigValidator`
- `BackendAdapterRegistry`
- CLI `plan` 生成的启动计划

README 新增分布式预训练配置、启动方式、支持范围和当前限制。

### 七、测试结果

新增：

- `tests/test_model_composition.py`
- `tests/test_native_model_lifecycle.py`
- `tests/test_model_distributed_runtime.py`
- `tests/test_distributed_pretrain.py`

分布式专项覆盖：

- Windows `spawn` 兼容的真实双进程 CPU/Gloo DDP。
- 相同 global batch 下，DDP 单步更新与单进程基线一致。
- activation checkpoint 前后 logits、loss、MoE stats 和参数梯度一致。
- MoE 空专家零梯度、全局负载同步和 router bias 单次更新。
- rank 0 checkpoint/manifest 与 rank 1 无文件副作用。
- resume world size 和各 rank RNG 完整性。
- FSDP 配置与 Accelerate JSON 构造。
- Hybrid Planner/Launcher fail-closed。

联合专项结果：

```text
64 passed
exit code: 0
```

全量回归：

```text
136 passed, 2 skipped, 4 warnings
exit code: 0
```

Windows 测试进程退出后仍可能打印现有 Triton AMD backend access-violation 尾栈，但 pytest 断言和退出码均正常。该噪声不是本阶段逻辑回归。

### 八、当前边界与下一步

- 当前机器只有单张 CUDA GPU，且 Windows PyTorch 不提供 NCCL，因此 CUDA DDP/FSDP 尚未在本机完成性能验收。
- FSDP 已完成配置、wrap 和启动接线，但正式声明可用前仍需 Linux 2/4 GPU fresh-run、保存和恢复验证。
- DeepSpeed ZeRO-2/3 尚未完成原生分片 checkpoint 生命周期验证。
- 分布式数据预处理当前要求非 streaming 数据；大规模数据应进一步改成预处理一次、所有 rank 只读分片。
- TP/PP/CP/SP/EP 尚未实现运行时。

下一步优先级：

1. 在 Linux 2/4 GPU 环境验证 DDP/FSDP loss parity、峰值显存、吞吐和中断恢复。
2. 建立 tokens/s、tokens/s/GPU、峰值显存和 scaling efficiency 基准报告。
3. 实现 TP + sequence parallel：Q/K/V、gate/up 列并行，O/down 行并行，并补 vocab-parallel CE。
4. 为 MoE 实现 expert parallel 和双向 all-to-all token dispatch。
5. 再实现 Pipeline Parallel 1F1B 与长上下文 Context Parallel/ring attention。

### 九、本阶段关键文件

```text
saddlellm/ModelBlueprint.py
saddlellm/ModelBuilder.py
saddlellm/SaddleModeling.py
saddlellm/ModelLoader.py
saddlellm/NativeTrainer.py
saddlellm/DistributedRuntime.py
saddlellm/DistributedConfig.py
saddlellm/BackendAdapters.py
saddlellm/FactoryBackendPlanner.py
saddlellm/TrainingOrchestrator.py
saddlellm/TrainingRecipe.py
saddlellm/TrainingConfigValidator.py
saddlellm/ModelExporter.py
saddlellm/InferenceServer.py
saddlellm/LLModelEvalute.py
tests/test_model_composition.py
tests/test_native_model_lifecycle.py
tests/test_model_distributed_runtime.py
tests/test_distributed_pretrain.py
README.md
```

---

## 原始多媒体到生成训练缓存闭环（2026-08-14）

### 一、本阶段目标

在已有文本预训练、SFT、偏好训练、图像/音乐/视频生成和世界模型训练能力之上，补齐此前缺失的原始媒体入口，使框架能够在同一条流水线中完成：

```text
原始 JSON/JSONL 数据
  -> 文本条件编码
  -> 图像/音频/视频表征编码
  -> 可恢复的分片训练缓存
  -> 图像、音乐或视频生成训练器
  -> checkpoint 与训练报告
```

本阶段的重点不是引入大型生产 Codec，而是先建立稳定的 Codec 契约、缓存协议、插件扩展点和端到端编排闭环，为后续接入 VAE、神经音频 Codec 和视频 Tokenizer 提供底座。

### 二、内置文本编码器与媒体 Codec

新增 `saddlellm/BuiltinMediaCodecs.py`：

- `HashTextConditionEncoder`
  - 使用带 seed 的 BLAKE2b 字节 n-gram 哈希生成固定维度条件向量。
  - 输出归一化、确定性向量，默认不依赖网络，适合测试和离线基线。
- `HuggingFaceTextConditionEncoder`
  - 通过 `AutoTokenizer` 和 `AutoModel` 加载文本模型。
  - 对最后一层隐藏状态执行 attention-mask mean pooling。
  - 支持 lazy dependency、`local_files_only` 和 `trust_remote_code`。
- `RGBImageCodec`
  - 使用 Pillow 完成 RGB 转换与尺寸归一化。
  - 编码为 `[-1, 1]` 范围的连续 CHW latent，并提供反向解码。
- `WAVResidualCodec`
  - 使用 Python 标准库读取 8/16/24/32-bit PCM WAV。
  - 支持单声道化、线性重采样、帧聚合和多级残差标量量化。
  - 输出 `[Q, T]` 离散 codes，同时生成有效帧 `attention_mask`。
- `FrameVideoCodec`
  - 支持视频帧目录及 Pillow 可读取的动画图像。
  - 完成均匀采样、尺寸归一化并输出 `[C, T, H, W]` latent。
- `register_builtin_media_codecs()`
  - 幂等注册内置 Codec，保持与自定义插件 Codec 共存。

这些 Codec 是端到端闭环和接口参考实现，不代表生产重建质量。

### 三、可恢复的分片媒体缓存

新增 `saddlellm/MediaCache.py`：

- `MediaCacheBuildConfig`
  - 统一描述输入数据、输出目录、模态、Codec、文本编码器、字段映射、分片大小和失败策略。
- `build_media_cache()`
  - 复用 `MultimodalDataAdapter` 读取 JSON/JSONL 记录。
  - 从通用媒体字段或多模态 segments 中解析本地资源。
  - 编码条件和媒体表征，并原子写入压缩 NPZ 分片。
- `ShardedNpzStore`
  - 支持从缓存目录或 `manifest.json` 打开数据集。
  - 支持跨分片随机访问、二分定位和单分片句柄缓存。

缓存 manifest 记录：

- schema 与版本；
- 输入绝对路径、大小、修改时间和 SHA256；
- Codec 与文本编码器身份指纹；
- 数组 shape、dtype、范围及可选 `attention_mask`；
- 每个分片的 SHA256、样本数和原始记录索引；
- 已处理记录、有效样本、失败记录和构建状态。

可靠性约束：

- manifest 与分片均采用临时文件后原子替换；
- `resume` 会校验输入、Codec 和条件编码器身份，拒绝不兼容续跑；
- 已完成缓存重复执行时保持幂等；
- strict 模式失败后不会越过尚未落盘的有效样本；
- overwrite 只清理受合法 manifest 管理的缓存文件，拒绝清空未知非空目录；
- non-strict 模式允许跳过坏样本，并把失败详情写入 manifest；
- `plugin_modules` 可在构建前导入第三方 Codec 注册模块。

### 四、训练数据集与统一编排接入

以下训练数据集现在同时兼容旧的单文件 NPZ 和新的分片缓存目录/manifest：

- `CachedLatentDataset`
- `CachedMusicCodeDataset`
- `CachedVideoLatentDataset`

音乐数据集会把音频有效帧 mask 传给训练模型；训练前的 shape、condition dimension、code range 和 vocabulary 校验可以直接读取 manifest 元数据，不需要把全部分片加载到内存。

`UnifiedTrainingOrchestrator` 新增 `media_cache` stage：

- 一个 pipeline 可以先构建缓存，再运行 `image_generation`、`music_generation` 或 `video_generation`。
- 下游阶段可以省略 `data_path`，由编排器按模态解析上游缓存。
- 同一阶段拒绝重复模态 job，避免下游选择产生歧义。
- dry-run 只验证和输出计划，不创建缓存。
- 下游训练计划及 checkpoint 配置会携带实际 Codec、文本编码器和缓存 manifest 信息。
- 音乐模型 vocabulary 优先使用 Codec 声明的 `codebook_size`，避免只按当前样本最大 code 推断。

### 五、CLI 与示例配置

CLI 新增：

```text
saddlellm build-media-cache CONFIG [--no-resume] [--overwrite] [--output PATH]
saddlellm media-codecs [--modality image|audio|video] [--output PATH]
```

新增配置：

- `configs/media_cache_image.yaml`
- `configs/media_cache_music.yaml`
- `configs/media_cache_video.yaml`
- `configs/raw_image_to_training.yaml`

其中 `raw_image_to_training.yaml` 演示单次运行从原始图像记录构建缓存，再完成一步图像生成训练并保存 checkpoint。

### 六、测试与验证

新增 `tests/test_media_cache.py`，覆盖：

- 三种内置媒体 Codec 的编码/解码 shape；
- Hash 文本编码器的确定性和归一化；
- 图像、音频、视频分片缓存构建及训练数据集读取；
- 音频 attention mask；
- 完成缓存的幂等执行；
- non-strict 坏样本记录；
- Codec 身份变化时拒绝 resume；
- CLI 缓存构建和 Codec 列举；
- 原始图像到生成训练 checkpoint 的单流水线闭环；
- dry-run 不产生缓存副作用。

本阶段相关定向回归：

```text
28 passed
exit code: 0
```

全量回归：

```text
170 passed, 2 skipped, 17 warnings
exit code: 0
```

Windows 进程退出阶段仍会打印已有的 Triton AMD backend access-violation 尾栈，但 pytest 已全部完成且退出码为 0，不属于本阶段代码失败。

### 七、当前边界与后续优先级

当前边界：

- `RGBImageCodec` 直接使用像素空间，不具备生产级 VAE 的压缩率和感知重建质量。
- `WAVResidualCodec` 是确定性残差标量量化基线，不等价于 EnCodec、DAC 或 SoundStream。
- `FrameVideoCodec` 仅建模归一化帧张量，没有生产级时空 VAE/Tokenizer 的压缩和运动表征能力。
- Hash 文本编码器只用于离线基线；正式训练应接入匹配目标模型的文本编码器。
- 尚未建立 FID/FVD、CLIPScore、音频质量、重建误差和跨模态一致性的统一评测门禁。
- 大规模对象存储、流式数据、分布式并行预处理和全局缓存索引尚未实现。

后续优先级：

1. 通过插件契约接入生产级图像 VAE、EnCodec/DAC 和视频时空 VAE/Tokenizer。
2. 增加 Codec 批量推理、GPU/混合精度、worker pool 和分布式离线编码。
3. 建立重建质量、生成质量、跨模态对齐和世界模型预测能力的统一评测体系。
4. 将图像、音频、视频和动作统一为带时间戳、mask、模态类型与来源信息的序列协议。
5. 将生成模型与世界模型从共享训练基础设施推进到共享表征空间、状态转移目标和联合训练任务。

### 八、本阶段关键文件

```text
saddlellm/BuiltinMediaCodecs.py
saddlellm/MediaCache.py
saddlellm/LatentFlowTrainer.py
saddlellm/MusicCodeTrainer.py
saddlellm/VideoLatentFlowTrainer.py
saddlellm/UnifiedTrainingOrchestrator.py
saddlellm/cli.py
saddlellm/__init__.py
configs/media_cache_image.yaml
configs/media_cache_music.yaml
configs/media_cache_video.yaml
configs/raw_image_to_training.yaml
tests/test_media_cache.py
docs/MULTIMODAL_GENERATION.md
docs/FUNCTION_REFERENCE.md
docs/ARCHITECTURE.md
README.md
```

## 2026-10-01 至 2026-10-02 源码导读与流水线缺陷修复

本次先梳理项目框架和主要功能的代码实现流程，形成源码学习手册；随后针对阅读中发现的问题做最小复现，并修复四处流水线缺陷。工作基于当前工作区源码，包版本为 `2.40`，包含此前已有的未提交改动。本节只记录本次会话完成的工作，不将其他工作区改动归入本次成果。

### 源码梳理与学习材料

- 梳理 CLI 入口、配置解析、Stage 注册、依赖计划、执行编排、状态恢复和产物登记之间的关系。
- 追踪数据与 Tokenizer、原生模型前向、预训练、SFT/DPO、蒸馏压缩、评测导出、推理服务及生成会话的主要调用链。
- 补充多模态、世界模型、空间规划和 WorldAgent 的入口、数据流及当前实现边界。
- 为主要功能整理关键代码、源码定位、断点建议、关联测试和自查题，配套 12 讲学习路线。

交付文件：

| 文件 | 内容 |
|---|---|
| [主要功能源码导读](docs/CORE_FEATURE_CODE_WALKTHROUGH_ZH.md) | 16 节功能实现流程与源码说明 |
| [源码阅读路线](docs/SOURCE_READING_ROADMAP_ZH.md) | 12 讲学习顺序、练习和掌握标准 |
| [Word 学习手册](SaddleLLM_主要功能源码解读与学习手册.docx) | 合并阅读路线与源码导读，含可跳转目录、源码索引和流程图 |
| [手册生成脚本](tools/build_source_reading_handbook.py) | 从上述 Markdown 生成 Word 手册 |

源码梳理时通过 AST 核对了 73 个类或函数定位，检查了 139 个本地链接。执行 TinyStories CLI dry-run 和纯 Python 计划示例，并运行编排及插件测试，结果为 `18 passed`；这些验证覆盖配置、计划和流程契约，不代表完整模型训练验收。

Word 手册共 32 页，包含 73 个源码书签、92 个内部链接、13 张表和 2 张图。通过 Word 页面渲染与逐页检查，修正了一处孤立表头和一条流程图连接线。文档结构及书签检查通过，记录见 `outputs/source_handbook_20261001/final_qa.json`。

### 四处已复现缺陷与修复

| 问题 | 复现行为 | 修复后的行为 |
|---|---|---|
| 关闭的门禁被当作通过 | `export.require_gate=true` 且 `eval.gate.enabled=false` 时，流水线仍完成导出 | 导出阶段复用 `release_gate_accepted()`，关闭门禁不能满足强制门禁要求 |
| 评测模型与导出模型不一致 | 评测 A，通过 `export.model_path` 导出 B，B 的发布清单仍携带 A 的通过结果 | 门禁记录评测模型来源；启用门禁时校验导出来源一致，同一本地模型的相对路径和绝对路径可匹配；非强制导出不携带不相关的通过结果 |
| 失败阶段在恢复时被跳过 | `eval.fail_on_error=false` 时，评测结果为 `failed`，外层状态却是 `completed`，`resume` 不重试 | 保存 `failed`、`rejected`、`blocked` 状态；恢复时重试未成功阶段，并使其依赖下游的旧结果和产物失效，保留独立成功分支 |
| `vision_alignment` 结果归属错误 | 执行该阶段却写入 `mllm_sft` 结果，自己的状态记录为空，产物登记缺失 | 增加对应阶段入口，按实际阶段名写结果、计划和规范化数据；与 `mllm_sft` 使用独立产物位置 |

兼容性处理：恢复逻辑同时检查外层状态、`result_status` 和结果内的状态，能够识别旧文件中“外层 completed、内部 failed”的记录。导出也兼容旧评测结果把模型来源放在门禁对象旁边的格式。

本次没有新增权限、审批流程、依赖或强制门禁配置。普通导出仍可在不运行评测的情况下执行。没有借此重构整个配置校验体系或改变所有 Stage 的状态协议。

主要代码与测试改动：

- [pipeline.py](saddlellm/framework/pipeline.py)：阶段状态、恢复选择及下游失效处理。
- [stages.py](saddlellm/framework/stages.py)：`vision_alignment` 内置执行入口映射。
- [TrainingOrchestrator.py](saddlellm/training/TrainingOrchestrator.py)：门禁判断、模型来源绑定、多模态阶段结果及产物归属。
- [test_pipeline_reliability.py](tests/test_pipeline_reliability.py)：新增 18 项回归用例，覆盖错误路径、默认导出、路径别名、旧状态兼容及两个多模态阶段共存。
- [FUNCTION_REFERENCE.md](docs/FUNCTION_REFERENCE.md)：重新生成函数签名和源码行号索引。

### 修复后的验证结果

执行以下相关测试集：

```powershell
python -m pytest tests/test_pipeline_reliability.py tests/test_unified_pipeline.py tests/test_framework_plugins.py tests/test_release_pipeline.py tests/test_cli_output.py -q --tb=short
```

结果为 **80 passed，0 failed，6 warnings**，耗时约 38 秒。警告来自 pynvml、PEFT 和 AnyIO 依赖，不影响本次用例通过。该结果是定向回归，不是仓库全量测试结果。

其他检查通过：

- 对修改的生产代码及新增测试执行 Ruff 的 `F601,F821,F822,F823` 检查；新增测试文件的默认 Ruff 检查也通过。
- `python tools/check_release_consistency.py` 确认版本元数据一致，版本仍为 `2.40`。
- `python tools/generate_function_reference.py --check` 确认函数索引与源码一致。
- 对修改的已跟踪文件执行 `git diff --check`，未发现补丁空白格式问题。

复现与验证记录保存在本地工作区：

- `outputs/review_findings_20261001/reproduction_results.json`：修复前四处问题的最小复现结果。
- `outputs/bugfix_20261002/validation.json`：修复范围、测试结果和相关文件哈希。
- `outputs/bugfix_20261002/baseline/` 及同目录差异文件：本次修复前的文件副本与任务内差异，便于区分此前已有改动。

### 验证范围与后续接续点

- 本次验证侧重流程、恢复和发布契约，没有执行完整大模型训练或 GPU 性能验收。
- 模型绑定比较规范化后的本地来源路径或远程模型标识，没有全量读取权重计算哈希；不能据此保证同一路径下被替换的权重或同名远程模型版本未变化。
- 已在旧状态文件中保存为空字典的 `vision_alignment` 结果没有做历史数据迁移；遇到这类旧记录时，可通过 `pipeline.rerun: [vision_alignment]` 重新生成结果。
- `planned` 等计划状态以及 `stages_completed` 摘要字段的整体表达尚未统一；DAG 仍负责依赖排序、串行执行，`consumes/produces` 仍主要是能力声明。这些未纳入本次四处缺陷修复。
- Word 手册是 2026-10-01 修复前的源码基线。后续精读时以当前源码、本节修复说明及更新后的函数索引为准；本次未重新生成 Word 手册，旧行号和恢复状态说明可能需要更新。
- 本次未执行 Git 提交、推送或发布。按用户要求先结束本轮工作；后续源码讲解可从 `train` 命令、配置解析和阶段编排开始。

## 2026-10-02 推理性能优化、远程对照与决策模型可行性

本轮基于已有脏工作区继续改进推理，没有将此前的原生合批、Qwen3 适配或其他未提交工作计为本轮新增。生产代码相对任务开始快照的差异保存在 `artifacts/inference_speed_20261002/task_production_changes.patch`。

- 优化前缀缓存的模型状态扫描和 token 前缀比较，保留结构、权重版本、hook 与设备等失效检查。支持最短缓存门槛和显式延迟快照实验；未通过整体收益门槛的部分复用策略没有自动启用。
- 对受支持的 CUDA 原生合批请求合并采样逐元素操作和 token 回传；保留逐行概率归约及采样顺序。自定义回调、采样/停止逻辑和 tokenizer decoder 回退到原路径。
- 增加统一的 `performance=auto|latency|throughput` / `--performance` 入口。原有 auto 行为保留；显式配置优先，预设本身不修改精度、量化或数值模式，模型加载后检查支持条件并原子回退。
- 本机真实 SSE 对照：1.5B 重复完整长提示的首个可见文本中位耗时从旧缓存 63.19 ms 降至 38.09 ms；28M 四并发流式交付的配对吞吐中位提升约 7.1%，7 轮中 6 轮更快。大模型单请求没有得到普遍提速结论。
- 修复基准适配器的全局方法恢复竞态，作废旧 HTTP 记录，增加实际进入新路径和方法恢复检查。跨引擎客户端支持预热/正式阶段互不重复的提示，保留实际输出长度、原文、首字与完整响应耗时。
- 相关推理与服务回归 676 passed、1 skipped；基准工具回归 25 passed。跳过项为上游 tokenizers 的 custom-decoder Sequence 构造 panic。所改文件静态检查、函数索引、发布元数据与 wheel 内容核对通过；全仓此前已有的重复 `RolloutSample` 导出键未在本轮修改。
- 生成并核验 SaddleLLM 2.40 wheel，SHA256 为 `c1d7fd3b6921a516e1a8a5a63e33ccf774a566bfd54a80963d53308c2471088c`，部署到指定 L20 服务器的独立测试环境。完整部署参数、实测及证据边界见远程报告。
- 按用户要求研究 Jev。结论为可以扩展训练自有决策模型；现有文本骨干、训练和部分评测可复用，但仍需决策头、动态候选数据/loss、独立概率校准、导出与服务闭环。本轮仅完成研究，没有声称已训练模型或复现厂商速度。

文档与证据：

- [本机优化与 A/B 结果](docs/INFERENCE_SPEED_20261002.md)
- [L20 跨引擎部署比较](docs/REMOTE_INFERENCE_COMPARISON_20261002.md)
- [Jev 可行性与实施路线](docs/FRONTIER_MODEL_ROADMAP.md)
- [本地验收清单](artifacts/inference_speed_20261002/local_validation.json)
- [远程结果独立审计](artifacts/inference_speed_20261002/remote_audit.json)

## 2026-10-18 — 后训练能力保持与压缩恢复

本轮在既有未提交修改上增量实现，不重置或归并此前工作。

- HF/原生 SFT 支持确定性通用/历史数据回放、比例或显式数量、验证数据排除与来源指纹。HF 在实际 TRL 分词、EOS 和截断后按输入 token 审核回放；自动验证拆分按 prompt/content/group 关联分组，显式主任务验证集保持原分配契约。
- 加入可选参考模型 KL：冻结教师，按移位后的有效监督 token 计算，验证仍使用任务 CE；单设备/DDP 支持，未适配的分片与 DataParallel 明确拒绝。
- 新增统一多任务 `regression` 套件：绝对退化预算、min/max、数据/设置/模型内容绑定、离线重比与非零失败退出。训练 eval 可直接使用套件，未通过或导出内容已改变时阻止导出。
- 新增 `recover`：实际加载 HF/原生学生检查点，以 CE 和可选教师 KL 恢复；梯度和优化后均施加剪枝掩码，保存精确掩码和恢复前后损失/稀疏率。独立验证按组/提示审核；拒绝直接训练打包量化产物。
- 新增模板配置、最小样例和 [使用说明](docs/POSTTRAIN_RETENTION_AND_RECOVERY.md)。示例只验证流程，未宣称生产模型能力提升、GPU 加速或实现 AWQ/GPTQ/QAT。
- 最终合并回归 **346 passed**（248.85 秒），包含独立损失/泄漏/身份边界测试、HF/native 真实 CPU 训练，以及 CLI 的 SFT→剪枝→恢复→量化→回归→受控导出。新增模块与测试 Ruff 通过，所改 Python 文件编译及 diff 空白检查通过。机器可读验收记录为 `outputs/lifecycle-validation/final.xml`。

## 2026-10-18 — 通过函数配置启用 AWQ/GPTQ 或 QAT

- 新增 `apply_quantization(model_or_path, tokenizer, config=..., output_dir=...)` 与 `QuantizationConfig`：`ptq.enabled`、`qat.enabled` 默认关闭，关闭路径保持输入对象、不导入模型运行时、不写文件；`dry_run` 只校验。当前两个后端的数值/导出格式不同，同时开启会在模型加载前报错。
- PTQ 使用实际 LLM Compressor AWQ/GPTQ 校准，支持 INT4/INT8，导出并核验 compressed-tensors 整数权重。依赖固定为 Torch 2.6、Transformers 4.57.6、LLM Compressor 0.6.0、compressed-tensors 0.10.2；Windows 使用 Triton 3.2.0.post21。默认 basic pipeline 使用真实前向统计，明确遵守 CPU 设备配置；大模型校准内存仍须实际验证。
- QAT 实现 `portable-int8-weight-only`：逐输出通道对称 INT8 权重、FP32 激活、STE fake quant、真实转换与重载，支持 HF 和原生模型、可选教师蒸馏与剪枝掩码。该参考后端用反量化后的浮点矩阵计算，未实现 INT8 加速内核。
- 统一加载、评测与本地推理识别新格式；QAT 自动精度保持 FP32。PTQ 使用官方 compressed-tensors 解压 API，避开固定版本 HF 自动量化加载的 meta tensor 问题；解压加载不代表低比特显存/吞吐收益。未宣称 vLLM 压缩部署已打通。
- CLI `quantize --config/--method` 复用函数入口，保留原 dynamic INT8 默认路径；提供两份配置和 [函数使用说明](docs/QUANTIZATION_FUNCTION_API.md)。扩展 wheel 依赖/模块契约，保护已有模型、数据和配置不被输出覆盖。
- 主环境定向回归 **323 passed、4 skipped**（65.76 秒）；4 个真实 PTQ 用例因主环境未安装可选后端而跳过，已在独立环境完整运行 PTQ 套件，结果 **47 passed、0 skipped**（38.96 秒），覆盖 AWQ/GPTQ × INT4/INT8 的实际 CPU 校准、整数打包、加载和生成。两组测试有重叠，不相加作为独立用例数。证据：`outputs/quantization-validation/final.xml`、`outputs/quantization-validation/real-ptq-final.xml`。
- 新模块与测试 Ruff、修改文件编译、diff 空白检查通过；最终 2.42 wheel 构建及内容检查通过。本轮没有改动全局 Python 依赖，PTQ 验证依赖安装在 `outputs/quantization-validation/ptq-env`。

## 2026-10-18 — 训练流水线中的量化交接与压缩产物导出

- 在标准调度器注册单进程 `quantize` 阶段，新增 `QuantizationStageConfig`；训练配置的 `quantization.ptq/qat` 复用函数入口，两组开关默认关闭。关闭时不发布量化模型路径；未安排量化阶段的旧运行保留原配置指纹，不因新增字段失去断点恢复能力。
- 开启时校验训练→量化→评测→导出的顺序；显式 DAG 同时检查传递依赖，防止重跑训练后恢复过期量化或评测结果。默认评测与导出使用量化产物，不再叠加 adapter；修复旧评测器在 CUDA 上给 QAT 强制选择 FP16 的问题。
- LoRA 输入先合并成 FP32 浮点检查点，保存在量化目录同级的 `-float` 目录。记录 adapter、基座和合并产物的内容身份；量化失败后的重试仅复用可核验且未变化的浮点模型。禁止把量化输出或合并产物写进原 adapter 的本地基座目录。量化执行前释放旧模型缓存。
- 导出器直接复制 PTQ/QAT 整数权重、量化 manifest 与 tokenizer；支持原生 QAT 的独立状态文件布局，导出前检查格式、分片与门禁身份。普通检查点/adapter 合并支持精确 `cuda:N`，不静默换设备。现有 dynamic INT8 路径保留。
- 新增 `configs/posttrain_quantization_qat.yaml`，更新函数说明和生成式函数索引；README 当前版本标记与既有包元数据 `2.42` 对齐，没有另行升级版本。CLI 样例 dry-run、发布元数据、函数索引、最终 wheel 内容及无模型运行时的导入/关闭路径检查通过。
- 主集成回归 **171 passed、3 skipped**（73.12 秒），函数/QAT/恢复等兼容回归 **115 passed**（47.17 秒），补充基座目录保护 **1 passed**。主环境跳过的 3 项真实 PTQ 已在独立后端环境验证为 **3 passed、0 skipped**（52.01 秒），覆盖 GPTQ 流水线与 AWQ/GPTQ 压缩产物导出重载。各测试组有重叠，不相加作全仓用例总数；未执行全仓测试，也未声称低比特部署加速验收。
- 补充真实 LoRA 优化一步→FP32 合并→QAT→导出→重载验收 **1 passed、0 skipped**（37.38 秒），确认适配器已更新、合并模型与训练后适配器 logits 一致、准备记录身份可核验、最终 INT8 权重及重载 logits 保持一致。记录为 `outputs/quantization-stage-validation/adapter-real.xml`。
- 本轮机器可读测试、构建日志及 wheel 位于 `outputs/quantization-stage-validation/`。

## 2026-10-18 — 大模型面试知识笔记与框架采用情况审计

- 整理 [40 个核心问答](docs/LLM_INTERVIEW_NOTES_ZH.md)，覆盖模型、数据、预训练、SFT/能力保持、偏好优化、蒸馏、量化剪枝、推理服务、RAG/Agent 与评测；技术解释使用论文或官方资料，面试题库只用于选题。
- 新增 [框架源码对照](docs/LLM_INTERVIEW_FRAMEWORK_AUDIT_ZH.md)，区分原生实现、第三方集成、受限能力、错误/占位和缺少闭环，给出源码位置、复现证据、整改优先级及验收条件。README 增加两份文档入口。
- 只读复现旧 benchmark 将 PPL 标成 GSM8K accuracy、投机解码输出错误、仅哈希前 2,000 字符导致不同全文被删、数据配比公开入口模拟训练，以及 Scaling 预测忽略拟合 A/B 系数等问题。发现均保留为待办，本轮没有修改生产训练、压缩或推理算法。
- 定向测试分别为架构/数据 **71 passed**、训练运行时 **24 passed**、后训练 **90 passed**、推理/应用 **345 passed**；组间可能重叠，不相加作全仓唯一用例数。没有执行全仓测试，也没有宣称真实多 GPU 或低比特部署性能已验收。
- 按用户要求生成根目录 `SaddleLLM_大模型面试知识与框架对照笔记.docx`，共 26 页，包含目录、源码/论文链接及 16 个原生可编辑公式。官方 Windows managed runtime 未携带 LibreOffice；保留标准渲染器失败日志，改用隐藏 Microsoft Word 16 原生导出与 managed Poppler 生成页面图。
- 文档链接检查覆盖 91 个本地引用；Word 结构检查覆盖 Q01–Q40 顺序、内部锚点、重复表头和 ZIP 完整性，并核验 54 个审计源码文件与两份正文 Markdown 的 SHA256 未变。全部 26 页已按原尺寸逐页目视检查，修复目录后的意外空白页；原始测试、探针、构建与页面核验记录位于 `outputs/interview-audit/`。

## 2026-10-19 — 正确性、训练契约与量化执行链路整改

- 按用户授权落实上轮建议，保留既有未提交修改。修正旧 BenchmarkRunner 的 GSM8K/PPL 混标、样本数/科目范围和失败记录；HumanEval/未知任务在加载前明确拒绝。全文规范化 SHA256 去重保留尾部不同的长文，缺近似依赖时记录实际 exact fallback；停用随机 loss/空 checkpoint 的 BalancedPretrainer 公开原型；Scaling 拟合和预算最优解使用实际 A/B/alpha/beta。
- 修复参考投机解码的 greedy 接受/拒绝、p/q 接受概率、残差分布、bonus/EOS/预算、拼接与固定截断问题；只报告实际生成计时和吞吐，没有 baseline 时不估算 speedup。
- split、replay 与审计复用统一来源身份，HF preference 在归一化展开前按来源/prompt 分组；HF/native SFT、DPO 与 HF ORPO/KTO、公开蒸馏接入 raw/prepared 审计，覆盖 token 截断、padding、TRL completion/assistant masks 和不同偏好布局。新增 packing_attention 配置，显式区分 concat/isolated，不支持的隔离请求提前拒绝。
- 公开蒸馏复用 causal shift/mask/tokenmean/T² loss，真实执行 batch 与独立验证，冻结并校验教师/词表/hidden；GRPO 保留真实 EOS，屏蔽其后 padding。补充独立 DPO/GRPO 目标、KL 与梯度数值对照。
- 新增 select_checkpoint 函数及 eval.checkpoint_candidates 配置：候选通过完整能力门槛后才按指定新任务指标排序，保存各候选与最终报告；覆盖来源文件保护、路径冲突、内容身份变更、失败阻断与实际导出获选模型。默认未开启的新增选模/packing 配置保持旧运行指纹兼容。消除 RolloutSample 重复导出键，保留原生效指向并增加明确的 RLScalingRolloutSample 名称。
- 为 portable QAT 增加显式 cuda-w8a16 加载，直接复用既有 Triton 核，保留原整数 codes/FP32 scales，不重新量化；默认 W8A32 参考加载保持原语义。真实 native/HF FP16/BF16 forward/generate/重载与非对齐 shape 已验证，训练/不兼容设备或 dtype/普通浮点导出明确拒绝。
- 最终主环境合并回归 **563 passed、3 skipped**（28 文件，231.06 秒）；三项真实 PTQ 已在既有独立后端环境补测 **3 passed、0 skipped**（35.41 秒），汇总核对了 testcase ID 完全对应主环境跳过项。没有执行全仓、业务封存数据或真实多 GPU 验收。
- RTX 4090 Laptop 的两组微型 GPU 夹具计时显示分配峰值下降，部分场景速度下降；当前不宣称普遍提速或生产质量达标。原始计时、实际内核计数、重载/文件身份及源码 SHA256 保存于 `outputs/correctness-hardening/`。
- 仓库必检 Ruff F601/F821/F822/F823、新模块/测试 Ruff、diff 空白、版本元数据与函数索引检查通过。使用声明的隔离构建依赖生成 2.42 wheel，内容检查、无 torch 的隔离 API 导入及全部 25 个修改生产文件与 wheel 字节一致性核对通过；没有修改全局 Python 依赖。
- [整改说明与配置示例](docs/CORRECTNESS_HARDENING_20261019.md)、[最终验证汇总](outputs/correctness-hardening/validation.json)。2026-10-18 的 Word 和原审计保留为整改前快照，并增加当前实施状态链接。

## 2026-10-19 — 主流程可读性整理

- 按用户要求，对刚修改的 12 个生产文件做可读性整理，不增加功能或配置。数据审计按收集、比较、报告、落盘/阻断排列；选模按验证、身份快照、评测、复查、排序、保存排列；量化入口按校验、加载、执行方式排列。
- 展开密集条件、嵌套三元和字典构造，区分原始数据、训练数据、验证数据的命名。只提取完整职责的报告路径校验、报告写入、产物加载和推理保护安装，不增加策略类或新的框架层；投机解码的数学循环保持不动。
- 后续编码要求：主流程按执行顺序展开，命名直接说明用途，注释解释原因与边界；减少嵌套和重复，仅在职责完整、便于阅读时提取函数。修改时核对公开接口、默认行为和相关回归，持续保证代码逻辑顺畅、容易阅读和维护。
- 相关 18 个测试文件合并回归 **343 passed、0 skipped**（169.39 秒）。额外核对 96 项公开签名/数据类字段与上一轮 wheel 一致；选模 10 个前后差分场景、量化加载 12 项结构核验和局部格式化 AST 核验通过。各专项结果与合并回归有重叠，不相加作唯一用例总数。
- 必检 Ruff 规则、修改的辅助模块 Ruff、diff 空白、函数索引和版本元数据通过；新 2.42 wheel 的内容、源码字节一致性及无 torch 的隔离 API 导入通过。未重测性能，之前的微型 GPU 计时保留为上一轮数值实现的证据。
- 记录位于 [readability-refinement/validation.json](outputs/readability-refinement/validation.json)，旧正确性验收与审计证据保留为历史快照。本轮没有执行全仓测试。

## 2026-10-20 — 恢复数据审计、精确剪枝保存与导出门槛

- 按用户“继续”的要求，沿后训练→压缩→回归→导出检查实际缺口，增量修改 5 个生产文件，保留此前未提交工作。主流程保持顺序清楚，不新增算法、配置开关或框架层。
- Recovery 与 QAT 复用统一切分审计，同时比较准备后的完整输入与有效提示；修复 padding 标签或分词归一化使相同提示漏检的情况。拒绝发生在优化及 fake-quant 替换前；成功报告增加 `data.split_audit`，不保存原始提示。
- ModelPruner 保存使用当前 `orig × mask`，修复优化后立即保存遗漏最后一次更新的问题。持久化精确 `pruning_masks.pt` 和参数清单，加载、二次剪枝、切换目标和同目录重存均保留并刷新掩码；天然零值仍可训练。Recovery/QAT 对已声明但缺失、不完整或与实体权重不一致的掩码拒绝，旧产物未声明精确掩码时仍允许注明来源的零值推断。
- 普通 metric gate 与 regression suite 均强制核对导出来源内容；adapter 评测与默认导出指向实际适配器。导出先在临时目录构建，核对复制内容、复查来源/adapter，再发布；失败保留旧产物，回滚失败保留备份并报告位置。正常合并、序列化、tokenizer 兼容转换及可选未绑定 gate 的既有语义保留。
- 两组互不重叠的 18 文件回归为 **225 passed**（59.64 秒）与 **195 passed、3 skipped**（80.51 秒）；三项真实 PTQ 在既有独立环境补测 **3 passed、0 skipped**（55.59 秒），核对 testcase ID 恰好覆盖跳过项，共验证 423 个不同用例。专项结果有重叠，不另行累加。新增用例在上一轮 wheel 上复现 **8 项失败**，当前源码中全部通过。
- 与上一轮 wheel 比较，541 项公开签名及带注解字段结构保持一致。新 2.42 wheel 构建、内容契约、5 个源码文件字节核验及无 torch/transformers 的隔离导入通过；仓库必检 Ruff、定向 Ruff、函数索引、版本元数据和 diff 空白检查通过。未执行全仓测试、生产能力评测或新的 GPU 性能验收。
- 更新 [整改说明](docs/CORRECTNESS_HARDENING_20261019.md)、恢复/量化使用说明及函数索引；原始 JUnit、前后复现和 wheel 证据汇总于 [compression-contracts-20261020/validation.json](outputs/compression-contracts-20261020/validation.json)。
