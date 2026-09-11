"""YAML/JSON-driven coordinator for the complete LLM training pipeline."""

import os
import json
import logging
import time
from typing import List, Optional, Literal, Dict, Any, Mapping
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Configuration dataclasses
# ============================================================


@dataclass
class DataSourceConfig:
    type: str = "local"  # local, huggingface, wikitext, wikipedia, c4
    path: str = ""
    name: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    format: str = "auto"
    pattern: str = "**/*.jsonl"
    streaming: bool = True
    lang: str = "zh"
    date: str = "20240301"


@dataclass
class TokenizerTrainingConfig:
    algorithm: str = "bpe"
    vocab_size: int = 32000
    min_frequency: int = 2
    max_token_length: int = 128
    byte_level: bool = True
    chinese_char_coverage: float = 0.995
    limit_gb: Optional[float] = None
    output_dir: str = "./tokenizer"
    eval_max_samples: int = 1000
    eval_domain_terms: List[str] = field(default_factory=list)


@dataclass
class DataConfig:
    sources: List[DataSourceConfig] = field(default_factory=list)
    source_weights: Optional[List[float]] = None
    max_seq_length: int = 2048
    min_text_length: int = 50
    dedup_method: str = "simhash"
    dedup_threshold: float = 0.8
    lang_filter: Optional[str] = None
    quality_min_score: float = 0.3
    num_proc: int = 4
    pack_sequences: bool = True
    output_dir: str = "./processed_data"


@dataclass
class TrainingHyperparams:
    max_steps: int = 100000
    per_device_batch_size: int = 4
    global_batch_size: int = 512
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    optimizer: str = "adamw_torch"
    gradient_checkpointing: bool = True
    save_every_steps: int = 5000
    eval_every_steps: int = 1000
    log_every_steps: int = 50
    keep_last_n_checkpoints: int = 5
    resume_from_checkpoint: Any = False
    dataloader_num_workers: int = 4
    stability_monitor: bool = True
    stability_loss_window: int = 50
    stability_spike_threshold: float = 2.5
    stability_stagnation_window: int = 200
    stability_output_dir: str = ""
    dry_run: bool = False
    preflight_only: bool = False


@dataclass
class DistributedTrainingConfig:
    strategy: Literal["single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"] = (
        "single"
    )
    num_gpus: int = 1
    num_nodes: int = 1
    gradient_accumulation_steps: int = 1
    bf16: bool = True
    fp16: bool = False
    zero_offload_to_cpu: bool = False
    deepspeed_config_path: Optional[str] = None
    ddp_backend: str = "auto"


@dataclass
class LoggingConfig:
    output_dir: str = "./outputs"
    experiment_name: str = "experiment"
    run_name: Optional[str] = None
    logging_backend: str = "tensorboard"  # tensorboard, wandb, local
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"
    save_total_limit: int = 5


@dataclass
class PipelineExecutionConfig:
    """Control-plane settings for dependency ordering and run recovery.

    ``dependencies=None`` preserves the legacy linear stage list.  Supplying a
    mapping enables explicit DAG semantics, where omitted stages are roots.
    ``resume`` restores only stages that completed under the same normalized
    configuration and graph.  ``rerun`` also invalidates every descendant.
    """

    dependencies: Optional[Dict[str, List[str]]] = None
    resume: bool = False
    rerun: List[str] = field(default_factory=list)
    state_path: Optional[str] = None


@dataclass
class SFTConfig:
    enabled: bool = False
    data_path: str = ""
    data_format: str = "jsonl"
    use_lora: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    epochs: int = 3
    learning_rate: float = 2e-4
    per_device_batch_size: int = 8
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 200
    eval_steps: int = 0
    logging_steps: int = 10
    validation_split: float = 0.0
    response_template: Optional[str] = "### Answer:"
    local_files_only: bool = False
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    optim: Optional[str] = None


@dataclass
class RLHFConfig:
    enabled: bool = False
    method: Literal["dpo", "ppo"] = "dpo"
    data_path: str = ""
    beta: float = 0.1  # DPO beta
    learning_rate: float = 5e-5
    epochs: int = 1
    per_device_batch_size: int = 1
    max_length: int = 2048
    max_prompt_length: int = 1024
    use_lora: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 200


@dataclass
class PreferenceConfig:
    enabled: bool = False
    method: Literal["dpo", "orpo", "kto"] = "dpo"
    data_path: str = ""
    beta: float = 0.1
    learning_rate: float = 5e-6
    epochs: int = 1
    per_device_batch_size: int = 1
    max_length: int = 2048
    max_prompt_length: int = 1024
    use_lora: bool = True
    use_qlora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    save_steps: int = 200
    logging_steps: int = 10
    validation_split: float = 0.0
    eval_steps: int = 0
    local_files_only: bool = False
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"


@dataclass
class MultimodalTrainingConfig:
    enabled: bool = False
    stage: str = "mllm_sft"
    data_path: str = ""
    image_root: Optional[str] = None
    image_token: str = "<image>"
    image_token_id: Optional[int] = None
    vision_backbone: str = "tiny_patch"
    projector: str = "mlp"
    image_token_strategy: str = "replace"
    freeze_vision: bool = True
    freeze_llm: bool = False
    train_projector_only: bool = True
    learning_rate: float = 2e-4
    epochs: int = 1
    per_device_batch_size: int = 1
    max_seq_length: int = 2048


@dataclass
class ImageGenerationTrainingConfig:
    enabled: bool = False
    data_path: str = ""
    output_dir: str = "./image_generation"
    codec_name: str = "external-image-codec"
    condition_encoder: str = "external-text-encoder"
    latent_channels: Optional[int] = None
    latent_height: Optional[int] = None
    latent_width: Optional[int] = None
    patch_size: int = 2
    condition_dim: Optional[int] = None
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    batch_size: int = 4
    epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    checkpoint_steps: int = 500
    device: str = "auto"
    mixed_precision: str = "no"


@dataclass
class MediaCacheStageConfig:
    enabled: bool = False
    jobs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MusicGenerationTrainingConfig:
    enabled: bool = False
    data_path: str = ""
    output_dir: str = "./music_generation"
    codec_name: str = "external-audio-codec"
    condition_encoder: str = "external-text-encoder"
    num_codebooks: Optional[int] = None
    codebook_size: Optional[int] = None
    max_sequence_length: Optional[int] = None
    condition_dim: Optional[int] = None
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    bos_token_id: int = 0
    batch_size: int = 4
    epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    checkpoint_steps: int = 500
    device: str = "auto"
    mixed_precision: str = "no"


@dataclass
class VideoGenerationTrainingConfig:
    enabled: bool = False
    data_path: str = ""
    output_dir: str = "./video_generation"
    codec_name: str = "external-causal-video-codec"
    condition_encoder: str = "external-text-encoder"
    latent_channels: Optional[int] = None
    latent_frames: Optional[int] = None
    latent_height: Optional[int] = None
    latent_width: Optional[int] = None
    temporal_patch_size: int = 1
    patch_size: int = 2
    condition_dim: Optional[int] = None
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    batch_size: int = 2
    epochs: int = 1
    max_steps: int = -1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_workers: int = 0
    checkpoint_steps: int = 500
    device: str = "auto"
    mixed_precision: str = "no"


@dataclass
class WorldModelStageConfig:
    enabled: bool = False
    config_path: Optional[str] = None
    backend: str = "rssm"
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EyeControlStageConfig:
    """Closed-loop eye-control evaluation after world-model training."""

    enabled: bool = False
    checkpoint_path: Optional[str] = None
    require_world_model: bool = True
    output_path: str = "eye_control/evaluation.json"
    episodes: int = 8
    steps: int = 150
    device: str = "auto"
    include_traces: bool = False
    safety: Dict[str, Any] = field(default_factory=dict)
    pid: Dict[str, Any] = field(default_factory=dict)
    plant: Dict[str, Any] = field(default_factory=dict)
    planner: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MOPDTrainingConfig:
    enabled: bool = False
    prompts_path: str = ""
    output_dir: str = "./mopd"
    teachers: List[Dict] = field(default_factory=list)
    num_rollouts_per_prompt: int = 1
    max_new_tokens: int = 1024
    student_temperature: float = 0.9
    teacher_temperature: float = 0.3
    aggregation: str = "best_score"
    dry_run: bool = True


@dataclass
class VLATrainingConfig:
    enabled: bool = False
    stage: str = "vla_sft"
    data_path: str = ""
    image_root: Optional[str] = None
    output_dir: str = "./vla"
    action_dim: int = 7
    action_type: str = "continuous"
    action_bins: int = 256
    action_min_value: float = -1.0
    action_max_value: float = 1.0
    include_gripper: bool = True
    control_hz: float = 10.0
    coordinate_frame: str = "end_effector_delta"
    vision_backbone: str = "tiny_patch"
    projector: str = "mlp"
    freeze_vision: bool = True
    freeze_llm: bool = False
    train_projector_only: bool = False
    train: bool = False
    use_lora: bool = True
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 1
    max_steps: int = -1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    validation_split: float = 0.0
    max_seq_length: int = 2048
    train_on_prompt: bool = False


@dataclass
class EvaluationGateConfig:
    enabled: bool = False
    rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    require_all: bool = True
    fail_on_rejection: bool = True
    output_path: Optional[str] = None


@dataclass
class EvalConfig:
    enabled: bool = True
    tasks: List[str] = field(default_factory=lambda: ["perplexity"])
    eval_dataset: str = "wikitext"
    eval_dataset_config: str = "wikitext-2-raw-v1"
    max_samples: int = 1000
    fail_on_error: bool = True
    gate: EvaluationGateConfig = field(default_factory=EvaluationGateConfig)


@dataclass
class ExportConfig:
    enabled: bool = False
    output_dir: Optional[str] = None
    format: str = "hf"
    model_path: Optional[str] = None
    merge_lora: Optional[bool] = None
    safe_serialization: bool = True
    require_gate: bool = False
    trust_remote_code: bool = False
    local_files_only: bool = False
    device: str = "auto"
    dtype: str = "auto"
    hash_weights: bool = False
    overwrite: bool = False


@dataclass
class OperatorStageConfig:
    """Adapter from visual-flow nodes into the canonical stage scheduler."""

    enabled: bool = False
    label: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    upstream_result: Dict[str, Any] = field(default_factory=dict)
    work_dir: str = ""


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    project: str = "saddlellm"
    experiment: str = "experiment"
    metadata: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    model_config: str = "gpt2-small-124m"
    model_name_or_path: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    model_backend: Literal["hf", "saddle"] = "hf"
    model_blueprint: Optional[Dict] = None
    pretrain_mode: Literal["scratch", "continue"] = "scratch"

    stages: List[str] = field(default_factory=lambda: ["tokenizer", "pretrain", "eval"])
    skip_tokenizer_training: bool = False

    data: DataConfig = field(default_factory=DataConfig)
    tokenizer_training: Optional[TokenizerTrainingConfig] = None
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    distributed: DistributedTrainingConfig = field(
        default_factory=DistributedTrainingConfig
    )
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineExecutionConfig = field(default_factory=PipelineExecutionConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    multimodal: MultimodalTrainingConfig = field(
        default_factory=MultimodalTrainingConfig
    )
    media_cache: MediaCacheStageConfig = field(default_factory=MediaCacheStageConfig)
    image_generation: ImageGenerationTrainingConfig = field(
        default_factory=ImageGenerationTrainingConfig
    )
    music_generation: MusicGenerationTrainingConfig = field(
        default_factory=MusicGenerationTrainingConfig
    )
    video_generation: VideoGenerationTrainingConfig = field(
        default_factory=VideoGenerationTrainingConfig
    )
    world_model: WorldModelStageConfig = field(default_factory=WorldModelStageConfig)
    eye_control: EyeControlStageConfig = field(default_factory=EyeControlStageConfig)
    mopd: MOPDTrainingConfig = field(default_factory=MOPDTrainingConfig)
    vla: VLATrainingConfig = field(default_factory=VLATrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    operator: OperatorStageConfig = field(default_factory=OperatorStageConfig)


# ============================================================
# Orchestrator
# ============================================================


class TrainingOrchestrator:
    """Run configured training stages in order."""

    def __init__(
        self,
        config: TrainingConfig,
        *,
        stage_registry=None,
    ):
        from ..framework import ArtifactRegistry, PipelinePlan
        from ..framework.stages import get_stage_registry

        self.config = config
        self._model = None
        self._tokenizer = None
        self._output_dir = self.config.logging.output_dir
        self._stage_results: Dict[str, Any] = {}
        self._stage_registry = stage_registry or get_stage_registry()
        self._stage_registry.discover()
        self._stage_plugins: Dict[str, Any] = {}
        self._artifact_registry = ArtifactRegistry()
        capabilities = {}
        for stage in self.config.stages:
            if self._stage_registry.contains(stage):
                capabilities[stage] = self._get_stage_plugin(stage).capabilities
        self._pipeline_plan = PipelinePlan.compile(
            self.config.stages,
            dependencies=self.config.pipeline.dependencies,
            capabilities=capabilities,
        )
        # All legacy validation and stage-to-stage handoff logic now sees the
        # deterministic topological order rather than the raw declaration order.
        self.config.stages = list(self._pipeline_plan.stages)
        self._pipeline_state_store = None
        self._restored_stages: tuple[str, ...] = ()

        self._validate_config()
        from .DistributedRuntime import current_distributed_runtime

        self._distributed_runtime = current_distributed_runtime(
            self.config.distributed.strategy
        )
        if self._distributed_runtime.is_main_process:
            os.makedirs(self._output_dir, exist_ok=True)
        self._setup_logging()

    def available_stages(self) -> tuple[str, ...]:
        """Return built-in and discovered Stage names without loading plugins."""

        return self._stage_registry.names()

    @property
    def pipeline_plan(self):
        """Return the compiled dependency graph for inspection or tooling."""

        return self._pipeline_plan

    @property
    def artifacts(self):
        """Return the live typed artifact registry for this run."""

        return self._artifact_registry

    def _get_stage_plugin(self, stage: str):
        plugin = self._stage_plugins.get(stage)
        if plugin is None:
            plugin = self._stage_registry.create(stage)
            self._stage_plugins[stage] = plugin
        return plugin

    def _stage_context(self, stage: str):
        from ..framework.stages import StageContext

        return StageContext(
            name=stage,
            config=self.config,
            output_dir=Path(self._output_dir),
            results=self._stage_results,
            distributed=getattr(self, "_distributed_runtime", None),
            artifacts=self._artifact_registry,
            metadata=self.config.metadata,
            services={
                "orchestrator": self,
                "artifacts": self._artifact_registry,
            },
        )

    def _validate_config(self):
        valid_stages = set(self.available_stages())
        unknown = [stage for stage in self.config.stages if stage not in valid_stages]
        if unknown:
            raise ValueError(
                f"Unknown training stages: {unknown}. Available stages: {sorted(valid_stages)}"
            )
        unknown_rerun = sorted(
            set(self.config.pipeline.rerun) - set(self.config.stages)
        )
        if unknown_rerun:
            raise ValueError(
                "pipeline.rerun contains stages that are not in this run: "
                + ", ".join(unknown_rerun)
            )
        for stage in self.config.stages:
            plugin = self._get_stage_plugin(stage)
            plugin.validate(self._stage_context(stage))
        if "pretrain" in self.config.stages and not self.config.data.sources:
            raise ValueError("pretrain stage requires data.sources")
        sft_can_use_mopd = (
            "mopd" in self.config.stages
            and "sft" in self.config.stages
            and self.config.stages.index("mopd") < self.config.stages.index("sft")
        )
        if (
            self.config.sft.enabled
            and not self.config.sft.data_path
            and not sft_can_use_mopd
        ):
            raise ValueError("SFT is enabled but sft.data_path is empty")
        if self.config.rlhf.enabled and not self.config.rlhf.data_path:
            raise ValueError("RLHF is enabled but rlhf.data_path is empty")
        if self.config.preference.enabled and not self.config.preference.data_path:
            raise ValueError("preference is enabled but preference.data_path is empty")
        if self.config.media_cache.enabled and not self.config.media_cache.jobs:
            raise ValueError("media_cache is enabled but media_cache.jobs is empty")
        if (
            self.config.image_generation.enabled
            and not self.config.image_generation.data_path
            and not self._has_upstream_media_cache("image", "image_generation")
        ):
            raise ValueError(
                "image_generation requires data_path or an upstream image media_cache job"
            )
        if (
            self.config.music_generation.enabled
            and not self.config.music_generation.data_path
            and not self._has_upstream_media_cache("audio", "music_generation")
        ):
            raise ValueError(
                "music_generation requires data_path or an upstream audio media_cache job"
            )
        if (
            self.config.video_generation.enabled
            and not self.config.video_generation.data_path
            and not self._has_upstream_media_cache("video", "video_generation")
        ):
            raise ValueError(
                "video_generation requires data_path or an upstream video media_cache job"
            )
        if self.config.world_model.enabled:
            if (
                not self.config.world_model.config_path
                and not self.config.world_model.data.get("train_path")
            ):
                raise ValueError(
                    "world_model requires config_path or world_model.data.train_path"
                )
        if self.config.eye_control.enabled:
            from ..spatial.prosthetic_eye_control import (
                EyePIDConfig,
                EyePlantConfig,
                EyeSafetyConfig,
            )
            from ..world_models.WorldModelInference import WorldModelPlannerConfig

            if "eye_control" not in self.config.stages:
                raise ValueError(
                    "eye_control.enabled=true requires the eye_control stage"
                )
            if self.config.eye_control.episodes <= 0:
                raise ValueError("eye_control.episodes must be positive")
            if self.config.eye_control.steps <= 0:
                raise ValueError("eye_control.steps must be positive")
            if not self.config.eye_control.output_path:
                raise ValueError("eye_control.output_path is required")
            EyeSafetyConfig.from_dict(self.config.eye_control.safety)
            EyePIDConfig.from_dict(self.config.eye_control.pid)
            EyePlantConfig.from_dict(self.config.eye_control.plant)
            WorldModelPlannerConfig.from_dict(self.config.eye_control.planner)
            if (
                self.config.eye_control.require_world_model
                and not self.config.eye_control.checkpoint_path
            ):
                if "world_model" not in self.config.stages:
                    raise ValueError(
                        "eye_control requires checkpoint_path or an upstream world_model stage"
                    )
                if self.config.stages.index("world_model") > self.config.stages.index(
                    "eye_control"
                ):
                    raise ValueError(
                        "eye_control requires world_model to execute first"
                    )
        if self.config.preference.enabled:
            if self.config.preference.method not in {"dpo", "orpo", "kto"}:
                raise ValueError(
                    f"Unsupported preference.method: {self.config.preference.method}"
                )
            preference_actual_batch_size = (
                self.config.preference.per_device_batch_size
                * max(1, self.config.distributed.num_gpus)
            )
            if (
                self.config.preference.method == "kto"
                and preference_actual_batch_size <= 1
            ):
                raise ValueError(
                    "KTO preference training requires actual train batch size > 1; "
                    "increase preference.per_device_batch_size or distributed.num_gpus."
                )
        if self.config.distributed.bf16 and self.config.distributed.fp16:
            raise ValueError(
                "distributed.bf16 and distributed.fp16 cannot both be enabled"
            )
        if self.config.distributed.strategy not in {
            "single",
            "ddp",
            "fsdp",
            "deepspeed_zero2",
            "deepspeed_zero3",
        }:
            raise ValueError(
                f"Unsupported distributed.strategy: {self.config.distributed.strategy!r}"
            )
        if (
            self.config.distributed.num_gpus < 1
            or self.config.distributed.num_nodes < 1
        ):
            raise ValueError(
                "distributed.num_gpus and distributed.num_nodes must be >= 1"
            )
        from .DistributedRuntime import validate_distributed_runtime

        if os.environ.get("WORLD_SIZE"):
            validate_distributed_runtime(
                self.config.distributed.strategy,
                self.config.distributed.num_gpus,
                self.config.distributed.num_nodes,
                require_initialized=False,
            )
        if self.config.distributed.strategy != "single":
            unsupported_distributed_stages = {
                stage
                for stage in self.config.stages
                if not self._get_stage_plugin(stage).capabilities.supports_strategy(
                    self.config.distributed.strategy
                )
            }
            if unsupported_distributed_stages:
                raise ValueError(
                    f"Distributed strategy {self.config.distributed.strategy!r} is not "
                    "supported by these stages: "
                    + ", ".join(sorted(unsupported_distributed_stages))
                )
            if (
                "pretrain" in self.config.stages
                and self.config.model_backend != "saddle"
            ):
                raise ValueError(
                    "This distributed pretraining path is verified for model.backend='saddle' only"
                )
        if "operator" in self.config.stages:
            if not self.config.operator.enabled or not self.config.operator.label:
                raise ValueError(
                    "operator stage requires operator.enabled=true and operator.label"
                )
        if self.config.model_blueprint and self.config.model_backend != "saddle":
            raise ValueError(
                "model.blueprint requires model.backend='saddle'; the HF backend "
                "does not consume SaddleLLM blueprints"
            )
        if self.config.model_blueprint:
            # Normalize and validate the complete architecture during
            # preflight, before backend/stage compatibility checks or any
            # data loading and training allocation.
            self._resolve_model_blueprint()
        from ..models.ModelAdapter import get_model_adapter

        adapter = get_model_adapter(self.config.model_backend)
        adapter_controlled_stages = {
            stage
            for stage in self.config.stages
            if self._get_stage_plugin(stage).capabilities.uses_model_adapter
        }
        unsupported_adapter_stages = adapter_controlled_stages - set(
            adapter.capabilities.stages
        )
        if unsupported_adapter_stages:
            raise ValueError(
                f"Model backend {self.config.model_backend!r} does not support these "
                "configured stages: " + ", ".join(sorted(unsupported_adapter_stages))
            )
        if "sft" in adapter_controlled_stages:
            adapter.validate_stage(
                "sft",
                use_lora=self.config.sft.use_lora,
                use_qlora=self.config.sft.use_qlora,
            )
        if "preference" in adapter_controlled_stages:
            adapter.validate_stage(
                "preference",
                method=self.config.preference.method,
                use_lora=self.config.preference.use_lora,
                use_qlora=self.config.preference.use_qlora,
            )
        if "rlhf" in adapter_controlled_stages:
            adapter.validate_stage(
                "rlhf",
                method=self.config.rlhf.method,
                use_lora=self.config.rlhf.use_lora,
                use_qlora=self.config.rlhf.use_qlora,
            )
        if self.config.model_backend == "saddle":
            if (
                self.config.pretrain_mode == "continue"
                and self.config.training.resume_from_checkpoint
            ):
                raise ValueError(
                    "pretrain_mode='continue' is a weights-only new run and cannot be combined "
                    "with training.resume_from_checkpoint (exact Trainer resume)"
                )
            if self.config.pretrain_mode == "continue":
                if not self.config.model_name_or_path:
                    raise ValueError(
                        "Saddle continue pretraining requires model.name_or_path; "
                        "use training.resume_from_checkpoint for exact Trainer resume"
                    )
                from ..models.ModelLoader import is_saddle_checkpoint

                continue_path = Path(self.config.model_name_or_path).expanduser()
                if continue_path.exists() and not is_saddle_checkpoint(continue_path):
                    raise ValueError(
                        "Saddle continue pretraining requires a complete native checkpoint "
                        "containing saddle_config.json and pytorch_model.bin"
                    )
                if not continue_path.exists():
                    raise FileNotFoundError(
                        "Saddle continue pretraining currently requires a local native "
                        f"checkpoint directory: {continue_path}"
                    )
            if (
                self.config.training.resume_from_checkpoint
                and "pretrain" not in self.config.stages
            ):
                raise ValueError(
                    "training.resume_from_checkpoint applies only to the pretrain stage"
                )
            if self.config.training.resume_from_checkpoint:
                # Fail during configuration/preflight rather than after data
                # collection and model allocation.
                self._resolve_exact_resume_checkpoint()
            if "export" in self.config.stages and self.config.export.enabled:
                export_format = str(self.config.export.format).lower()
                if export_format == "hf":
                    raise ValueError(
                        "Native Saddle checkpoints cannot be losslessly exported as HF format; "
                        "set export.format='saddle'"
                    )
                if self.config.export.merge_lora is True:
                    raise ValueError(
                        "Native Saddle export does not use LoRA merging; set export.merge_lora=false"
                    )
        self._validate_architecture_support()

    def _has_upstream_media_cache(self, modality: str, consumer: str) -> bool:
        stages = self.config.stages
        if (
            not self.config.media_cache.enabled
            or "media_cache" not in stages
            or consumer not in stages
        ):
            return False
        if stages.index("media_cache") >= stages.index(consumer):
            return False
        return any(
            isinstance(job, dict) and str(job.get("modality", "")).lower() == modality
            for job in self.config.media_cache.jobs
        )

    def _resolve_upstream_media_cache(self, modality: str) -> str:
        result = self._stage_results.get("media_cache", {})
        path = result.get("by_modality", {}).get(modality)
        if not path:
            raise ValueError(
                f"No completed upstream media_cache output for modality {modality!r}"
            )
        return path

    def _validate_architecture_support(self):
        from ..models.Architecture import ArchitectureRegistry
        from ..models.ModelRegistry import MODEL_SPECS

        if "pretrain" in self.config.stages:
            if self.config.pretrain_mode == "scratch":
                spec = MODEL_SPECS.get(self.config.model_config)
                if spec is not None:
                    ArchitectureRegistry.require(spec.architecture, "pretrain")
            elif self.config.model_name_or_path:
                support = ArchitectureRegistry.detect_from_model_name(
                    self.config.model_name_or_path
                )
                if not support.supports("continue_pretrain"):
                    raise ValueError(
                        f"Model architecture does not support continued pretraining: {support.name} ({support.notes})"
                    )

        model_hint = self.config.model_name_or_path
        if model_hint:
            support = ArchitectureRegistry.detect_from_model_name(model_hint)
            stage_capabilities = {
                "sft": "sft",
                "preference": "preference",
                "rlhf": "preference",
            }
            for stage, capability in stage_capabilities.items():
                if stage in self.config.stages and not support.supports(capability):
                    raise ValueError(
                        f"Model architecture {support.name} does not support stage {stage}. "
                        f"support_level={support.support_level}; notes={support.notes}"
                    )

    def _default_blueprint_for_spec(self, spec):
        if spec is None:
            return None
        from ..models.ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
            ObjectiveBlueprint,
        )

        attention_kind = (
            "mla" if spec.kv_lora_rank > 0 else ("gqa" if spec.num_kv_heads else "mha")
        )
        return ModelBlueprint(
            name=spec.name,
            family=spec.architecture,
            hidden_size=spec.hidden_size,
            num_layers=spec.num_hidden_layers,
            vocab_size=spec.vocab_size,
            max_position_embeddings=spec.max_position_embeddings,
            norm_eps=spec.norm_eps,
            tie_word_embeddings=spec.tie_word_embeddings,
            attention=AttentionBlueprint(
                kind=attention_kind,
                num_heads=spec.num_attention_heads,
                num_kv_heads=spec.num_kv_heads or spec.num_attention_heads,
                q_lora_rank=spec.q_lora_rank,
                kv_lora_rank=spec.kv_lora_rank,
                rope_theta=spec.rope_theta,
                sliding_window=spec.sliding_window,
            ),
            ffn=FFNBlueprint(
                kind="moe" if spec.num_experts > 0 else "swiglu",
                intermediate_size=spec.intermediate_size,
                num_experts=spec.num_experts,
                experts_per_token=spec.num_experts_per_tok,
                expert_intermediate_size=spec.expert_intermediate_size,
                shared_expert=spec.shared_expert,
                aux_loss_free=spec.router_aux_loss_coef == 0 and spec.num_experts > 0,
                router_aux_loss_coef=spec.router_aux_loss_coef,
            ),
            objective=ObjectiveBlueprint(
                multi_token_prediction=spec.multi_token_prediction,
                mtp_extra_tokens=spec.num_mtp_layers,
            ),
        )

    def _resolve_model_blueprint(self, spec=None):
        from ..models.ModelBlueprint import ModelBlueprint
        from ..models.ModelRegistry import MODEL_SPECS

        if spec is None:
            spec = MODEL_SPECS.get(self.config.model_config)
        if spec is None:
            raw = self.config.model_blueprint or {}
            required = {
                "name",
                "hidden_size",
                "vocab_size",
                "max_position_embeddings",
                "attention",
                "ffn",
            }
            if not raw.get("layers"):
                required.add("num_layers")
            missing = sorted(key for key in required if key not in raw)
            if missing:
                raise ValueError(
                    "A blueprint without a known model.config must explicitly "
                    "define: " + ", ".join(missing)
                )
        defaults = self._default_blueprint_for_spec(spec)
        return ModelBlueprint.from_dict(self.config.model_blueprint, defaults=defaults)

    def _create_saddle_model(self, spec):
        if not self.config.model_blueprint:
            from ..models.ModelRegistry import ModelRegistry

            if spec is None:
                raise KeyError(
                    "A known model.config or a self-contained model.blueprint is required"
                )
            logger.info(f"Using SaddleLLM modular model: {spec.name}")
            return ModelRegistry.create_saddle_model(spec)

        blueprint = self._resolve_model_blueprint(spec)
        logger.info(f"Using SaddleLLM blueprint model: {blueprint.name}")
        return blueprint.build_model()

    def _resolve_exact_resume_checkpoint(self) -> Optional[str]:
        """Resolve and validate an exact Trainer checkpoint for native resume."""
        resume = self.config.training.resume_from_checkpoint
        if not resume or self.config.model_backend != "saddle":
            return None

        checkpoint_path: Optional[str]
        if resume is True:
            from transformers.trainer_utils import get_last_checkpoint

            checkpoint_root = os.path.join(self._output_dir, "checkpoints")
            checkpoint_path = (
                get_last_checkpoint(checkpoint_root)
                if os.path.isdir(checkpoint_root)
                else None
            )
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"resume_from_checkpoint=true but no checkpoint-* exists in {checkpoint_root}"
                )
        elif isinstance(resume, (str, os.PathLike)):
            checkpoint_path = os.path.abspath(os.fspath(resume))
        else:
            raise TypeError(
                "training.resume_from_checkpoint must be false, true, or a checkpoint path"
            )

        strategy = self.config.distributed.strategy
        if strategy not in {"single", "ddp"}:
            raise ValueError(
                "Native exact resume is currently verified for distributed.strategy "
                f"'single' and 'ddp'; got {strategy!r}. FSDP/DeepSpeed use sharded engine checkpoints."
            )
        required = {
            "saddle_config.json",
            "pytorch_model.bin",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
        }
        missing = sorted(
            name
            for name in required
            if not os.path.isfile(os.path.join(checkpoint_path, name))
        )
        expected_world_size = max(
            1,
            int(self.config.distributed.num_gpus)
            * int(self.config.distributed.num_nodes),
        )
        if expected_world_size == 1:
            if not os.path.isfile(os.path.join(checkpoint_path, "rng_state.pth")):
                missing.append("rng_state.pth")
        else:
            missing.extend(
                f"rng_state_{rank}.pth"
                for rank in range(expected_world_size)
                if not os.path.isfile(
                    os.path.join(checkpoint_path, f"rng_state_{rank}.pth")
                )
            )
        manifest_path = os.path.join(checkpoint_path, "saddle_checkpoint.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            topology = manifest.get("parallelism") or {}
            saved_world_size = topology.get("world_size")
            if (
                saved_world_size is not None
                and int(saved_world_size) != expected_world_size
            ):
                raise ValueError(
                    "Exact resume world-size mismatch: checkpoint was saved with "
                    f"world_size={saved_world_size}, configured world_size={expected_world_size}"
                )
        if missing:
            raise ValueError(
                "Exact resume requires a Trainer checkpoint-* directory, not a weights-only "
                f"model directory; missing in {checkpoint_path}: {', '.join(missing)}"
            )
        self.config.training.resume_from_checkpoint = checkpoint_path
        return checkpoint_path

    @classmethod
    def from_yaml(
        cls,
        path: str,
        *,
        stage_registry=None,
    ) -> "TrainingOrchestrator":
        """Create an orchestrator from a YAML config file."""
        with open(path, "r", encoding="utf-8") as handle:
            if path.lower().endswith(".json"):
                raw = json.load(handle)
            else:
                import yaml

                raw = yaml.safe_load(handle) or {}
        return cls(cls._parse_config(raw), stage_registry=stage_registry)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        stage_registry=None,
    ) -> "TrainingOrchestrator":
        return cls(cls._parse_config(data), stage_registry=stage_registry)

    def run(self):
        """Run all configured training stages."""
        stages = list(self._pipeline_plan.stages)
        from .DistributedRuntime import validate_distributed_runtime

        self._distributed_runtime = validate_distributed_runtime(
            self.config.distributed.strategy,
            self.config.distributed.num_gpus,
            self.config.distributed.num_nodes,
            require_initialized=self.config.distributed.strategy != "single",
        )
        logger.info(f"=== Starting training pipeline: {' -> '.join(stages)} ===")
        start_time = time.time()
        active_stage: Optional[str] = None
        state_prepared = False
        try:
            self._prepare_pipeline_state()
            state_prepared = True
            for active_stage in stages:
                if self._pipeline_state_store.should_skip(active_stage):
                    logger.info(
                        "Restored completed stage from pipeline state: %s", active_stage
                    )
                    continue
                self._pipeline_state_store.begin(
                    active_stage,
                    write=self._distributed_runtime.is_main_process,
                )
                self.run_stage(active_stage)
                stage_result = self._stage_results.get(active_stage, {})
                self._artifact_registry.capture(active_stage, stage_result)
                self._pipeline_state_store.finish(
                    active_stage,
                    stage_result,
                    artifacts=self._artifact_registry.to_dict(),
                    write=self._distributed_runtime.is_main_process,
                )
        except Exception as exc:
            elapsed = time.time() - start_time
            if state_prepared:
                self._pipeline_state_store.fail(
                    active_stage,
                    exc,
                    write=self._distributed_runtime.is_main_process,
                )
            self._stage_results["pipeline"] = {
                "status": "failed",
                "failed_stage": active_stage,
                "error": str(exc),
                "state_path": str(self._pipeline_state_path()),
            }
            logger.error(
                "=== Training pipeline failed during %s after %.1f seconds: %s ===",
                active_stage,
                elapsed,
                exc,
            )
            try:
                self._print_summary(elapsed_seconds=elapsed)
            finally:
                self._close_logging_handlers()
            raise

        elapsed = time.time() - start_time
        self._pipeline_state_store.complete(
            artifacts=self._artifact_registry.to_dict(),
            write=self._distributed_runtime.is_main_process,
        )
        logger.info(
            f"=== Training pipeline completed in {elapsed / 3600:.1f} hours ==="
        )
        try:
            self._print_summary(elapsed_seconds=elapsed)
        finally:
            self._close_logging_handlers()
        return self._stage_results

    def _pipeline_state_path(self) -> Path:
        requested = self.config.pipeline.state_path
        if not requested:
            return Path(self._output_dir) / "pipeline_state.json"
        path = Path(requested).expanduser()
        return path if path.is_absolute() else Path(self._output_dir) / path

    def _prepare_pipeline_state(self) -> None:
        from ..framework import PipelineStateStore, normalized_run_fingerprint

        store = PipelineStateStore(
            self._pipeline_state_path(),
            run_fingerprint=normalized_run_fingerprint(self.config),
            plan=self._pipeline_plan,
        )
        restored = store.prepare(
            resume=self.config.pipeline.resume,
            rerun=self.config.pipeline.rerun,
            write=self._distributed_runtime.is_main_process,
        )
        self._pipeline_state_store = store
        self._restored_stages = restored
        self._stage_results.update(store.restored_results())
        self._artifact_registry.load_dict(store.restored_artifacts())

    @classmethod
    def run_operator(
        cls,
        label: str,
        config: Optional[Dict[str, Any]],
        upstream_result: Optional[Dict[str, Any]],
        work_dir: str,
    ) -> Dict[str, Any]:
        """Run one visual operator through the normal stage scheduler.

        ``ai-node`` owns graph ordering, while SaddleLLM remains authoritative
        for stage validation, logging, summaries and the actual LLM backend.
        """

        operator_dir = os.path.abspath(work_dir)
        training_config = TrainingConfig(
            project="ai-node",
            experiment="visual-%s" % label.lower(),
            stages=["operator"],
            logging=LoggingConfig(
                output_dir=operator_dir,
                experiment_name="visual-%s" % label.lower(),
                logging_backend="local",
            ),
            eval=EvalConfig(enabled=False),
            operator=OperatorStageConfig(
                enabled=True,
                label=label,
                config=dict(config or {}),
                upstream_result=dict(upstream_result or {}),
                work_dir=operator_dir,
            ),
        )
        orchestrator = cls(training_config)
        try:
            results = orchestrator.run()
        except Exception:
            orchestrator._close_logging_handlers()
            structured_error = orchestrator._stage_results.get("operator")
            if isinstance(structured_error, dict):
                return structured_error
            raise
        return results["operator"]

    def run_stage(self, stage: str):
        """Run a single training stage."""
        plugin = self._get_stage_plugin(stage)
        strategy = self.config.distributed.strategy
        if (
            getattr(self, "_distributed_runtime", None) is not None
            and self._distributed_runtime.is_distributed
            and not plugin.capabilities.supports_strategy(strategy)
        ):
            raise ValueError(
                f"Stage {stage!r} does not support distributed strategy {strategy!r}"
            )
        logger.info(f"{'=' * 60}")
        logger.info(f"Stage: {stage.upper()}")
        logger.info(f"{'=' * 60}")

        context = self._stage_context(stage)
        plugin.validate(context)
        result = plugin.run(context)
        if result is not None:
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Stage {stage!r} returned {type(result).__name__}; expected a mapping or None"
                )
            self._stage_results[stage] = dict(result)

    # ========================================
    # Stage implementations
    # ========================================

    def _run_media_cache_stage(self):
        """Build resumable raw-media caches for downstream generation stages."""

        cfg = self.config.media_cache
        if not cfg.enabled:
            self._stage_results["media_cache"] = {
                "status": "skipped",
                "reason": "media_cache.enabled is false",
            }
            return
        from ..multimodal.MediaCache import MediaCacheBuildConfig, build_media_cache

        results = []
        by_modality: Dict[str, str] = {}
        dry_run = self.config.training.dry_run or self.config.training.preflight_only
        seen_modalities = set()
        for index, raw_job in enumerate(cfg.jobs):
            if not isinstance(raw_job, dict):
                raise TypeError(f"media_cache.jobs[{index}] must be a mapping")
            job = dict(raw_job)
            output_path = str(job.get("output_dir", f"media_cache/job-{index}"))
            if not os.path.isabs(output_path):
                output_path = os.path.join(self._output_dir, output_path)
            job["output_dir"] = output_path
            parsed = MediaCacheBuildConfig(**job)
            if parsed.modality in seen_modalities:
                raise ValueError(
                    f"media_cache contains more than one {parsed.modality!r} job; "
                    "a downstream generation stage requires one unambiguous cache"
                )
            seen_modalities.add(parsed.modality)
            if dry_run:
                if not os.path.isfile(parsed.input_path):
                    raise FileNotFoundError(
                        f"Media cache input manifest not found: {parsed.input_path}"
                    )
                result = {
                    "status": "planned",
                    "modality": parsed.modality,
                    "input_path": os.path.abspath(parsed.input_path),
                    "output_dir": os.path.abspath(parsed.output_dir),
                    "codec_name": parsed.codec_name,
                    "shard_size": parsed.shard_size,
                }
            else:
                result = build_media_cache(parsed)
            results.append(result)
            by_modality[str(result["modality"])] = output_path
        self._stage_results["media_cache"] = {
            "status": "planned" if dry_run else "completed",
            "jobs": results,
            "by_modality": by_modality,
        }

    def _run_operator_stage(self):
        """Execute a visual-flow LLM operator as a first-class stage."""

        stage = self.config.operator
        from saddle_ml.agent import (
            RESULT_SCHEMA_VERSION,
            SUPPORTED_AGENT_OPERATORS,
            run_agent_operator,
        )

        try:
            if stage.label in SUPPORTED_AGENT_OPERATORS:
                result = run_agent_operator(
                    label=stage.label,
                    config=stage.config,
                    upstream_result=stage.upstream_result,
                    work_dir=stage.work_dir or self._output_dir,
                )
            else:
                from ..agents.RealWorldOperators import run_large_model_operator

                result = run_large_model_operator(
                    label=stage.label,
                    config=stage.config,
                    upstream_result=stage.upstream_result,
                    work_dir=stage.work_dir or self._output_dir,
                )
        except Exception as exc:
            if stage.label not in SUPPORTED_AGENT_OPERATORS:
                raise
            detail = str(exc)[:2000] or exc.__class__.__name__
            result = {
                "status": "error",
                "operator": "Agent",
                "schemaVersion": RESULT_SCHEMA_VERSION,
                "message": "%s failed: %s" % (stage.label, detail),
                "response": "",
                "text_response": "",
                "outputs": [],
                "trace": [],
                "warnings": [],
                "metrics": {
                    "agent_count": 0,
                    "round_count": 0,
                    "turn_count": 0,
                    "tool_call_count": 0,
                },
                "lineage": {"sourceLabel": stage.label},
                "artifacts": {},
                "error": {"type": exc.__class__.__name__, "message": detail},
            }
            try:
                error_dir = Path(stage.work_dir or self._output_dir).resolve()
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / "agent_error.json"
                result["artifacts"]["agent_error.json"] = str(error_path)
                result["reportPath"] = str(error_path)
                error_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError as artifact_exc:
                result["warnings"].append(
                    "Could not persist Agent error artifact: %s" % artifact_exc
                )
            self._stage_results["operator"] = result
            raise RuntimeError(result["message"]) from exc
        self._stage_results["operator"] = result
        if str(result.get("status", "")).lower() in {"error", "failed", "failure"}:
            raise RuntimeError(
                result.get("message") or "%s operator failed" % stage.label
            )
        logger.info(
            "Visual operator %s completed: %s", stage.label, result.get("message", "")
        )

    def _run_tokenizer_stage(self):
        """Stage 1: train tokenizer."""
        from ..models.TokenizerTrainer import TokenizerTrainer

        tok_config = self.config.tokenizer_training
        if tok_config is None:
            logger.info(
                "No tokenizer_training config found; loading an existing tokenizer"
            )
            self._load_tokenizer()
            return

        trainer = TokenizerTrainer(
            vocab_size=tok_config.vocab_size,
            algorithm=tok_config.algorithm,
            min_frequency=tok_config.min_frequency,
            max_token_length=tok_config.max_token_length,
            byte_level=tok_config.byte_level,
            chinese_char_coverage=tok_config.chinese_char_coverage,
        )

        corpus_files = []
        for src in self.config.data.sources:
            if src.type == "local":
                corpus_files.append(src.path)

        if not corpus_files:
            corpus_files = [self.config.data.output_dir]

        logger.info(f"Training tokenizer from corpus files: {corpus_files}")
        trainer.fit(corpus_files, limit_gb=tok_config.limit_gb)

        output = tok_config.output_dir
        trainer.save(output)
        eval_result = trainer.evaluate(
            corpus_files=corpus_files,
            max_samples=tok_config.eval_max_samples,
            domain_terms=tok_config.eval_domain_terms,
            output_path=os.path.join(output, "tokenizer_eval.json"),
        )
        self._tokenizer = trainer.get_hf_tokenizer()
        self._stage_results["tokenizer"] = {
            "vocab_size": trainer.vocab_size,
            "path": output,
            "eval": eval_result.to_dict(),
        }
        logger.info(f"Tokenizer saved to {output}; vocab_size={trainer.vocab_size}")

    def _run_pretrain_stage(self):
        """Stage 2: pretrain model."""
        import torch
        from transformers import (
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from ..models.ModelRegistry import ModelRegistry, MODEL_SPECS
        from ..data.pipeline import DataPipeline, PipelineConfig
        from .DistributedConfig import DistributedConfig as DistCfg
        from .DistributedRuntime import distributed_barrier

        exact_resume_path = self._resolve_exact_resume_checkpoint()
        spec = MODEL_SPECS.get(self.config.model_config)
        if spec is None and not (
            self.config.model_backend == "saddle"
            and (
                self.config.model_blueprint
                or exact_resume_path
                or self.config.pretrain_mode == "continue"
            )
        ):
            raise KeyError(
                f"Unknown model config: {self.config.model_config}. Available: {list(MODEL_SPECS.keys())}"
            )

        if spec is not None:
            logger.info(
                f"Model spec: {spec.name} ({spec.human_params()} params, {spec.human_tokens()} target tokens)"
            )
        elif not exact_resume_path and self.config.pretrain_mode != "continue":
            blueprint = self._resolve_model_blueprint()
            logger.info(
                "Self-contained model blueprint: %s (%s estimated params)",
                blueprint.name,
                blueprint.analyze()["total_params_human"],
            )

        # 1. Create or load model
        if exact_resume_path:
            from ..models.ModelLoader import load_causal_lm

            logger.info("Exact Trainer resume from %s", exact_resume_path)
            self._model = load_causal_lm(exact_resume_path, map_location="cpu")
        elif self.config.pretrain_mode == "continue" and self.config.model_name_or_path:
            logger.info(f"Continuing pretraining from {self.config.model_name_or_path}")
            from ..models.ModelLoader import load_causal_lm

            continue_dtype = (
                torch.bfloat16
                if self.config.distributed.bf16
                else torch.float16
                if self.config.distributed.fp16
                else torch.float32
            )
            self._model = load_causal_lm(
                self.config.model_name_or_path,
                dtype=continue_dtype,
                trust_remote_code=True,
            )
        else:
            logger.info(
                "Creating model from scratch: %s",
                spec.name
                if spec is not None
                else self.config.model_blueprint.get("name"),
            )
            if self.config.model_backend == "saddle":
                self._model = self._create_saddle_model(spec)
            else:
                self._model = ModelRegistry.create_model(spec)

        total_params = sum(p.numel() for p in self._model.parameters())
        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        logger.info(
            f"Parameters: total={total_params / 1e6:.1f}M, trainable={trainable_params / 1e6:.1f}M"
        )

        # 2. Load or create tokenizer
        if exact_resume_path:
            from ..models.TokenizerLoader import load_tokenizer_compatible

            self._tokenizer = load_tokenizer_compatible(
                exact_resume_path, trust_remote_code=True
            )
        else:
            self._load_tokenizer()

        # 3. Prepare data
        if self._distributed_runtime.is_distributed and any(
            source.streaming for source in self.config.data.sources
        ):
            raise ValueError(
                "Distributed pretraining requires materialized data sources for now; "
                "set every data source streaming=false so preprocessing is deterministic"
            )
        pipe_cfg = PipelineConfig(
            sources=[
                {
                    "type": s.type,
                    "path": s.path,
                    "name": s.name,
                    "split": s.split,
                    "text_column": s.text_column,
                    "format": s.format,
                    "pattern": s.pattern,
                    "streaming": s.streaming,
                    "lang": s.lang,
                    "date": s.date,
                }
                for s in self.config.data.sources
            ],
            source_weights=self.config.data.source_weights,
            output_dir=self.config.data.output_dir,
            max_seq_length=self.config.data.max_seq_length,
            min_text_length=self.config.data.min_text_length,
            dedup_method=self.config.data.dedup_method,
            dedup_threshold=self.config.data.dedup_threshold,
            lang_filter=self.config.data.lang_filter,
            num_proc=self.config.data.num_proc,
            pack_sequences=self.config.data.pack_sequences,
        )

        pipeline = DataPipeline(pipe_cfg)
        pipeline.collect().clean().deduplicate().filter_quality()
        pipeline.tokenize_and_pack(self._tokenizer)
        dataset = pipeline.to_iterable_dataset()

        # 4. Build training arguments
        grad_accum = self.config.distributed.gradient_accumulation_steps or max(
            1,
            self.config.training.global_batch_size
            // (
                self.config.training.per_device_batch_size
                * max(1, self.config.distributed.num_gpus)
            ),
        )
        dist_cfg = DistCfg(
            strategy=self.config.distributed.strategy,
            num_gpus=self.config.distributed.num_gpus,
            num_nodes=self.config.distributed.num_nodes,
            gradient_accumulation_steps=grad_accum,
            bf16=self.config.distributed.bf16,
            fp16=self.config.distributed.fp16,
            zero_offload_to_cpu=self.config.distributed.zero_offload_to_cpu,
            deepspeed_config_path=self.config.distributed.deepspeed_config_path,
            ddp_backend=self.config.distributed.ddp_backend,
        )
        dist_args = dist_cfg.to_training_args()
        for duplicate_key in (
            "gradient_accumulation_steps",
            "bf16",
            "fp16",
            "dataloader_num_workers",
        ):
            dist_args.pop(duplicate_key, None)

        training_args = TrainingArguments(
            output_dir=os.path.join(self._output_dir, "checkpoints"),
            per_device_train_batch_size=self.config.training.per_device_batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=self.config.training.learning_rate,
            lr_scheduler_type=self.config.training.lr_scheduler,
            warmup_steps=self.config.training.warmup_steps,
            max_steps=self.config.training.max_steps,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            optim=self.config.training.optimizer,
            logging_steps=self.config.training.log_every_steps,
            save_steps=self.config.training.save_every_steps,
            eval_steps=self.config.training.eval_every_steps,
            save_total_limit=self.config.training.keep_last_n_checkpoints,
            bf16=self.config.distributed.bf16,
            fp16=self.config.distributed.fp16,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            dataloader_num_workers=self.config.training.dataloader_num_workers,
            seed=self.config.seed,
            report_to=[self.config.logging.logging_backend]
            if self.config.logging.logging_backend != "local"
            else ["none"],
            run_name=self.config.logging.run_name or self.config.experiment,
            **dist_args,
        )

        # 5. Train
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, mlm=False
        )
        callbacks = []
        if self.config.training.stability_monitor:
            from ..experiments.PretrainStability import PretrainStabilityCallback, StabilityConfig

            stability_dir = self.config.training.stability_output_dir or os.path.join(
                self._output_dir, "stability"
            )
            callbacks.append(
                PretrainStabilityCallback(
                    StabilityConfig(
                        enabled=True,
                        output_dir=stability_dir,
                        loss_window=self.config.training.stability_loss_window,
                        spike_threshold=self.config.training.stability_spike_threshold,
                        stagnation_window=self.config.training.stability_stagnation_window,
                    )
                )
            )
        if self.config.model_backend == "saddle":
            from ..models.ModelMetrics import ModelMetricsCallback

            callbacks.append(
                ModelMetricsCallback(
                    output_dir=os.path.join(self._output_dir, "model_metrics"),
                    log_every=max(1, self.config.training.log_every_steps),
                )
            )

        trainer_class = Trainer
        if self.config.model_backend == "saddle":
            from .NativeTrainer import SaddleTrainer

            trainer_class = SaddleTrainer
        trainer = trainer_class(
            model=self._model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        logger.info(
            f"Starting pretraining: {self.config.training.max_steps} steps, "
            f"batch_size={self.config.training.per_device_batch_size}, "
            f"grad_accum={grad_accum}"
        )

        trainer.train(
            resume_from_checkpoint=self.config.training.resume_from_checkpoint
        )

        # 6. Save
        final_path = os.path.join(self._output_dir, "final_model")
        trainer.save_model(final_path)
        if trainer.is_world_process_zero():
            self._tokenizer.save_pretrained(final_path)
            trainer.save_state()
        distributed_barrier()
        logger.info(f"Model saved to {final_path}")

        self._stage_results["pretrain"] = {
            "model_path": final_path,
            "steps": self.config.training.max_steps,
            "global_step": int(trainer.state.global_step),
            "continuation": "weights_only"
            if self.config.pretrain_mode == "continue"
            else "scratch_or_exact_resume",
            "resume_from_checkpoint": self.config.training.resume_from_checkpoint
            or None,
            "total_params": total_params,
            "estimated_params": spec.estimated_params
            if spec is not None
            else total_params,
        }
        distributed_barrier()

    def _inspect_post_training_data(self, path: str, task: str) -> Dict:
        from ..data.TrainingDataInspector import TrainingDataInspector

        report = TrainingDataInspector.inspect_file(path, task=task)
        if not report.ready:
            self._close_logging_handlers()
            raise ValueError(
                f"{task} data inspection failed: {'; '.join(report.errors)}"
            )
        return report.to_dict()

    def _safe_dataclass_dict(self, value: Any) -> Dict:
        from dataclasses import asdict, is_dataclass

        if is_dataclass(value):
            return asdict(value)
        return dict(value) if isinstance(value, dict) else {}

    def _write_stage_plan(self, stage: str, payload: Dict) -> Dict:
        stage_dir = os.path.join(self._output_dir, "plans")
        os.makedirs(stage_dir, exist_ok=True)
        path = os.path.join(stage_dir, f"{stage}_plan.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"path": path, "payload": payload}

    def _estimate_stage_training(
        self,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        max_seq_length: int,
        epochs: Optional[int] = None,
    ) -> Dict:
        from .TrainingPlanEstimator import TrainingPlanEstimator

        return TrainingPlanEstimator.estimate(
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_gpus=self.config.distributed.num_gpus,
            max_seq_length=max_seq_length,
            max_steps=self.config.training.max_steps,
            epochs=epochs,
        ).to_dict()

    def _run_sft_stage(self):
        """Stage 3: supervised fine-tuning."""
        if not self.config.sft.enabled:
            logger.info("SFT is disabled; skipping")
            return

        logger.info("Starting SFT stage")

        model_path = (
            self._stage_results.get("pretrain", {}).get("model_path")
            or self.config.model_name_or_path
        )
        if not model_path:
            raise ValueError(
                "SFT requires model.name_or_path or an upstream pretrain output"
            )

        sft_cfg = self.config.sft
        dataset_path = sft_cfg.data_path or self._stage_results.get("mopd", {}).get(
            "data_path"
        )
        if not dataset_path:
            raise ValueError(
                "SFT requires sft.data_path or a completed upstream MOPD stage with data_path"
            )
        data_inspection = self._inspect_post_training_data(dataset_path, task="sft")
        output_path = os.path.join(self._output_dir, "sft_checkpoints")
        plan = self._write_stage_plan(
            "sft",
            {
                "stage": "sft",
                "status": "planned",
                "model_path": model_path,
                "data_path": dataset_path,
                "output_path": output_path,
                "data_inspection": data_inspection,
                "training_estimate": self._estimate_stage_training(
                    per_device_batch_size=sft_cfg.per_device_batch_size,
                    gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
                    max_seq_length=sft_cfg.max_seq_length,
                    epochs=sft_cfg.epochs,
                ),
                "config": self._safe_dataclass_dict(sft_cfg),
            },
        )
        if self.config.training.dry_run or self.config.training.preflight_only:
            self._stage_results["sft"] = {
                "status": "planned",
                "model_path": model_path,
                "data_path": dataset_path,
                "plan_path": plan["path"],
                "data_inspection": data_inspection,
            }
            return
        try:
            if self.config.model_backend == "saddle":
                from .NativePostTraining import NativeSFTConfig, train_native_sft

                result = train_native_sft(
                    NativeSFTConfig(
                        model_path=model_path,
                        dataset_path=dataset_path,
                        output_path=output_path,
                        num_epochs=sft_cfg.epochs,
                        max_steps=self.config.training.max_steps,
                        learning_rate=sft_cfg.learning_rate,
                        batch_size=sft_cfg.per_device_batch_size,
                        max_length=sft_cfg.max_seq_length,
                        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
                        warmup_steps=sft_cfg.warmup_steps,
                        save_steps=sft_cfg.save_steps,
                        eval_steps=sft_cfg.eval_steps,
                        logging_steps=sft_cfg.logging_steps,
                        validation_split=sft_cfg.validation_split,
                        gradient_checkpointing=sft_cfg.gradient_checkpointing,
                        report_to=self.config.logging.logging_backend
                        if self.config.logging.logging_backend != "local"
                        else "none",
                        seed=self.config.seed,
                    )
                )
            else:
                from .PeftSFTTrainer import train_model

                result = train_model(
                    model_path=model_path,
                    dataset_path=dataset_path,
                    output_path=output_path,
                    use_lora=sft_cfg.use_lora,
                    use_qlora=sft_cfg.use_qlora,
                    lora_r=sft_cfg.lora_r,
                    lora_alpha=sft_cfg.lora_alpha,
                    lora_dropout=sft_cfg.lora_dropout,
                    lora_target_modules=sft_cfg.lora_target_modules,
                    num_epochs=sft_cfg.epochs,
                    max_steps=self.config.training.max_steps,
                    learning_rate=sft_cfg.learning_rate,
                    batch_size=sft_cfg.per_device_batch_size,
                    max_seq_length=sft_cfg.max_seq_length,
                    gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
                    warmup_steps=sft_cfg.warmup_steps,
                    save_steps=sft_cfg.save_steps,
                    eval_steps=sft_cfg.eval_steps,
                    logging_steps=sft_cfg.logging_steps,
                    validation_split=sft_cfg.validation_split,
                    response_template=sft_cfg.response_template,
                    gradient_checkpointing=sft_cfg.gradient_checkpointing,
                    optim=sft_cfg.optim,
                    local_files_only=sft_cfg.local_files_only,
                    trust_remote_code=sft_cfg.trust_remote_code,
                    report_to=self.config.logging.logging_backend
                    if self.config.logging.logging_backend != "local"
                    else "none",
                    resume_from_checkpoint=self.config.training.resume_from_checkpoint,
                )
            self._stage_results["sft"] = {
                "status": "completed",
                "model_path": output_path,
                "data_path": dataset_path,
                "plan_path": plan["path"],
                "data_inspection": data_inspection,
                "result": result,
            }
        except Exception as e:
            logger.warning(f"SFT stage failed: {e}")
            logger.info(
                "Use --debug for a full traceback from PeftSFTTrainer.train_model()"
            )

            self._stage_results["sft"] = {
                "status": "failed",
                "model_path": output_path,
                "data_path": dataset_path,
                "plan_path": plan["path"],
                "data_inspection": data_inspection,
                "error": str(e),
            }
            raise

    def _run_preference_stage(self):
        """Stage 4: preference training with DPO, ORPO, or KTO."""
        if not self.config.preference.enabled:
            logger.info("Preference training is disabled; skipping")
            return

        logger.info(
            f"Starting preference training stage ({self.config.preference.method})"
        )

        model_path = (
            self._stage_results.get("sft", {}).get("model_path")
            or self._stage_results.get("pretrain", {}).get("model_path")
            or self.config.model_name_or_path
        )
        if not model_path:
            raise ValueError(
                "Preference training requires model.name_or_path or an upstream sft/pretrain output"
            )

        pref_cfg = self.config.preference
        output_path = os.path.join(self._output_dir, "preference_checkpoints")
        data_inspection = self._inspect_post_training_data(
            pref_cfg.data_path, task=pref_cfg.method
        )
        plan = self._write_stage_plan(
            "preference",
            {
                "stage": "preference",
                "status": "planned",
                "model_path": model_path,
                "data_path": pref_cfg.data_path,
                "output_path": output_path,
                "data_inspection": data_inspection,
                "training_estimate": self._estimate_stage_training(
                    per_device_batch_size=pref_cfg.per_device_batch_size,
                    gradient_accumulation_steps=pref_cfg.gradient_accumulation_steps,
                    max_seq_length=pref_cfg.max_length,
                    epochs=pref_cfg.epochs,
                ),
                "config": self._safe_dataclass_dict(pref_cfg),
            },
        )
        if self.config.training.dry_run or self.config.training.preflight_only:
            self._stage_results["preference"] = {
                "status": "planned",
                "model_path": model_path,
                "data_path": pref_cfg.data_path,
                "plan_path": plan["path"],
                "data_inspection": data_inspection,
            }
            return
        if self.config.model_backend == "saddle":
            from .NativePostTraining import NativeDPOConfig, train_native_dpo

            result = train_native_dpo(
                NativeDPOConfig(
                    model_path=model_path,
                    dataset_path=pref_cfg.data_path,
                    output_path=output_path,
                    beta=pref_cfg.beta,
                    learning_rate=pref_cfg.learning_rate,
                    batch_size=pref_cfg.per_device_batch_size,
                    num_epochs=pref_cfg.epochs,
                    max_steps=self.config.training.max_steps,
                    max_length=pref_cfg.max_length,
                    max_prompt_length=pref_cfg.max_prompt_length,
                    gradient_accumulation_steps=pref_cfg.gradient_accumulation_steps,
                    warmup_steps=pref_cfg.warmup_steps,
                    save_steps=pref_cfg.save_steps,
                    logging_steps=pref_cfg.logging_steps,
                    validation_split=pref_cfg.validation_split,
                    eval_steps=pref_cfg.eval_steps,
                    gradient_checkpointing=pref_cfg.gradient_checkpointing,
                    report_to=pref_cfg.report_to
                    if pref_cfg.report_to != "local"
                    else "none",
                    seed=self.config.seed,
                )
            )
        else:
            from .Preference import PreferenceTrainConfig, train_preference

            result = train_preference(
                PreferenceTrainConfig(
                    model_path=model_path,
                    dataset_path=pref_cfg.data_path,
                    output_path=output_path,
                    method=pref_cfg.method,
                    beta=pref_cfg.beta,
                    learning_rate=pref_cfg.learning_rate,
                    batch_size=pref_cfg.per_device_batch_size,
                    num_epochs=pref_cfg.epochs,
                    max_steps=self.config.training.max_steps,
                    max_length=pref_cfg.max_length,
                    max_prompt_length=pref_cfg.max_prompt_length,
                    use_lora=pref_cfg.use_lora,
                    use_qlora=pref_cfg.use_qlora,
                    lora_r=pref_cfg.lora_r,
                    lora_alpha=pref_cfg.lora_alpha,
                    lora_dropout=pref_cfg.lora_dropout,
                    gradient_accumulation_steps=pref_cfg.gradient_accumulation_steps,
                    warmup_steps=pref_cfg.warmup_steps,
                    save_steps=pref_cfg.save_steps,
                    logging_steps=pref_cfg.logging_steps,
                    validation_split=pref_cfg.validation_split,
                    eval_steps=pref_cfg.eval_steps,
                    local_files_only=pref_cfg.local_files_only,
                    trust_remote_code=pref_cfg.trust_remote_code,
                    gradient_checkpointing=pref_cfg.gradient_checkpointing,
                    report_to=pref_cfg.report_to
                    if pref_cfg.report_to != "local"
                    else "none",
                    resume_from_checkpoint=self.config.training.resume_from_checkpoint,
                )
            )
        self._stage_results["preference"] = {
            "status": "completed",
            "model_path": output_path,
            "data_path": pref_cfg.data_path,
            "plan_path": plan["path"],
            "data_inspection": data_inspection,
            "result": result,
        }

    def _run_rlhf_stage(self):
        """Stage 4: RLHF alignment."""
        if not self.config.rlhf.enabled:
            logger.info("RLHF is disabled; skipping")
            return

        logger.info(f"Starting RLHF alignment stage ({self.config.rlhf.method})")
        model_path = (
            self._stage_results.get("preference", {}).get("model_path")
            or self._stage_results.get("sft", {}).get("model_path")
            or self._stage_results.get("pretrain", {}).get("model_path")
            or self.config.model_name_or_path
        )

        if not model_path:
            raise ValueError(
                "RLHF requires model_name_or_path or an upstream trained model."
            )

        rlhf_cfg = self.config.rlhf
        output_path = os.path.join(self._output_dir, "rlhf_checkpoints")
        data_inspection = self._inspect_post_training_data(
            rlhf_cfg.data_path,
            task="dpo" if rlhf_cfg.method == "dpo" else "rlhf",
        )
        plan = self._write_stage_plan(
            "rlhf",
            {
                "stage": "rlhf",
                "status": "planned",
                "method": rlhf_cfg.method,
                "model_path": model_path,
                "data_path": rlhf_cfg.data_path,
                "output_path": output_path,
                "data_inspection": data_inspection,
                "training_estimate": self._estimate_stage_training(
                    per_device_batch_size=rlhf_cfg.per_device_batch_size,
                    gradient_accumulation_steps=rlhf_cfg.gradient_accumulation_steps,
                    max_seq_length=rlhf_cfg.max_length,
                    epochs=rlhf_cfg.epochs,
                ),
                "config": self._safe_dataclass_dict(rlhf_cfg),
            },
        )
        if self.config.training.dry_run or self.config.training.preflight_only:
            self._stage_results["rlhf"] = {
                "status": "planned",
                "method": rlhf_cfg.method,
                "model_path": model_path,
                "data_path": rlhf_cfg.data_path,
                "plan_path": plan["path"],
                "data_inspection": data_inspection,
            }
            return
        if rlhf_cfg.method == "dpo":
            if self.config.model_backend == "saddle":
                from .NativePostTraining import NativeDPOConfig, train_native_dpo

                result = train_native_dpo(
                    NativeDPOConfig(
                        model_path=model_path,
                        dataset_path=rlhf_cfg.data_path,
                        output_path=output_path,
                        beta=rlhf_cfg.beta,
                        learning_rate=rlhf_cfg.learning_rate,
                        batch_size=rlhf_cfg.per_device_batch_size,
                        num_epochs=rlhf_cfg.epochs,
                        max_steps=self.config.training.max_steps,
                        max_length=rlhf_cfg.max_length,
                        max_prompt_length=rlhf_cfg.max_prompt_length,
                        gradient_accumulation_steps=rlhf_cfg.gradient_accumulation_steps,
                        warmup_steps=rlhf_cfg.warmup_steps,
                        save_steps=rlhf_cfg.save_steps,
                        seed=self.config.seed,
                    )
                )
            else:
                from .Preference import PreferenceTrainConfig, train_preference

                result = train_preference(
                    PreferenceTrainConfig(
                        model_path=model_path,
                        dataset_path=rlhf_cfg.data_path,
                        output_path=output_path,
                        method="dpo",
                        beta=rlhf_cfg.beta,
                        learning_rate=rlhf_cfg.learning_rate,
                        batch_size=rlhf_cfg.per_device_batch_size,
                        num_epochs=rlhf_cfg.epochs,
                        max_length=rlhf_cfg.max_length,
                        max_prompt_length=rlhf_cfg.max_prompt_length,
                        use_lora=rlhf_cfg.use_lora,
                        use_qlora=rlhf_cfg.use_qlora,
                        lora_r=rlhf_cfg.lora_r,
                        lora_alpha=rlhf_cfg.lora_alpha,
                        lora_dropout=rlhf_cfg.lora_dropout,
                        gradient_accumulation_steps=rlhf_cfg.gradient_accumulation_steps,
                        warmup_steps=rlhf_cfg.warmup_steps,
                        save_steps=rlhf_cfg.save_steps,
                    )
                )
            self._stage_results["rlhf"] = {
                "status": "completed",
                "model_path": output_path,
                "data_path": rlhf_cfg.data_path,
                "plan_path": plan["path"],
                "data_inspection": data_inspection,
                "result": result,
            }
            return

        self._stage_results["rlhf"] = {
            "status": "planned_unsupported",
            "method": rlhf_cfg.method,
            "data_path": rlhf_cfg.data_path,
            "plan_path": plan["path"],
            "data_inspection": data_inspection,
            "reason": "PPO requires reward-model scoring and rollout-specific TRL wiring; use preference/dpo for stable offline RLHF.",
        }
        return

    def _run_multimodal_stage(self):
        """Materialize multimodal/VLM training plans and normalized data."""
        cfg = self.config.multimodal
        output_dir = os.path.join(self._output_dir, "multimodal")
        os.makedirs(output_dir, exist_ok=True)
        plan = {
            "stage": cfg.stage,
            "status": "planned",
            "data_path": cfg.data_path,
            "image_root": cfg.image_root,
            "image_token": cfg.image_token,
            "image_token_id": cfg.image_token_id,
            "vision_backbone": cfg.vision_backbone,
            "projector": cfg.projector,
            "image_token_strategy": cfg.image_token_strategy,
            "freeze_vision": cfg.freeze_vision,
            "freeze_llm": cfg.freeze_llm,
            "train_projector_only": cfg.train_projector_only,
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "per_device_batch_size": cfg.per_device_batch_size,
            "max_seq_length": cfg.max_seq_length,
            "notes": [
                "Start with projector warmup before unfreezing the language model.",
                "Evaluate OCR, VQA, document QA, and visual hallucination separately from text perplexity.",
            ],
        }
        if cfg.data_path and os.path.exists(cfg.data_path):
            from ..multimodal.MultimodalData import MultimodalDataAdapter

            normalized_path = os.path.join(output_dir, "mllm_sft_normalized.jsonl")
            report = MultimodalDataAdapter.normalize_file(
                cfg.data_path,
                normalized_path,
                image_root=cfg.image_root,
                task="mllm_sft",
            )
            plan["normalized_data_path"] = normalized_path
            plan["normalization_report"] = report.to_dict()
        else:
            plan["status"] = "planned_no_data"
            plan["warnings"] = [
                "No multimodal data_path found; wrote planning artifacts only."
            ]
        plan_path = os.path.join(output_dir, "multimodal_training_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        self._stage_results["mllm_sft"] = {
            "status": plan["status"],
            "plan_path": plan_path,
            "normalized_data_path": plan.get("normalized_data_path"),
        }

    def _run_image_generation_stage(self):
        """Train a text-conditioned flow model from cached image latents."""

        cfg = self.config.image_generation
        if not cfg.enabled:
            self._stage_results["image_generation"] = {
                "status": "skipped",
                "reason": "image_generation.enabled is false",
            }
            return
        if (
            not cfg.data_path
            and self._stage_results.get("media_cache", {}).get("status") == "planned"
        ):
            self._stage_results["image_generation"] = {
                "status": "planned",
                "depends_on": "media_cache:image",
                "reason": "model dimensions will be inferred after cache construction",
            }
            return
        from ..multimodal.LatentFlowTrainer import CachedLatentDataset

        data_path = cfg.data_path or self._resolve_upstream_media_cache("image")
        dataset = CachedLatentDataset(data_path)
        cache_manifest = getattr(getattr(dataset, "_sharded", None), "manifest", {})
        codec_name = cache_manifest.get("codec", {}).get("name", cfg.codec_name)
        condition_encoder = cache_manifest.get("condition_encoder", {}).get(
            "name", cfg.condition_encoder
        )
        inferred = dataset.infer_model_config(
            patch_size=cfg.patch_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )
        dimensions = {
            "latent_channels": cfg.latent_channels,
            "latent_height": cfg.latent_height,
            "latent_width": cfg.latent_width,
            "condition_dim": cfg.condition_dim,
        }
        for name, configured in dimensions.items():
            if configured is not None and int(configured) != getattr(inferred, name):
                raise ValueError(
                    f"image_generation.{name}={configured} does not match cached data "
                    f"value {getattr(inferred, name)}"
                )
        output_path = cfg.output_dir
        if not os.path.isabs(output_path):
            output_path = os.path.join(self._output_dir, output_path)
        plan = self._write_stage_plan(
            "image_generation",
            {
                "stage": "image_generation",
                "status": "planned",
                "data_path": os.path.abspath(data_path),
                "output_path": os.path.abspath(output_path),
                "samples": len(dataset),
                "model_config": inferred.to_dict(),
                "codec_name": codec_name,
                "condition_encoder": condition_encoder,
                "cache_manifest": cache_manifest or None,
                "contract": (
                    "NPZ arrays latents[N,C,H,W] and conditions[N,D]; media encoding "
                    "is an upstream codec/cache stage"
                ),
            },
        )
        if self.config.training.dry_run or self.config.training.preflight_only:
            self._stage_results["image_generation"] = {
                "status": "planned",
                "plan_path": plan["path"],
                "model_config": inferred.to_dict(),
                "samples": len(dataset),
            }
            return
        from ..multimodal.LatentFlowTrainer import LatentFlowTrainingConfig, train_latent_flow

        result = train_latent_flow(
            inferred,
            LatentFlowTrainingConfig(
                data_path=data_path,
                output_dir=output_path,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                max_steps=cfg.max_steps,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                warmup_steps=cfg.warmup_steps,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                max_grad_norm=cfg.max_grad_norm,
                num_workers=cfg.num_workers,
                checkpoint_steps=cfg.checkpoint_steps,
                seed=self.config.seed,
                device=cfg.device,
                mixed_precision=cfg.mixed_precision,
                codec_name=codec_name,
                condition_encoder=condition_encoder,
            ),
        )
        self._stage_results["image_generation"] = {
            **result,
            "plan_path": plan["path"],
        }

    def _run_music_generation_stage(self):
        """Train a text-conditioned AR model from cached audio codec tokens."""

        cfg = self.config.music_generation
        if not cfg.enabled:
            self._stage_results["music_generation"] = {
                "status": "skipped",
                "reason": "music_generation.enabled is false",
            }
            return
        if (
            not cfg.data_path
            and self._stage_results.get("media_cache", {}).get("status") == "planned"
        ):
            self._stage_results["music_generation"] = {
                "status": "planned",
                "depends_on": "media_cache:audio",
                "reason": "model dimensions will be inferred after cache construction",
            }
            return
        from ..multimodal.MusicCodeModel import MusicCodeConfig
        from ..multimodal.MusicCodeTrainer import CachedMusicCodeDataset

        data_path = cfg.data_path or self._resolve_upstream_media_cache("audio")
        dataset = CachedMusicCodeDataset(data_path)
        cache_manifest = getattr(getattr(dataset, "_sharded", None), "manifest", {})
        codec_fingerprint = cache_manifest.get("codec", {})
        codec_name = codec_fingerprint.get("name", cfg.codec_name)
        condition_encoder = cache_manifest.get("condition_encoder", {}).get(
            "name", cfg.condition_encoder
        )
        inferred = dataset.infer_model_config(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            bos_token_id=cfg.bos_token_id,
        )
        if (
            cfg.num_codebooks is not None
            and int(cfg.num_codebooks) != inferred.num_codebooks
        ):
            raise ValueError(
                f"music_generation.num_codebooks={cfg.num_codebooks} does not match "
                f"cached data value {inferred.num_codebooks}"
            )
        if (
            cfg.condition_dim is not None
            and int(cfg.condition_dim) != inferred.condition_dim
        ):
            raise ValueError(
                f"music_generation.condition_dim={cfg.condition_dim} does not match "
                f"cached data value {inferred.condition_dim}"
            )
        sequence_length = (
            int(cfg.max_sequence_length)
            if cfg.max_sequence_length is not None
            else inferred.max_sequence_length
        )
        if sequence_length < inferred.max_sequence_length:
            raise ValueError(
                "music_generation.max_sequence_length cannot be smaller than cached frames"
            )
        codebook_size = (
            int(cfg.codebook_size)
            if cfg.codebook_size is not None
            else int(codec_fingerprint.get("codebook_size", inferred.codebook_size))
        )
        if codebook_size < inferred.codebook_size:
            raise ValueError(
                "music_generation.codebook_size must be greater than the largest cached code"
            )
        model_config = MusicCodeConfig(
            num_codebooks=inferred.num_codebooks,
            codebook_size=codebook_size,
            max_sequence_length=sequence_length,
            condition_dim=inferred.condition_dim,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            bos_token_id=cfg.bos_token_id,
        )
        output_path = cfg.output_dir
        if not os.path.isabs(output_path):
            output_path = os.path.join(self._output_dir, output_path)
        plan = self._write_stage_plan(
            "music_generation",
            {
                "stage": "music_generation",
                "status": "planned",
                "data_path": os.path.abspath(data_path),
                "output_path": os.path.abspath(output_path),
                "samples": len(dataset),
                "model_config": model_config.to_dict(),
                "codec_name": codec_name,
                "condition_encoder": condition_encoder,
                "cache_manifest": cache_manifest or None,
                "contract": (
                    "NPZ arrays codes[N,Q,T], conditions[N,D], and optional "
                    "attention_mask[N,T]; audio encoding is an upstream codec/cache stage"
                ),
            },
        )
        if self.config.training.dry_run or self.config.training.preflight_only:
            self._stage_results["music_generation"] = {
                "status": "planned",
                "plan_path": plan["path"],
                "model_config": model_config.to_dict(),
                "samples": len(dataset),
            }
            return
        from ..multimodal.MusicCodeTrainer import MusicCodeTrainingConfig, train_music_code

        result = train_music_code(
            model_config,
            MusicCodeTrainingConfig(
                data_path=data_path,
                output_dir=output_path,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                max_steps=cfg.max_steps,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                warmup_steps=cfg.warmup_steps,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                max_grad_norm=cfg.max_grad_norm,
                num_workers=cfg.num_workers,
                checkpoint_steps=cfg.checkpoint_steps,
                seed=self.config.seed,
                device=cfg.device,
                mixed_precision=cfg.mixed_precision,
                codec_name=codec_name,
                condition_encoder=condition_encoder,
            ),
        )
        self._stage_results["music_generation"] = {
            **result,
            "plan_path": plan["path"],
        }

    def _run_video_generation_stage(self):
        """Train a text-conditioned flow model from cached video latents."""

        cfg = self.config.video_generation
        if not cfg.enabled:
            self._stage_results["video_generation"] = {
                "status": "skipped",
                "reason": "video_generation.enabled is false",
            }
            return
        if (
            not cfg.data_path
            and self._stage_results.get("media_cache", {}).get("status") == "planned"
        ):
            self._stage_results["video_generation"] = {
                "status": "planned",
                "depends_on": "media_cache:video",
                "reason": "model dimensions will be inferred after cache construction",
            }
            return
        from ..multimodal.VideoLatentFlowTrainer import CachedVideoLatentDataset

        data_path = cfg.data_path or self._resolve_upstream_media_cache("video")
        dataset = CachedVideoLatentDataset(data_path)
        cache_manifest = getattr(getattr(dataset, "_sharded", None), "manifest", {})
        codec_name = cache_manifest.get("codec", {}).get("name", cfg.codec_name)
        condition_encoder = cache_manifest.get("condition_encoder", {}).get(
            "name", cfg.condition_encoder
        )
        model_config = dataset.infer_model_config(
            temporal_patch_size=cfg.temporal_patch_size,
            patch_size=cfg.patch_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )
        dimensions = {
            "latent_channels": cfg.latent_channels,
            "latent_frames": cfg.latent_frames,
            "latent_height": cfg.latent_height,
            "latent_width": cfg.latent_width,
            "condition_dim": cfg.condition_dim,
        }
        for name, configured in dimensions.items():
            if configured is not None and int(configured) != getattr(
                model_config, name
            ):
                raise ValueError(
                    f"video_generation.{name}={configured} does not match cached "
                    f"data value {getattr(model_config, name)}"
                )
        output_path = cfg.output_dir
        if not os.path.isabs(output_path):
            output_path = os.path.join(self._output_dir, output_path)
        plan = self._write_stage_plan(
            "video_generation",
            {
                "stage": "video_generation",
                "status": "planned",
                "data_path": os.path.abspath(data_path),
                "output_path": os.path.abspath(output_path),
                "samples": len(dataset),
                "model_config": model_config.to_dict(),
                "codec_name": codec_name,
                "condition_encoder": condition_encoder,
                "cache_manifest": cache_manifest or None,
                "contract": (
                    "NPZ arrays video_latents[N,C,T,H,W] and conditions[N,D]; "
                    "video encoding is an upstream causal-codec/cache stage"
                ),
            },
        )
        if self.config.training.dry_run or self.config.training.preflight_only:
            self._stage_results["video_generation"] = {
                "status": "planned",
                "plan_path": plan["path"],
                "model_config": model_config.to_dict(),
                "samples": len(dataset),
            }
            return
        from ..multimodal.VideoLatentFlowTrainer import (
            VideoLatentFlowTrainingConfig,
            train_video_latent_flow,
        )

        result = train_video_latent_flow(
            model_config,
            VideoLatentFlowTrainingConfig(
                data_path=data_path,
                output_dir=output_path,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                max_steps=cfg.max_steps,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                warmup_steps=cfg.warmup_steps,
                gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                max_grad_norm=cfg.max_grad_norm,
                num_workers=cfg.num_workers,
                checkpoint_steps=cfg.checkpoint_steps,
                seed=self.config.seed,
                device=cfg.device,
                mixed_precision=cfg.mixed_precision,
                codec_name=codec_name,
                condition_encoder=condition_encoder,
            ),
        )
        self._stage_results["video_generation"] = {
            **result,
            "plan_path": plan["path"],
        }

    def _run_world_model_stage(self):
        """Run the native action-conditioned world-model trainer."""

        cfg = self.config.world_model
        if not cfg.enabled:
            self._stage_results["world_model"] = {
                "status": "skipped",
                "reason": "world_model.enabled is false",
            }
            return
        if cfg.config_path:
            source: Any = cfg.config_path
        else:
            training = dict(cfg.training)
            output_path = training.get("output_dir", "world_model")
            if not os.path.isabs(output_path):
                output_path = os.path.join(self._output_dir, output_path)
            training["output_dir"] = output_path
            source = {
                "backend": cfg.backend,
                "model": dict(cfg.model),
                "data": dict(cfg.data),
                "training": training,
            }
        from ..world_models._WorldModelTrainer import train_world_model_from_config

        dry_run = self.config.training.dry_run or self.config.training.preflight_only
        result = train_world_model_from_config(source, dry_run=dry_run)
        plan = self._write_stage_plan(
            "world_model",
            {
                "stage": "world_model",
                "status": "planned" if dry_run else "executed",
                "source": (
                    os.path.abspath(cfg.config_path) if cfg.config_path else "inline"
                ),
                "backend": result.get("backend", cfg.backend),
                "summary": result,
            },
        )
        normalized_status = "planned" if dry_run else result.get("status", "completed")
        self._stage_results["world_model"] = {
            **result,
            "status": normalized_status,
            "plan_path": plan["path"],
        }

    def _run_eye_control_stage(self):
        """Evaluate a trained world model inside the safety-first eye simulator."""

        cfg = self.config.eye_control
        if not cfg.enabled:
            self._stage_results["eye_control"] = {
                "status": "skipped",
                "reason": "eye_control.enabled is false",
            }
            return
        world_result = self._stage_results.get("world_model", {})
        checkpoint = cfg.checkpoint_path or world_result.get("output_dir")
        if checkpoint:
            checkpoint = os.path.abspath(os.fspath(checkpoint))
        output_path = Path(cfg.output_path).expanduser()
        if not output_path.is_absolute():
            output_path = Path(self._output_dir) / output_path
        output_path = output_path.resolve()
        settings = {
            "checkpoint": checkpoint,
            "require_world_model": cfg.require_world_model,
            "episodes": cfg.episodes,
            "steps": cfg.steps,
            "device": cfg.device,
            "include_traces": cfg.include_traces,
            "safety": dict(cfg.safety),
            "pid": dict(cfg.pid),
            "plant": dict(cfg.plant),
            "planner": dict(cfg.planner),
        }
        dry_run = self.config.training.dry_run or self.config.training.preflight_only
        plan = self._write_stage_plan(
            "eye_control",
            {
                "stage": "eye_control",
                "status": "planned" if dry_run else "executed",
                "settings": settings,
                "output_path": str(output_path),
            },
        )
        if dry_run:
            self._stage_results["eye_control"] = {
                "status": "planned",
                "checkpoint_path": checkpoint,
                "output_path": str(output_path),
                "plan_path": plan["path"],
                "settings": settings,
                "artifacts": {
                    "report": {
                        "kind": "control_report",
                        "uri": str(output_path),
                    }
                },
            }
            return
        if cfg.require_world_model and not checkpoint:
            raise ValueError(
                "eye_control requires a checkpoint_path or completed world_model output"
            )

        from ..spatial.prosthetic_eye_control import (
            EyePIDConfig,
            EyePlantConfig,
            EyeSafetyConfig,
            simulate_prosthetic_eye_control,
        )
        from ..world_models.WorldModelInference import WorldModelPlannerConfig

        result = simulate_prosthetic_eye_control(
            checkpoint=checkpoint,
            episodes=cfg.episodes,
            steps=cfg.steps,
            seed=self.config.seed,
            device=cfg.device,
            safety_config=EyeSafetyConfig.from_dict(cfg.safety),
            pid_config=EyePIDConfig.from_dict(cfg.pid),
            plant_config=EyePlantConfig.from_dict(cfg.plant),
            planner_config=WorldModelPlannerConfig.from_dict(cfg.planner),
            include_traces=cfg.include_traces,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        self._stage_results["eye_control"] = {
            **result,
            "output_path": str(output_path),
            "checkpoint_path": checkpoint,
            "plan_path": plan["path"],
            "artifacts": {
                "report": {
                    "kind": "control_report",
                    "uri": str(output_path),
                }
            },
        }

    def _run_vla_stage(self):
        """Materialize VLA behavior-cloning plans and normalized robot data."""
        cfg = self.config.vla
        output_dir = (
            os.path.join(self._output_dir, "vla")
            if cfg.output_dir == "./vla"
            else cfg.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)

        from ..multimodal.VLA import VLAActionSpace, VLADataAdapter, VLATrainingPlanner

        action_space = VLAActionSpace(
            action_dim=cfg.action_dim,
            action_type=cfg.action_type,
            bins=cfg.action_bins,
            min_value=cfg.action_min_value,
            max_value=cfg.action_max_value,
            include_gripper=cfg.include_gripper,
            control_hz=cfg.control_hz,
            coordinate_frame=cfg.coordinate_frame,
        )
        plan = VLATrainingPlanner.create_plan(
            data_path=cfg.data_path,
            image_root=cfg.image_root,
            action_space=action_space,
            output_dir=output_dir,
        )
        plan.update(
            {
                "vision_backbone": cfg.vision_backbone,
                "projector": cfg.projector,
                "freeze_vision": cfg.freeze_vision,
                "freeze_llm": cfg.freeze_llm,
                "train_projector_only": cfg.train_projector_only,
                "train": cfg.train,
                "use_lora": cfg.use_lora,
                "use_qlora": cfg.use_qlora,
                "lora_r": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "learning_rate": cfg.learning_rate,
                "epochs": cfg.epochs,
                "max_steps": cfg.max_steps,
                "per_device_batch_size": cfg.per_device_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "validation_split": cfg.validation_split,
                "max_seq_length": cfg.max_seq_length,
                "train_on_prompt": cfg.train_on_prompt,
            }
        )
        if cfg.data_path and os.path.exists(cfg.data_path):
            normalized_path = os.path.join(output_dir, "vla_normalized.jsonl")
            report = VLADataAdapter.normalize_file(
                cfg.data_path,
                normalized_path,
                image_root=cfg.image_root,
                action_space=action_space,
            )
            plan["status"] = "normalized"
            plan["normalized_data_path"] = normalized_path
            plan["normalization_report"] = report.to_dict()
        else:
            plan["status"] = "planned_no_data"
            plan["warnings"] = [
                "No VLA data_path found; wrote planning artifacts only."
            ]

        train_result = None
        if cfg.train and plan.get("normalized_data_path"):
            from ..multimodal.VLATrainer import VLASFTConfig, train_vla_sft

            train_result = train_vla_sft(
                VLASFTConfig(
                    model_path=self.config.model_name_or_path
                    or self.config.model_config,
                    dataset_path=plan["normalized_data_path"],
                    output_path=os.path.join(output_dir, "checkpoints"),
                    image_root=cfg.image_root,
                    action_space=action_space,
                    use_lora=cfg.use_lora,
                    use_qlora=cfg.use_qlora,
                    lora_r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    learning_rate=cfg.learning_rate,
                    batch_size=cfg.per_device_batch_size,
                    num_epochs=cfg.epochs,
                    max_steps=cfg.max_steps,
                    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
                    validation_split=cfg.validation_split,
                    max_seq_length=cfg.max_seq_length,
                    train_on_prompt=cfg.train_on_prompt,
                    seed=self.config.seed,
                )
            )
            plan["status"] = "trained"
            plan["train_result"] = train_result

        action_space_path = os.path.join(output_dir, "action_space.json")
        with open(action_space_path, "w", encoding="utf-8") as f:
            json.dump(action_space.to_dict(), f, ensure_ascii=False, indent=2)
        plan["action_space_path"] = action_space_path

        plan_path = os.path.join(output_dir, "vla_training_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        self._stage_results["vla_sft"] = {
            "status": plan["status"],
            "model_path": (
                train_result.get("output_path")
                if isinstance(train_result, dict)
                else None
            ),
            "action_space": action_space.to_dict(),
            "training_plan": plan,
            "plan_path": plan_path,
            "normalized_data_path": plan.get("normalized_data_path"),
            "action_space_path": action_space_path,
            "train_result": train_result,
        }

    def _run_mopd_stage(self):
        """Materialize or collect MOPD-style on-policy distillation data."""
        cfg = self.config.mopd
        output_dir = (
            os.path.join(self._output_dir, "mopd")
            if cfg.output_dir == "./mopd"
            else cfg.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)
        prompts = self._load_mopd_prompts(cfg.prompts_path)
        if not cfg.dry_run and prompts:
            self._ensure_policy_model_loaded()

        from ..distillation.OnPolicyDistillation import (
            MOPDConfig,
            MultiTeacherOnPolicyDistiller,
            OnPolicyTeacherSpec,
        )

        teacher_specs = [
            OnPolicyTeacherSpec(
                name=t.get("name", t.get("model", f"teacher_{idx}")),
                domain=t.get("domain", "general"),
                weight=float(t.get("weight", 1.0)),
                instruction=t.get("instruction", ""),
                teacher=None if cfg.dry_run else self._build_mopd_teacher(t),
            )
            for idx, t in enumerate(cfg.teachers or [])
        ]
        if not teacher_specs:
            teacher_specs = [
                OnPolicyTeacherSpec(
                    name="placeholder_teacher", domain="general", weight=1.0
                )
            ]

        mopd_config = MOPDConfig(
            num_rollouts_per_prompt=cfg.num_rollouts_per_prompt,
            max_new_tokens=cfg.max_new_tokens,
            student_temperature=cfg.student_temperature,
            teacher_temperature=cfg.teacher_temperature,
            aggregation=cfg.aggregation,
            output_dir=output_dir,
        )
        distiller = MultiTeacherOnPolicyDistiller(
            teachers=teacher_specs,
            student_model=self._model,
            student_tokenizer=self._tokenizer,
            config=mopd_config,
        )
        plan = distiller.planning_summary()
        plan["dry_run"] = cfg.dry_run
        plan["prompts_path"] = cfg.prompts_path

        if cfg.dry_run or not prompts:
            plan["status"] = "planned_no_collection" if not prompts else "planned"
            if not prompts:
                plan["warnings"] = [
                    "No prompts_path found; wrote planning artifacts only."
                ]
        else:
            records = distiller.collect(prompts)
            data_path = distiller.save_jsonl(
                os.path.join(output_dir, "mopd_on_policy.jsonl"), records
            )
            plan["status"] = "completed"
            plan["num_records"] = len(records)
            plan["data_path"] = data_path

        plan_path = os.path.join(output_dir, "mopd_plan.json")
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        self._stage_results["mopd"] = {
            "status": plan["status"],
            "plan_path": plan_path,
            "data_path": plan.get("data_path"),
        }

    def _ensure_policy_model_loaded(self):
        """Load the current policy model/tokenizer for rollout-based stages."""
        if self._model is not None and self._tokenizer is not None:
            return
        model_path = (
            self._stage_results.get("rlhf", {}).get("model_path")
            or self._stage_results.get("preference", {}).get("model_path")
            or self._stage_results.get("sft", {}).get("model_path")
            or self._stage_results.get("pretrain", {}).get("model_path")
            or self.config.model_name_or_path
        )
        if not model_path:
            raise ValueError(
                "MOPD collection requires a loaded policy model or model.name_or_path"
            )
        self._load_tokenizer()
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM

            kwargs = {"trust_remote_code": True}
            if torch.cuda.is_available():
                kwargs["torch_dtype"] = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                kwargs["device_map"] = "auto"
            self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            if not torch.cuda.is_available():
                self._model = self._model.to("cpu")

    def _build_mopd_teacher(self, raw: Dict):
        from ..distillation.UniversalDistiller import TeacherInterface

        provider = raw.get("type", raw.get("provider", "openai"))
        model = raw.get("model", raw.get("name", ""))
        api_key = raw.get("api_key")
        if raw.get("api_key_env"):
            api_key = os.environ.get(raw["api_key_env"])
        if provider == "openai":
            return TeacherInterface.from_openai(model or "gpt-4o", api_key=api_key)
        if provider == "anthropic":
            return TeacherInterface.from_anthropic(
                model or "claude-sonnet-4-6", api_key=api_key
            )
        if provider == "deepseek":
            return TeacherInterface.from_deepseek(
                model or "deepseek-chat", api_key=api_key
            )
        if provider == "endpoint":
            return TeacherInterface(
                model_type="endpoint",
                model_name=model,
                endpoint_url=raw.get("endpoint_url"),
            )
        raise ValueError(f"Unsupported MOPD teacher provider: {provider}")

    @staticmethod
    def _load_mopd_prompts(path: str) -> List[str]:
        if not path or not os.path.exists(path):
            return []
        ext = os.path.splitext(path)[1].lower()
        prompts: List[str] = []
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prompt = (
                        row.get("prompt")
                        or row.get("instruction")
                        or row.get("question")
                        or row.get("text")
                    )
                    if prompt:
                        prompts.append(str(prompt))
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rows = (
                data
                if isinstance(data, list)
                else data.get("data", data.get("prompts", []))
            )
            for row in rows:
                if isinstance(row, str):
                    prompts.append(row)
                elif isinstance(row, dict):
                    prompt = (
                        row.get("prompt")
                        or row.get("instruction")
                        or row.get("question")
                        or row.get("text")
                    )
                    if prompt:
                        prompts.append(str(prompt))
        else:
            with open(path, "r", encoding="utf-8") as f:
                prompts = [line.strip() for line in f if line.strip()]
        return prompts

    def _run_eval_stage(self):
        """Stage 5: evaluate model."""
        if not self.config.eval.enabled:
            return

        logger.info("Starting evaluation stage")

        from ..evaluation.LLModelEvalute import Evaluator

        model_path = (
            self._stage_results.get("rlhf", {}).get("model_path")
            or self._stage_results.get("preference", {}).get("model_path")
            or self._stage_results.get("sft", {}).get("model_path")
            or self._stage_results.get("pretrain", {}).get("model_path")
            or self.config.model_name_or_path
        )
        if not model_path:
            raise ValueError(
                "Evaluation requires model.name_or_path or an upstream trained model."
            )

        # Basic evaluation, such as perplexity.
        try:
            evaluator = Evaluator()
            results = evaluator.evaluate(
                model_path=model_path,
                dataset=self.config.eval.eval_dataset,
                dataset_config=self.config.eval.eval_dataset_config,
                metrics=self.config.eval.tasks,
                max_samples=self.config.eval.max_samples,
            )
            if results.get("error"):
                raise RuntimeError(str(results["error"]))
            logger.info(
                f"Evaluation results: {json.dumps(results, ensure_ascii=False)}"
            )
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            self._stage_results["eval"] = {"status": "failed", "error": str(e)}
            if self.config.eval.fail_on_error:
                raise
            return

        from ..evaluation.ReleaseGate import EvaluationReleaseGate

        gate_config = self.config.eval.gate
        gate = EvaluationReleaseGate.evaluate(
            results.get("metrics", {}),
            gate_config.rules,
            enabled=gate_config.enabled,
            require_all=gate_config.require_all,
        )
        gate_path = gate_config.output_path or os.path.join(
            self._output_dir, "release_gate.json"
        )
        gate.save(gate_path)
        gate_payload = gate.to_dict()
        stage_status = "completed" if gate.accepted else "rejected"
        self._stage_results["eval"] = {
            "status": stage_status,
            **results,
            "release_gate": gate_payload,
            "release_gate_path": os.path.abspath(gate_path),
        }
        if gate_config.enabled and not gate.accepted:
            message = "Evaluation release gate rejected the model: " + "; ".join(
                gate.failures
            )
            logger.warning(message)
            if gate_config.fail_on_rejection:
                raise RuntimeError(message)

    def _run_export_stage(self):
        """Package the latest trained checkpoint as a deployable HF release."""
        export_config = self.config.export
        if not export_config.enabled:
            self._stage_results["export"] = {
                "status": "skipped",
                "reason": "export.enabled is false",
            }
            return

        model_path = (
            export_config.model_path
            or self._stage_results.get("rlhf", {}).get("model_path")
            or self._stage_results.get("preference", {}).get("model_path")
            or self._stage_results.get("sft", {}).get("model_path")
            or self._stage_results.get("pretrain", {}).get("model_path")
            or self.config.model_name_or_path
        )
        if not model_path:
            raise ValueError(
                "Export requires export.model_path, model.name_or_path, or an upstream trained model."
            )

        eval_result = self._stage_results.get("eval", {})
        gate_payload = (
            eval_result.get("release_gate") if isinstance(eval_result, dict) else None
        )
        gate_was_accepted = bool(gate_payload and gate_payload.get("accepted"))
        if self.config.eval.gate.enabled and not gate_was_accepted:
            message = "Export blocked because the enabled evaluation release gate was not accepted."
            self._stage_results["export"] = {"status": "blocked", "error": message}
            raise RuntimeError(message)
        if export_config.require_gate and not gate_was_accepted:
            message = "Export requires an accepted evaluation release gate."
            self._stage_results["export"] = {"status": "blocked", "error": message}
            raise RuntimeError(message)

        output_dir = export_config.output_dir or os.path.join(
            self._output_dir, "release"
        )
        if self.config.training.dry_run or self.config.training.preflight_only:
            self._stage_results["export"] = {
                "status": "planned",
                "model_path": model_path,
                "output_dir": os.path.abspath(output_dir),
                "format": export_config.format,
                "merge_lora": (
                    export_config.merge_lora
                    if export_config.merge_lora is not None
                    else self.config.model_backend != "saddle"
                ),
                "gate": gate_payload,
            }
            return

        from ..runtime.ModelExporter import ModelExportRequest, ModelExporter

        request = ModelExportRequest(
            model_path=model_path,
            output_dir=output_dir,
            format=export_config.format,
            merge_lora=(
                export_config.merge_lora
                if export_config.merge_lora is not None
                else self.config.model_backend != "saddle"
            ),
            safe_serialization=export_config.safe_serialization,
            trust_remote_code=export_config.trust_remote_code,
            local_files_only=export_config.local_files_only,
            device=export_config.device,
            dtype=export_config.dtype,
            hash_weights=export_config.hash_weights,
            overwrite=export_config.overwrite,
        )
        try:
            result = ModelExporter.export(request, gate=gate_payload)
        except Exception as exc:
            self._stage_results["export"] = {"status": "failed", "error": str(exc)}
            raise
        self._stage_results["export"] = result.to_dict()

    # ========================================
    # Utility methods
    # ========================================

    def _load_tokenizer(self):
        """Load or reuse a tokenizer."""
        if self._tokenizer is not None:
            return

        from ..models.TokenizerLoader import load_tokenizer_compatible

        tok_path = self.config.tokenizer_name_or_path

        if tok_path:
            self._tokenizer = load_tokenizer_compatible(
                tok_path, trust_remote_code=True
            )
        elif self.config.tokenizer_training and os.path.exists(
            self.config.tokenizer_training.output_dir
        ):
            self._tokenizer = load_tokenizer_compatible(
                self.config.tokenizer_training.output_dir, trust_remote_code=True
            )
        elif self.config.model_name_or_path:
            self._tokenizer = load_tokenizer_compatible(
                self.config.model_name_or_path, trust_remote_code=True
            )
        else:
            logger.warning(
                "No tokenizer path found; falling back to the gpt2 tokenizer"
            )
            self._tokenizer = load_tokenizer_compatible("gpt2")

        if not self._tokenizer.pad_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def _setup_logging(self):
        level = getattr(logging, self.config.logging.log_level.upper(), logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        root = logging.getLogger()
        root.setLevel(level)
        runtime = getattr(self, "_distributed_runtime", None)
        is_main_process = runtime is None or runtime.is_main_process
        log_path = os.path.join(self._output_dir, "training.log")
        for handler in list(root.handlers):
            if getattr(handler, "_saddlellm_handler", False):
                root.removeHandler(handler)
                handler.close()
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler._saddlellm_handler = True
        root.addHandler(stream_handler)
        if is_main_process:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler._saddlellm_handler = True
            root.addHandler(file_handler)

    def _close_logging_handlers(self):
        root = logging.getLogger()
        for handler in list(root.handlers):
            if getattr(handler, "_saddlellm_handler", False):
                root.removeHandler(handler)
                handler.close()

    def _print_summary(self, elapsed_seconds: Optional[float] = None):
        runtime = getattr(self, "_distributed_runtime", None)
        if runtime is not None and not runtime.is_main_process:
            return
        summary = {
            "project": self.config.project,
            "experiment": self.config.experiment,
            "model": self.config.model_name_or_path or self.config.model_config,
            "stages_completed": list(self._stage_results.keys()),
            "pipeline_plan": self._pipeline_plan.to_dict(),
            "pipeline_state_path": str(self._pipeline_state_path()),
            "restored_stages": list(self._restored_stages),
            "artifacts": self._artifact_registry.to_dict(),
            "results": self._stage_results,
            "elapsed_seconds": elapsed_seconds,
            "dry_run": self.config.training.dry_run,
            "preflight_only": self.config.training.preflight_only,
        }
        summary_path = os.path.join(self._output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        training_summary_path = os.path.join(self._output_dir, "training_summary.json")
        with open(training_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Training summary saved to {summary_path}")

        # ========================================
        # YAML/JSON parsing
        # ========================================

        try:
            from ..experiments.PretrainReport import PretrainReport

            PretrainReport.generate(self._output_dir)
        except Exception as exc:
            logger.warning("Pretrain report generation failed: %s", exc)

    @staticmethod
    def _parse_config(raw: Dict) -> TrainingConfig:
        """Parse a YAML/JSON dictionary into TrainingConfig."""
        if not isinstance(raw, Mapping):
            raise TypeError("training configuration must be a mapping")
        old_fields = [
            name
            for name in ("flow", "stage", "apiVersion", "kind", "spec")
            if name in raw
        ]
        if old_fields:
            raise ValueError(
                "Unsupported old config fields: "
                + ", ".join(old_fields)
                + ". Use one plain config with a top-level `stages` list."
            )
        stages = raw.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("training configuration requires a non-empty `stages` list")
        if any(not isinstance(stage, str) or not stage.strip() for stage in stages):
            raise ValueError("every item in `stages` must be a non-empty string")
        if len(stages) != len(set(stages)):
            raise ValueError("`stages` must not contain duplicates")
        metadata_raw = raw.get("metadata", {})
        if not isinstance(metadata_raw, Mapping):
            raise TypeError("metadata must be a mapping")
        model_raw = raw.get("model", {})
        top_level_blueprint = raw.get("model_blueprint")
        nested_blueprint = (
            model_raw.get("blueprint") if isinstance(model_raw, dict) else None
        )
        if (
            top_level_blueprint is not None
            and nested_blueprint is not None
            and top_level_blueprint != nested_blueprint
        ):
            raise ValueError(
                "model_blueprint and model.blueprint both exist but are different"
            )
        model_blueprint = (
            nested_blueprint if nested_blueprint is not None else top_level_blueprint
        )
        model_backend = model_raw.get("backend", raw.get("model_backend", "hf"))
        data_sources = []
        for s in raw.get("data", {}).get("sources", []):
            data_sources.append(
                DataSourceConfig(
                    type=s.get("type", "local"),
                    path=s.get(
                        "path",
                        s.get("paths", [""])[0]
                        if isinstance(s.get("paths"), list)
                        else s.get("paths", ""),
                    ),
                    name=s.get("name"),
                    split=s.get("split", "train"),
                    text_column=s.get("text_column", "text"),
                    format=s.get("format", "auto"),
                    pattern=s.get("pattern", "**/*.jsonl"),
                    streaming=s.get("streaming", True),
                    lang=s.get("lang", "zh"),
                    date=s.get("date", "20240301"),
                )
            )

        tok_raw = raw.get("tokenizer_training") or raw.get("tokenizer")
        tok_config = None
        if tok_raw:
            tok_config = TokenizerTrainingConfig(
                algorithm=tok_raw.get("algorithm", "bpe"),
                vocab_size=tok_raw.get("vocab_size", 32000),
                min_frequency=tok_raw.get("min_frequency", 2),
                max_token_length=tok_raw.get("max_token_length", 128),
                byte_level=tok_raw.get("byte_level", True),
                chinese_char_coverage=tok_raw.get("chinese_char_coverage", 0.995),
                limit_gb=tok_raw.get("limit_gb"),
                output_dir=tok_raw.get("output_dir", "./tokenizer"),
                eval_max_samples=tok_raw.get("eval_max_samples", 1000),
                eval_domain_terms=tok_raw.get(
                    "eval_domain_terms", tok_raw.get("domain_terms", [])
                ),
            )

        train_raw = raw.get("training", {})
        resume_from_checkpoint = train_raw.get("resume_from_checkpoint", False)
        if isinstance(resume_from_checkpoint, str):
            normalized_resume = resume_from_checkpoint.strip()
            lowered_resume = normalized_resume.lower()
            if lowered_resume in {"false", "no", "off", "0", ""}:
                resume_from_checkpoint = False
            elif lowered_resume in {"true", "yes", "on", "1"}:
                resume_from_checkpoint = True
            else:
                resume_from_checkpoint = normalized_resume
        elif not isinstance(resume_from_checkpoint, bool):
            raise TypeError(
                "training.resume_from_checkpoint must be a boolean or checkpoint path"
            )
        dist_raw = raw.get("distributed", {})
        if not isinstance(dist_raw, dict):
            raise TypeError("distributed must be a mapping")
        strategy = str(dist_raw.get("strategy", "single")).lower()
        if strategy not in {
            "single",
            "ddp",
            "fsdp",
            "deepspeed_zero2",
            "deepspeed_zero3",
        }:
            raise ValueError(f"Unsupported distributed.strategy: {strategy!r}")
        ddp_backend = str(dist_raw.get("ddp_backend", "auto")).lower()
        if ddp_backend not in {"auto", "nccl", "gloo", "mpi", "ucc", "hccl"}:
            raise ValueError(f"Unsupported distributed.ddp_backend: {ddp_backend!r}")
        pipeline_raw = raw.get("pipeline", {})
        if not isinstance(pipeline_raw, dict):
            raise TypeError("pipeline must be a mapping")
        if "dependencies" in pipeline_raw and "depends_on" in pipeline_raw:
            if pipeline_raw["dependencies"] != pipeline_raw["depends_on"]:
                raise ValueError(
                    "pipeline.dependencies and pipeline.depends_on both exist but differ"
                )
        pipeline_dependencies = pipeline_raw.get(
            "dependencies", pipeline_raw.get("depends_on")
        )
        if pipeline_dependencies is not None and not isinstance(
            pipeline_dependencies, dict
        ):
            raise TypeError("pipeline.dependencies must be a mapping")
        pipeline_resume = pipeline_raw.get("resume", False)
        if not isinstance(pipeline_resume, bool):
            raise TypeError("pipeline.resume must be a boolean")
        pipeline_rerun = pipeline_raw.get("rerun", [])
        if isinstance(pipeline_rerun, str) or not isinstance(pipeline_rerun, list):
            raise TypeError("pipeline.rerun must be a list of stage names")
        sft_raw = raw.get("sft", {})
        sft_lora_raw = sft_raw.get("lora", {})
        rlhf_raw = raw.get("rlhf", {})
        pref_raw = raw.get("preference", {})
        pref_lora_raw = pref_raw.get("lora", {})
        multimodal_raw = raw.get("multimodal", {})
        media_cache_raw = raw.get("media_cache", {})
        image_generation_raw = raw.get("image_generation", {})
        music_generation_raw = raw.get("music_generation", {})
        video_generation_raw = raw.get("video_generation", {})
        world_model_raw = raw.get("world_model", {})
        eye_control_raw = raw.get("eye_control", {})
        if not isinstance(eye_control_raw, dict):
            raise TypeError("eye_control must be a mapping")
        for section in ("safety", "pid", "plant", "planner"):
            value = eye_control_raw.get(section, {})
            if not isinstance(value, dict):
                raise TypeError(f"eye_control.{section} must be a mapping")
        for field_name in ("enabled", "require_world_model", "include_traces"):
            value = eye_control_raw.get(field_name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"eye_control.{field_name} must be a boolean")
        for field_name in ("episodes", "steps"):
            value = eye_control_raw.get(field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"eye_control.{field_name} must be an integer")
        mopd_raw = raw.get("mopd", {})
        vla_raw = raw.get("vla", {})
        eval_raw = raw.get("eval", {})
        eval_gate_raw = eval_raw.get("gate", {})
        export_raw = raw.get("export", {})
        operator_raw = raw.get("operator", {})

        cfg = TrainingConfig(
            project=raw.get("project", "saddlellm"),
            experiment=raw.get("experiment", "experiment"),
            metadata=dict(metadata_raw),
            seed=raw.get("seed", 42),
            model_config=model_raw.get("config", "gpt2-small-124m"),
            model_name_or_path=model_raw.get("name_or_path"),
            tokenizer_name_or_path=model_raw.get(
                "tokenizer", raw.get("tokenizer_name_or_path")
            ),
            model_backend=model_backend,
            model_blueprint=model_blueprint,
            pretrain_mode=raw.get("pretrain_mode", "scratch"),
            stages=list(stages),
            tokenizer_training=tok_config,
            data=DataConfig(
                sources=data_sources,
                source_weights=raw.get("data", {}).get("source_weights"),
                output_dir=raw.get("data", {}).get("output_dir", "./processed_data"),
                max_seq_length=raw.get("data", {}).get("max_seq_length", 2048),
                min_text_length=raw.get("data", {}).get("min_text_length", 50),
                dedup_method=raw.get("data", {}).get("dedup_method", "simhash"),
                dedup_threshold=raw.get("data", {}).get("dedup_threshold", 0.8),
                lang_filter=raw.get("data", {}).get("lang_filter"),
                num_proc=raw.get("data", {}).get("num_proc", 4),
                pack_sequences=raw.get("data", {}).get("pack_sequences", True),
            ),
            training=TrainingHyperparams(
                max_steps=train_raw.get("max_steps", 100000),
                per_device_batch_size=train_raw.get("per_device_batch_size", 4),
                global_batch_size=train_raw.get("global_batch_size", 512),
                learning_rate=train_raw.get("learning_rate", 3e-4),
                min_lr=train_raw.get("min_lr", 3e-5),
                warmup_steps=train_raw.get("warmup_steps", 2000),
                weight_decay=train_raw.get("weight_decay", 0.1),
                max_grad_norm=train_raw.get("max_grad_norm", 1.0),
                lr_scheduler=train_raw.get("lr_scheduler", "cosine"),
                save_every_steps=train_raw.get("save_every_steps", 5000),
                eval_every_steps=train_raw.get("eval_every_steps", 1000),
                log_every_steps=train_raw.get("log_every_steps", 50),
                keep_last_n_checkpoints=train_raw.get("keep_last_n", 5),
                resume_from_checkpoint=resume_from_checkpoint,
                dataloader_num_workers=train_raw.get("dataloader_num_workers", 4),
                stability_monitor=train_raw.get("stability_monitor", True),
                stability_loss_window=train_raw.get("stability_loss_window", 50),
                stability_spike_threshold=train_raw.get(
                    "stability_spike_threshold", 2.5
                ),
                stability_stagnation_window=train_raw.get(
                    "stability_stagnation_window", 200
                ),
                stability_output_dir=train_raw.get("stability_output_dir", ""),
                dry_run=train_raw.get("dry_run", False),
                preflight_only=train_raw.get("preflight_only", False),
            ),
            distributed=DistributedTrainingConfig(
                strategy=strategy,
                num_gpus=dist_raw.get("num_gpus", 1),
                num_nodes=dist_raw.get("num_nodes", 1),
                gradient_accumulation_steps=dist_raw.get(
                    "gradient_accumulation_steps", 1
                ),
                bf16=dist_raw.get("bf16", True),
                fp16=dist_raw.get("fp16", False),
                zero_offload_to_cpu=dist_raw.get("zero_offload_to_cpu", False),
                deepspeed_config_path=dist_raw.get("deepspeed_config_path"),
                ddp_backend=ddp_backend,
            ),
            logging=LoggingConfig(
                output_dir=raw.get("logging", {}).get("output_dir", "./outputs"),
                experiment_name=raw.get("experiment", "experiment"),
                run_name=raw.get("logging", {}).get("run_name"),
                logging_backend=raw.get("logging", {}).get("backend", "tensorboard"),
                wandb_project=raw.get("logging", {}).get("wandb_project"),
                wandb_entity=raw.get("logging", {}).get("wandb_entity"),
                log_level=raw.get("logging", {}).get("log_level", "INFO"),
                save_total_limit=raw.get("logging", {}).get("save_total_limit", 5),
            ),
            pipeline=PipelineExecutionConfig(
                dependencies=(
                    {
                        str(stage): needs
                        for stage, needs in pipeline_dependencies.items()
                    }
                    if pipeline_dependencies is not None
                    else None
                ),
                resume=pipeline_resume,
                rerun=[str(stage) for stage in pipeline_rerun],
                state_path=pipeline_raw.get("state_path"),
            ),
            sft=SFTConfig(
                enabled=sft_raw.get("enabled", False),
                data_path=sft_raw.get("data_path", ""),
                data_format=sft_raw.get("data_format", "jsonl"),
                use_lora=sft_raw.get("use_lora", True),
                use_qlora=sft_raw.get("use_qlora", True),
                lora_r=sft_lora_raw.get("r", sft_raw.get("lora_r", 16)),
                lora_alpha=sft_lora_raw.get("alpha", sft_raw.get("lora_alpha", 32)),
                lora_dropout=sft_lora_raw.get(
                    "dropout", sft_raw.get("lora_dropout", 0.05)
                ),
                lora_target_modules=sft_lora_raw.get(
                    "target_modules",
                    sft_raw.get(
                        "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                    ),
                ),
                epochs=sft_raw.get("epochs", 3),
                learning_rate=sft_raw.get("learning_rate", 2e-4),
                per_device_batch_size=sft_raw.get("per_device_batch_size", 8),
                max_seq_length=sft_raw.get(
                    "max_seq_length", raw.get("data", {}).get("max_seq_length", 2048)
                ),
                gradient_accumulation_steps=sft_raw.get(
                    "gradient_accumulation_steps",
                    dist_raw.get("gradient_accumulation_steps", 4),
                ),
                warmup_steps=sft_raw.get(
                    "warmup_steps", train_raw.get("warmup_steps", 100)
                ),
                save_steps=sft_raw.get(
                    "save_steps", train_raw.get("save_every_steps", 200)
                ),
                eval_steps=sft_raw.get(
                    "eval_steps", train_raw.get("eval_every_steps", 0)
                ),
                logging_steps=sft_raw.get(
                    "logging_steps", train_raw.get("log_every_steps", 10)
                ),
                validation_split=sft_raw.get("validation_split", 0.0),
                response_template=sft_raw.get("response_template", "### Answer:"),
                local_files_only=sft_raw.get("local_files_only", False),
                trust_remote_code=sft_raw.get("trust_remote_code", True),
                gradient_checkpointing=sft_raw.get(
                    "gradient_checkpointing",
                    train_raw.get("gradient_checkpointing", True),
                ),
                optim=sft_raw.get("optim", train_raw.get("optimizer")),
            ),
            rlhf=RLHFConfig(
                enabled=rlhf_raw.get("enabled", False),
                method=rlhf_raw.get("method", "dpo"),
                data_path=rlhf_raw.get("data_path", ""),
                beta=rlhf_raw.get("beta", 0.1),
                learning_rate=rlhf_raw.get("learning_rate", 5e-5),
                epochs=rlhf_raw.get("epochs", 1),
                per_device_batch_size=rlhf_raw.get("per_device_batch_size", 1),
                max_length=rlhf_raw.get(
                    "max_length", raw.get("data", {}).get("max_seq_length", 2048)
                ),
                max_prompt_length=rlhf_raw.get(
                    "max_prompt_length",
                    max(128, raw.get("data", {}).get("max_seq_length", 2048) // 2),
                ),
                use_lora=rlhf_raw.get("use_lora", True),
                use_qlora=rlhf_raw.get("use_qlora", True),
                lora_r=rlhf_raw.get("lora_r", 16),
                lora_alpha=rlhf_raw.get("lora_alpha", 32),
                lora_dropout=rlhf_raw.get("lora_dropout", 0.05),
                gradient_accumulation_steps=rlhf_raw.get(
                    "gradient_accumulation_steps",
                    dist_raw.get("gradient_accumulation_steps", 4),
                ),
                warmup_steps=rlhf_raw.get(
                    "warmup_steps", train_raw.get("warmup_steps", 100)
                ),
                save_steps=rlhf_raw.get(
                    "save_steps", train_raw.get("save_every_steps", 200)
                ),
            ),
            preference=PreferenceConfig(
                enabled=pref_raw.get("enabled", False),
                method=pref_raw.get("method", "dpo"),
                data_path=pref_raw.get("data_path", ""),
                beta=pref_raw.get("beta", 0.1),
                learning_rate=pref_raw.get("learning_rate", 5e-6),
                epochs=pref_raw.get("epochs", 1),
                per_device_batch_size=pref_raw.get("per_device_batch_size", 1),
                max_length=pref_raw.get("max_length", 2048),
                max_prompt_length=pref_raw.get("max_prompt_length", 1024),
                use_lora=pref_raw.get("use_lora", True),
                use_qlora=pref_raw.get("use_qlora", True),
                lora_r=pref_lora_raw.get("r", pref_raw.get("lora_r", 16)),
                lora_alpha=pref_lora_raw.get("alpha", pref_raw.get("lora_alpha", 32)),
                lora_dropout=pref_lora_raw.get(
                    "dropout", pref_raw.get("lora_dropout", 0.05)
                ),
                gradient_accumulation_steps=pref_raw.get(
                    "gradient_accumulation_steps",
                    dist_raw.get("gradient_accumulation_steps", 4),
                ),
                warmup_steps=pref_raw.get(
                    "warmup_steps", train_raw.get("warmup_steps", 100)
                ),
                save_steps=pref_raw.get(
                    "save_steps", train_raw.get("save_every_steps", 200)
                ),
                logging_steps=pref_raw.get(
                    "logging_steps", train_raw.get("log_every_steps", 10)
                ),
                validation_split=pref_raw.get("validation_split", 0.0),
                eval_steps=pref_raw.get(
                    "eval_steps", train_raw.get("eval_every_steps", 0)
                ),
                local_files_only=pref_raw.get("local_files_only", False),
                trust_remote_code=pref_raw.get("trust_remote_code", True),
                gradient_checkpointing=pref_raw.get(
                    "gradient_checkpointing",
                    train_raw.get("gradient_checkpointing", True),
                ),
                report_to=pref_raw.get(
                    "report_to", raw.get("logging", {}).get("backend", "none")
                ),
            ),
            multimodal=MultimodalTrainingConfig(
                enabled=multimodal_raw.get("enabled", False),
                stage=multimodal_raw.get("stage", "mllm_sft"),
                data_path=multimodal_raw.get("data_path", ""),
                image_root=multimodal_raw.get("image_root"),
                image_token=multimodal_raw.get("image_token", "<image>"),
                image_token_id=multimodal_raw.get("image_token_id"),
                vision_backbone=multimodal_raw.get("vision_backbone", "tiny_patch"),
                projector=multimodal_raw.get("projector", "mlp"),
                image_token_strategy=multimodal_raw.get(
                    "image_token_strategy", "replace"
                ),
                freeze_vision=multimodal_raw.get("freeze_vision", True),
                freeze_llm=multimodal_raw.get("freeze_llm", False),
                train_projector_only=multimodal_raw.get("train_projector_only", True),
                learning_rate=multimodal_raw.get("learning_rate", 2e-4),
                epochs=multimodal_raw.get("epochs", 1),
                per_device_batch_size=multimodal_raw.get("per_device_batch_size", 1),
                max_seq_length=multimodal_raw.get(
                    "max_seq_length", raw.get("data", {}).get("max_seq_length", 2048)
                ),
            ),
            media_cache=MediaCacheStageConfig(
                enabled=media_cache_raw.get("enabled", False),
                jobs=list(media_cache_raw.get("jobs", []) or []),
            ),
            image_generation=ImageGenerationTrainingConfig(
                enabled=image_generation_raw.get("enabled", False),
                data_path=image_generation_raw.get("data_path", ""),
                output_dir=image_generation_raw.get("output_dir", "./image_generation"),
                codec_name=image_generation_raw.get(
                    "codec_name", "external-image-codec"
                ),
                condition_encoder=image_generation_raw.get(
                    "condition_encoder", "external-text-encoder"
                ),
                latent_channels=image_generation_raw.get("latent_channels"),
                latent_height=image_generation_raw.get("latent_height"),
                latent_width=image_generation_raw.get("latent_width"),
                patch_size=image_generation_raw.get("patch_size", 2),
                condition_dim=image_generation_raw.get("condition_dim"),
                hidden_size=image_generation_raw.get("hidden_size", 512),
                num_layers=image_generation_raw.get("num_layers", 8),
                num_heads=image_generation_raw.get("num_heads", 8),
                mlp_ratio=image_generation_raw.get("mlp_ratio", 4.0),
                dropout=image_generation_raw.get("dropout", 0.0),
                batch_size=image_generation_raw.get("batch_size", 4),
                epochs=image_generation_raw.get("epochs", 1),
                max_steps=image_generation_raw.get("max_steps", -1),
                learning_rate=image_generation_raw.get("learning_rate", 1e-4),
                weight_decay=image_generation_raw.get("weight_decay", 0.01),
                warmup_steps=image_generation_raw.get("warmup_steps", 0),
                gradient_accumulation_steps=image_generation_raw.get(
                    "gradient_accumulation_steps", 1
                ),
                max_grad_norm=image_generation_raw.get("max_grad_norm", 1.0),
                num_workers=image_generation_raw.get("num_workers", 0),
                checkpoint_steps=image_generation_raw.get("checkpoint_steps", 500),
                device=image_generation_raw.get("device", "auto"),
                mixed_precision=image_generation_raw.get("mixed_precision", "no"),
            ),
            music_generation=MusicGenerationTrainingConfig(
                enabled=music_generation_raw.get("enabled", False),
                data_path=music_generation_raw.get("data_path", ""),
                output_dir=music_generation_raw.get("output_dir", "./music_generation"),
                codec_name=music_generation_raw.get(
                    "codec_name", "external-audio-codec"
                ),
                condition_encoder=music_generation_raw.get(
                    "condition_encoder", "external-text-encoder"
                ),
                num_codebooks=music_generation_raw.get("num_codebooks"),
                codebook_size=music_generation_raw.get("codebook_size"),
                max_sequence_length=music_generation_raw.get("max_sequence_length"),
                condition_dim=music_generation_raw.get("condition_dim"),
                hidden_size=music_generation_raw.get("hidden_size", 512),
                num_layers=music_generation_raw.get("num_layers", 8),
                num_heads=music_generation_raw.get("num_heads", 8),
                mlp_ratio=music_generation_raw.get("mlp_ratio", 4.0),
                dropout=music_generation_raw.get("dropout", 0.0),
                bos_token_id=music_generation_raw.get("bos_token_id", 0),
                batch_size=music_generation_raw.get("batch_size", 4),
                epochs=music_generation_raw.get("epochs", 1),
                max_steps=music_generation_raw.get("max_steps", -1),
                learning_rate=music_generation_raw.get("learning_rate", 2e-4),
                weight_decay=music_generation_raw.get("weight_decay", 0.01),
                warmup_steps=music_generation_raw.get("warmup_steps", 0),
                gradient_accumulation_steps=music_generation_raw.get(
                    "gradient_accumulation_steps", 1
                ),
                max_grad_norm=music_generation_raw.get("max_grad_norm", 1.0),
                num_workers=music_generation_raw.get("num_workers", 0),
                checkpoint_steps=music_generation_raw.get("checkpoint_steps", 500),
                device=music_generation_raw.get("device", "auto"),
                mixed_precision=music_generation_raw.get("mixed_precision", "no"),
            ),
            video_generation=VideoGenerationTrainingConfig(
                enabled=video_generation_raw.get("enabled", False),
                data_path=video_generation_raw.get("data_path", ""),
                output_dir=video_generation_raw.get("output_dir", "./video_generation"),
                codec_name=video_generation_raw.get(
                    "codec_name", "external-causal-video-codec"
                ),
                condition_encoder=video_generation_raw.get(
                    "condition_encoder", "external-text-encoder"
                ),
                latent_channels=video_generation_raw.get("latent_channels"),
                latent_frames=video_generation_raw.get("latent_frames"),
                latent_height=video_generation_raw.get("latent_height"),
                latent_width=video_generation_raw.get("latent_width"),
                temporal_patch_size=video_generation_raw.get("temporal_patch_size", 1),
                patch_size=video_generation_raw.get("patch_size", 2),
                condition_dim=video_generation_raw.get("condition_dim"),
                hidden_size=video_generation_raw.get("hidden_size", 512),
                num_layers=video_generation_raw.get("num_layers", 8),
                num_heads=video_generation_raw.get("num_heads", 8),
                mlp_ratio=video_generation_raw.get("mlp_ratio", 4.0),
                dropout=video_generation_raw.get("dropout", 0.0),
                batch_size=video_generation_raw.get("batch_size", 2),
                epochs=video_generation_raw.get("epochs", 1),
                max_steps=video_generation_raw.get("max_steps", -1),
                learning_rate=video_generation_raw.get("learning_rate", 1e-4),
                weight_decay=video_generation_raw.get("weight_decay", 0.01),
                warmup_steps=video_generation_raw.get("warmup_steps", 0),
                gradient_accumulation_steps=video_generation_raw.get(
                    "gradient_accumulation_steps", 1
                ),
                max_grad_norm=video_generation_raw.get("max_grad_norm", 1.0),
                num_workers=video_generation_raw.get("num_workers", 0),
                checkpoint_steps=video_generation_raw.get("checkpoint_steps", 500),
                device=video_generation_raw.get("device", "auto"),
                mixed_precision=video_generation_raw.get("mixed_precision", "no"),
            ),
            world_model=WorldModelStageConfig(
                enabled=world_model_raw.get("enabled", False),
                config_path=world_model_raw.get("config_path"),
                backend=world_model_raw.get("backend", "rssm"),
                model=dict(world_model_raw.get("model", {}) or {}),
                data=dict(world_model_raw.get("data", {}) or {}),
                training=dict(world_model_raw.get("training", {}) or {}),
            ),
            eye_control=EyeControlStageConfig(
                enabled=eye_control_raw.get("enabled", False),
                checkpoint_path=eye_control_raw.get("checkpoint_path"),
                require_world_model=eye_control_raw.get("require_world_model", True),
                output_path=eye_control_raw.get(
                    "output_path", "eye_control/evaluation.json"
                ),
                episodes=eye_control_raw.get("episodes", 8),
                steps=eye_control_raw.get("steps", 150),
                device=eye_control_raw.get("device", "auto"),
                include_traces=eye_control_raw.get("include_traces", False),
                safety=dict(eye_control_raw.get("safety", {}) or {}),
                pid=dict(eye_control_raw.get("pid", {}) or {}),
                plant=dict(eye_control_raw.get("plant", {}) or {}),
                planner=dict(eye_control_raw.get("planner", {}) or {}),
            ),
            mopd=MOPDTrainingConfig(
                enabled=mopd_raw.get("enabled", False),
                prompts_path=mopd_raw.get("prompts_path", ""),
                output_dir=mopd_raw.get("output_dir", "./mopd"),
                teachers=mopd_raw.get("teachers", []),
                num_rollouts_per_prompt=mopd_raw.get("num_rollouts_per_prompt", 1),
                max_new_tokens=mopd_raw.get("max_new_tokens", 1024),
                student_temperature=mopd_raw.get("student_temperature", 0.9),
                teacher_temperature=mopd_raw.get("teacher_temperature", 0.3),
                aggregation=mopd_raw.get("aggregation", "best_score"),
                dry_run=mopd_raw.get("dry_run", True),
            ),
            vla=VLATrainingConfig(
                enabled=vla_raw.get("enabled", False),
                stage=vla_raw.get("stage", "vla_sft"),
                data_path=vla_raw.get("data_path", ""),
                image_root=vla_raw.get("image_root"),
                output_dir=vla_raw.get("output_dir", "./vla"),
                action_dim=vla_raw.get("action_space", {}).get(
                    "action_dim", vla_raw.get("action_dim", 7)
                ),
                action_type=vla_raw.get("action_space", {}).get(
                    "action_type", vla_raw.get("action_type", "continuous")
                ),
                action_bins=vla_raw.get("action_space", {}).get(
                    "bins", vla_raw.get("action_bins", 256)
                ),
                action_min_value=vla_raw.get("action_space", {}).get(
                    "min_value", vla_raw.get("action_min_value", -1.0)
                ),
                action_max_value=vla_raw.get("action_space", {}).get(
                    "max_value", vla_raw.get("action_max_value", 1.0)
                ),
                include_gripper=vla_raw.get("action_space", {}).get(
                    "include_gripper", vla_raw.get("include_gripper", True)
                ),
                control_hz=vla_raw.get("action_space", {}).get(
                    "control_hz", vla_raw.get("control_hz", 10.0)
                ),
                coordinate_frame=vla_raw.get("action_space", {}).get(
                    "coordinate_frame",
                    vla_raw.get("coordinate_frame", "end_effector_delta"),
                ),
                vision_backbone=vla_raw.get("vision_backbone", "tiny_patch"),
                projector=vla_raw.get("projector", "mlp"),
                freeze_vision=vla_raw.get("freeze_vision", True),
                freeze_llm=vla_raw.get("freeze_llm", False),
                train_projector_only=vla_raw.get("train_projector_only", False),
                train=vla_raw.get("train", False),
                use_lora=vla_raw.get("use_lora", True),
                use_qlora=vla_raw.get("use_qlora", False),
                lora_r=vla_raw.get("lora_r", 16),
                lora_alpha=vla_raw.get("lora_alpha", 32),
                lora_dropout=vla_raw.get("lora_dropout", 0.05),
                learning_rate=vla_raw.get("learning_rate", 2e-4),
                epochs=vla_raw.get("epochs", 1),
                max_steps=vla_raw.get("max_steps", -1),
                per_device_batch_size=vla_raw.get("per_device_batch_size", 1),
                gradient_accumulation_steps=vla_raw.get(
                    "gradient_accumulation_steps", 4
                ),
                validation_split=vla_raw.get("validation_split", 0.0),
                max_seq_length=vla_raw.get(
                    "max_seq_length", raw.get("data", {}).get("max_seq_length", 2048)
                ),
                train_on_prompt=vla_raw.get("train_on_prompt", False),
            ),
            eval=EvalConfig(
                enabled=eval_raw.get("enabled", True),
                tasks=eval_raw.get("tasks", ["perplexity"]),
                eval_dataset=eval_raw.get("dataset", "wikitext"),
                eval_dataset_config=eval_raw.get("dataset_config", "wikitext-2-raw-v1"),
                max_samples=eval_raw.get("max_samples", 1000),
                fail_on_error=eval_raw.get("fail_on_error", True),
                gate=EvaluationGateConfig(
                    enabled=eval_gate_raw.get("enabled", False),
                    rules=eval_gate_raw.get("rules", {}),
                    require_all=eval_gate_raw.get("require_all", True),
                    fail_on_rejection=eval_gate_raw.get("fail_on_rejection", True),
                    output_path=eval_gate_raw.get("output_path"),
                ),
            ),
            export=ExportConfig(
                enabled=export_raw.get("enabled", False),
                output_dir=export_raw.get("output_dir"),
                format=export_raw.get(
                    "format", "saddle" if model_backend == "saddle" else "hf"
                ),
                model_path=export_raw.get("model_path"),
                merge_lora=export_raw.get("merge_lora"),
                safe_serialization=export_raw.get("safe_serialization", True),
                require_gate=export_raw.get("require_gate", False),
                trust_remote_code=export_raw.get("trust_remote_code", False),
                local_files_only=export_raw.get("local_files_only", False),
                device=export_raw.get("device", "auto"),
                dtype=export_raw.get("dtype", "auto"),
                hash_weights=export_raw.get("hash_weights", False),
                overwrite=export_raw.get("overwrite", False),
            ),
            operator=OperatorStageConfig(
                enabled=operator_raw.get("enabled", False),
                label=operator_raw.get("label", ""),
                config=operator_raw.get("config", {}),
                upstream_result=operator_raw.get("upstream_result", {}),
                work_dir=operator_raw.get(
                    "work_dir", raw.get("logging", {}).get("output_dir", "./outputs")
                ),
            ),
        )
        return cfg
