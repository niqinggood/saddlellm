"""Unified recipe schema for SaddleLLM training jobs.

The recipe is a compact user-facing config that compiles into the existing
TrainingOrchestrator dictionary format.  It keeps the public interface stable
while allowing the execution internals to evolve.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Union


@dataclass
class RecipeModelConfig:
    name_or_path: Optional[str] = None
    config: str = "qwen-tiny-160m"
    tokenizer: Optional[str] = None
    backend: str = "hf"
    pretrain_mode: str = "scratch"
    params: Optional[int] = None
    is_moe: bool = False
    blueprint: Optional[Dict] = None


@dataclass
class RecipeMethodConfig:
    type: str = "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    preference_method: str = "dpo"
    beta: float = 0.1


@dataclass
class RecipeDataConfig:
    manifest: Optional[str] = None
    manifest_role: Optional[str] = None
    sources: List[Dict] = field(default_factory=list)
    source_weights: Optional[List[float]] = None
    output_dir: str = "./processed_data"
    max_seq_length: int = 2048
    min_text_length: int = 50
    dedup_method: str = "simhash"
    dedup_threshold: float = 0.8
    lang_filter: Optional[str] = None
    num_proc: int = 4
    pack_sequences: bool = True


@dataclass
class RecipeMultimodalConfig:
    enabled: bool = False
    data_path: Optional[str] = None
    image_root: Optional[str] = None
    image_token: str = "<image>"
    image_token_id: Optional[int] = None
    vision_backbone: str = "tiny_patch"
    projector: str = "mlp"
    image_token_strategy: str = "replace"
    freeze_vision: bool = True
    freeze_llm: bool = False
    train_projector_only: bool = True


@dataclass
class RecipeMOPDConfig:
    enabled: bool = False
    prompts_path: Optional[str] = None
    output_dir: str = "./mopd"
    teachers: List[Dict] = field(default_factory=list)
    num_rollouts_per_prompt: int = 1
    max_new_tokens: int = 1024
    student_temperature: float = 0.9
    teacher_temperature: float = 0.3
    aggregation: str = "best_score"
    dry_run: bool = True


@dataclass
class RecipeVLAConfig:
    enabled: bool = False
    data_path: Optional[str] = None
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
    gradient_accumulation_steps: int = 4
    validation_split: float = 0.0
    train_on_prompt: bool = False


@dataclass
class RecipeBackendConfig:
    strategy: str = "auto"
    num_gpus: int = 1
    num_nodes: int = 1
    gpu_memory_gb: float = 24.0
    gradient_accumulation_steps: Optional[int] = None
    bf16: bool = True
    fp16: bool = False
    zero_offload_to_cpu: bool = False
    deepspeed_config_path: Optional[str] = None
    ddp_backend: str = "auto"


@dataclass
class RecipeTrainingConfig:
    max_steps: int = 100000
    per_device_batch_size: int = 4
    global_batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    save_every_steps: int = 5000
    eval_every_steps: int = 1000
    log_every_steps: int = 50
    epochs: int = 3
    dry_run: bool = False
    preflight_only: bool = False


@dataclass
class RecipeEvalConfig:
    enabled: bool = True
    tasks: List[str] = field(default_factory=lambda: ["perplexity"])
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_samples: int = 1000


@dataclass
class RecipeLoggingConfig:
    output_dir: str = "./outputs"
    backend: str = "tensorboard"
    run_name: Optional[str] = None
    log_level: str = "INFO"


@dataclass
class TrainingRecipe:
    project: str = "saddlellm"
    experiment: str = "experiment"
    stage: Union[str, List[str]] = "sft"
    seed: int = 42
    model: RecipeModelConfig = field(default_factory=RecipeModelConfig)
    method: RecipeMethodConfig = field(default_factory=RecipeMethodConfig)
    data: RecipeDataConfig = field(default_factory=RecipeDataConfig)
    multimodal: RecipeMultimodalConfig = field(default_factory=RecipeMultimodalConfig)
    mopd: RecipeMOPDConfig = field(default_factory=RecipeMOPDConfig)
    vla: RecipeVLAConfig = field(default_factory=RecipeVLAConfig)
    backend: RecipeBackendConfig = field(default_factory=RecipeBackendConfig)
    training: RecipeTrainingConfig = field(default_factory=RecipeTrainingConfig)
    eval: RecipeEvalConfig = field(default_factory=RecipeEvalConfig)
    logging: RecipeLoggingConfig = field(default_factory=RecipeLoggingConfig)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict) -> "TrainingRecipe":
        return cls(
            project=raw.get("project", "saddlellm"),
            experiment=raw.get("experiment", "experiment"),
            stage=raw.get("stage", raw.get("stages", "sft")),
            seed=raw.get("seed", 42),
            model=RecipeModelConfig(**raw.get("model", {})),
            method=RecipeMethodConfig(**raw.get("method", {})),
            data=RecipeDataConfig(**raw.get("data", {})),
            multimodal=RecipeMultimodalConfig(**raw.get("multimodal", {})),
            mopd=RecipeMOPDConfig(**raw.get("mopd", {})),
            vla=cls._parse_vla_config(raw.get("vla", {})),
            backend=RecipeBackendConfig(**raw.get("backend", raw.get("distributed", {}))),
            training=RecipeTrainingConfig(**raw.get("training", {})),
            eval=RecipeEvalConfig(**raw.get("eval", {})),
            logging=RecipeLoggingConfig(**raw.get("logging", {})),
        )

    @staticmethod
    def _parse_vla_config(raw: Dict) -> RecipeVLAConfig:
        data = dict(raw or {})
        action_space = data.pop("action_space", {}) or {}
        if isinstance(action_space, dict):
            aliases = {
                "action_dim": "action_dim",
                "action_type": "action_type",
                "bins": "action_bins",
                "action_bins": "action_bins",
                "min_value": "action_min_value",
                "action_min_value": "action_min_value",
                "max_value": "action_max_value",
                "action_max_value": "action_max_value",
                "include_gripper": "include_gripper",
                "control_hz": "control_hz",
                "coordinate_frame": "coordinate_frame",
            }
            for key, target in aliases.items():
                if key in action_space and target not in data:
                    data[target] = action_space[key]
        return RecipeVLAConfig(**data)

    @classmethod
    def load(cls, path: str) -> "TrainingRecipe":
        with open(path, "r", encoding="utf-8") as f:
            if path.lower().endswith((".yaml", ".yml")):
                import yaml
                return cls.from_dict(yaml.safe_load(f) or {})
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            if path.lower().endswith((".yaml", ".yml")):
                import yaml
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            else:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def template(
        cls,
        stage: str = "sft",
        domain: str = "research",
        output_dir: str = "./outputs",
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    ) -> "TrainingRecipe":
        model = RecipeModelConfig(
            name_or_path=base_model if stage != "pretrain" else None,
            config="qwen-tiny-160m",
            tokenizer=base_model,
            pretrain_mode="continue" if stage == "continued_pretrain" else "scratch",
            params=7_000_000_000 if stage != "pretrain" else 160_000_000,
        )
        data_role = {
            "pretrain": "pretrain",
            "continued_pretrain": "pretrain",
            "sft": "sft",
            "preference": "preference",
            "rlhf": "preference",
            "mopd": "sft",
            "vla_sft": "vla_sft",
            "mllm_sft": "mllm_sft",
            "vision_alignment": "mllm_sft",
        }.get(stage, "sft")
        return cls(
            experiment=f"{domain}-{stage}",
            stage="pretrain" if stage == "continued_pretrain" else ("mllm_sft" if stage == "vision_alignment" else stage),
            model=model,
            data=RecipeDataConfig(
                manifest="./configs/dataset_manifest.json",
                manifest_role=data_role,
                output_dir="./data/processed",
            ),
            logging=RecipeLoggingConfig(output_dir=output_dir),
        )

    def stages(self) -> List[str]:
        if isinstance(self.stage, str):
            if self.stage == "continued_pretrain":
                return ["pretrain", "eval"]
            if self.stage == "vision_alignment":
                return ["mllm_sft", "eval"] if self.eval.enabled else ["mllm_sft"]
            return [self.stage, "eval"] if self.stage != "eval" and self.eval.enabled else [self.stage]
        return list(self.stage)

    def compile(self, base_dir: Optional[str] = None, save_backend_plan: bool = False) -> Dict:
        """Compile this recipe into a TrainingOrchestrator-compatible dict."""
        base_dir = base_dir or os.getcwd()
        stages = self.stages()
        sources, weights = self._resolve_sources(base_dir)
        distributed = self._resolve_distributed(stages, save_backend_plan)

        compiled = {
            "project": self.project,
            "experiment": self.experiment,
            "seed": self.seed,
            "model": {
                "config": self.model.config,
                "name_or_path": self.model.name_or_path,
                "tokenizer": self.model.tokenizer or self.model.name_or_path,
                "backend": self.model.backend,
                "blueprint": self.model.blueprint,
            },
            "pretrain_mode": self.model.pretrain_mode,
            "stages": stages,
            "data": {
                "sources": sources,
                "source_weights": weights,
                "output_dir": self.data.output_dir,
                "max_seq_length": self.data.max_seq_length,
                "min_text_length": self.data.min_text_length,
                "dedup_method": self.data.dedup_method,
                "dedup_threshold": self.data.dedup_threshold,
                "lang_filter": self.data.lang_filter,
                "num_proc": self.data.num_proc,
                "pack_sequences": self.data.pack_sequences,
            },
            "training": {
                "max_steps": self.training.max_steps,
                "per_device_batch_size": self.training.per_device_batch_size,
                "global_batch_size": self.training.global_batch_size,
                "learning_rate": self.training.learning_rate,
                "min_lr": self.training.min_lr,
                "warmup_steps": self.training.warmup_steps,
                "weight_decay": self.training.weight_decay,
                "save_every_steps": self.training.save_every_steps,
                "eval_every_steps": self.training.eval_every_steps,
                "log_every_steps": self.training.log_every_steps,
                "dry_run": self.training.dry_run,
                "preflight_only": self.training.preflight_only,
            },
            "distributed": distributed,
            "logging": {
                "output_dir": self.logging.output_dir,
                "backend": self.logging.backend,
                "run_name": self.logging.run_name,
                "log_level": self.logging.log_level,
            },
            "eval": {
                "enabled": self.eval.enabled,
                "tasks": self.eval.tasks,
                "dataset": self.eval.dataset,
                "dataset_config": self.eval.dataset_config,
                "max_samples": self.eval.max_samples,
            },
        }
        if "sft" in stages:
            compiled["sft"] = self._compile_sft()
        if "preference" in stages:
            compiled["preference"] = self._compile_preference()
        if "rlhf" in stages:
            compiled["rlhf"] = self._compile_rlhf()
        if "mllm_sft" in stages:
            compiled["multimodal"] = self._compile_multimodal()
        if "mopd" in stages:
            compiled["mopd"] = self._compile_mopd()
        if "vla_sft" in stages:
            compiled["vla"] = self._compile_vla()
        return compiled

    def save_compiled(self, path: str, base_dir: Optional[str] = None, save_backend_plan: bool = True) -> str:
        compiled = self.compile(base_dir=base_dir, save_backend_plan=save_backend_plan)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if path.lower().endswith((".yaml", ".yml")):
                import yaml
                yaml.safe_dump(compiled, f, allow_unicode=True, sort_keys=False)
            else:
                json.dump(compiled, f, ensure_ascii=False, indent=2)
        return path

    def _resolve_sources(self, base_dir: str) -> (List[Dict], Optional[List[float]]):
        if self.data.sources:
            return list(self.data.sources), self.data.source_weights
        if not self.data.manifest:
            return [], self.data.source_weights

        from .DatasetManifest import DatasetManifest

        manifest_path = self._resolve_path(self.data.manifest, base_dir)
        if not os.path.exists(manifest_path):
            return [], self.data.source_weights
        manifest = DatasetManifest.load(manifest_path)
        role = self.data.manifest_role
        return manifest.to_saddle_sources(role=role), manifest.source_weights(role=role)

    def _resolve_distributed(self, stages: Sequence[str], save_backend_plan: bool) -> Dict:
        strategy = self.backend.strategy
        distributed = {
            "strategy": strategy if strategy != "auto" else "single",
            "num_gpus": self.backend.num_gpus,
            "num_nodes": self.backend.num_nodes,
            "gradient_accumulation_steps": self.backend.gradient_accumulation_steps or 1,
            "bf16": self.backend.bf16,
            "fp16": self.backend.fp16,
            "zero_offload_to_cpu": self.backend.zero_offload_to_cpu,
            "deepspeed_config_path": self.backend.deepspeed_config_path,
            "ddp_backend": self.backend.ddp_backend,
        }
        if strategy == "auto" and self.model.params:
            from .FactoryBackendPlanner import FactoryBackendPlanner

            stage = "pretrain" if "pretrain" in stages else stages[0]
            plan = FactoryBackendPlanner.recommend_parallelism(
                model_params=self.model.params,
                num_gpus=self.backend.num_gpus,
                gpu_memory_gb=self.backend.gpu_memory_gb,
                seq_length=self.data.max_seq_length,
                global_batch_size=self.training.global_batch_size,
                is_moe=self.model.is_moe,
                training_stage=stage,
            )
            distributed.update(FactoryBackendPlanner.to_training_orchestrator_distributed(plan))
            if self.backend.gradient_accumulation_steps:
                distributed["gradient_accumulation_steps"] = self.backend.gradient_accumulation_steps
            if save_backend_plan:
                path = os.path.join(self.logging.output_dir, "backend_plan.json")
                FactoryBackendPlanner.save_plan(path, plan)
                distributed["backend_plan_path"] = path
        return distributed

    def _compile_sft(self) -> Dict:
        data_path = "" if self._stage_precedes("mopd", "sft") else self._first_data_path()
        return {
            "enabled": True,
            "data_path": data_path,
            "data_format": "jsonl",
            "use_lora": self.method.type in {"lora", "qlora"},
            "use_qlora": self.method.type == "qlora",
            "lora": {
                "r": self.method.lora_r,
                "alpha": self.method.lora_alpha,
                "dropout": self.method.lora_dropout,
                "target_modules": self.method.target_modules,
            },
            "epochs": self.training.epochs,
            "learning_rate": self.training.learning_rate,
            "per_device_batch_size": self.training.per_device_batch_size,
            "max_seq_length": self.data.max_seq_length,
            "gradient_accumulation_steps": self.backend.gradient_accumulation_steps or 4,
            "warmup_steps": self.training.warmup_steps,
            "save_steps": self.training.save_every_steps,
            "eval_steps": self.training.eval_every_steps,
            "validation_split": 0.0,
        }

    def _compile_preference(self) -> Dict:
        return {
            "enabled": True,
            "method": self.method.preference_method,
            "data_path": self._first_data_path(),
            "beta": self.method.beta,
            "learning_rate": self.training.learning_rate,
            "epochs": self.training.epochs,
            "per_device_batch_size": self.training.per_device_batch_size,
            "max_length": self.data.max_seq_length,
            "max_prompt_length": max(128, self.data.max_seq_length // 2),
            "use_lora": self.method.type in {"lora", "qlora"},
            "use_qlora": self.method.type == "qlora",
            "lora": {
                "r": self.method.lora_r,
                "alpha": self.method.lora_alpha,
                "dropout": self.method.lora_dropout,
            },
            "gradient_accumulation_steps": self.backend.gradient_accumulation_steps or 4,
            "warmup_steps": self.training.warmup_steps,
            "save_steps": self.training.save_every_steps,
            "report_to": self.logging.backend if self.logging.backend != "local" else "none",
        }

    def _compile_rlhf(self) -> Dict:
        return {
            "enabled": True,
            "method": self.method.preference_method if self.method.preference_method in {"dpo", "ppo"} else "dpo",
            "data_path": self._first_data_path(),
            "beta": self.method.beta,
            "learning_rate": self.training.learning_rate,
            "epochs": self.training.epochs,
            "per_device_batch_size": self.training.per_device_batch_size,
            "max_length": self.data.max_seq_length,
            "max_prompt_length": max(128, self.data.max_seq_length // 2),
            "use_lora": self.method.type in {"lora", "qlora"},
            "use_qlora": self.method.type == "qlora",
            "lora_r": self.method.lora_r,
            "lora_alpha": self.method.lora_alpha,
            "lora_dropout": self.method.lora_dropout,
            "gradient_accumulation_steps": self.backend.gradient_accumulation_steps or 4,
            "warmup_steps": self.training.warmup_steps,
            "save_steps": self.training.save_every_steps,
        }

    def _compile_multimodal(self) -> Dict:
        return {
            "enabled": True,
            "stage": "mllm_sft",
            "data_path": self.multimodal.data_path or self._first_data_path(),
            "image_root": self.multimodal.image_root,
            "image_token": self.multimodal.image_token,
            "image_token_id": self.multimodal.image_token_id,
            "vision_backbone": self.multimodal.vision_backbone,
            "projector": self.multimodal.projector,
            "image_token_strategy": self.multimodal.image_token_strategy,
            "freeze_vision": self.multimodal.freeze_vision,
            "freeze_llm": self.multimodal.freeze_llm,
            "train_projector_only": self.multimodal.train_projector_only,
            "learning_rate": self.training.learning_rate,
            "epochs": self.training.epochs,
            "per_device_batch_size": self.training.per_device_batch_size,
            "max_seq_length": self.data.max_seq_length,
            "status": "planning_only",
            "notes": [
                "This stage materializes multimodal data/config plans.",
                "Full multimodal trainer execution should be enabled after image preprocessing and collator support are validated.",
            ],
        }

    def _compile_mopd(self) -> Dict:
        return {
            "enabled": True,
            "prompts_path": self.mopd.prompts_path or self._first_data_path(),
            "output_dir": self.mopd.output_dir,
            "teachers": self.mopd.teachers,
            "num_rollouts_per_prompt": self.mopd.num_rollouts_per_prompt,
            "max_new_tokens": self.mopd.max_new_tokens,
            "student_temperature": self.mopd.student_temperature,
            "teacher_temperature": self.mopd.teacher_temperature,
            "aggregation": self.mopd.aggregation,
            "dry_run": self.mopd.dry_run,
            "notes": [
                "MOPD uses student on-policy rollouts before teacher supervision.",
                "Set dry_run=false and provide teacher specs to materialize distillation data.",
            ],
        }

    def _compile_vla(self) -> Dict:
        return {
            "enabled": True,
            "stage": "vla_sft",
            "data_path": self.vla.data_path or self._first_data_path(),
            "image_root": self.vla.image_root,
            "output_dir": self.vla.output_dir,
            "action_space": {
                "action_dim": self.vla.action_dim,
                "action_type": self.vla.action_type,
                "bins": self.vla.action_bins,
                "min_value": self.vla.action_min_value,
                "max_value": self.vla.action_max_value,
                "include_gripper": self.vla.include_gripper,
                "control_hz": self.vla.control_hz,
                "coordinate_frame": self.vla.coordinate_frame,
            },
            "vision_backbone": self.vla.vision_backbone,
            "projector": self.vla.projector,
            "freeze_vision": self.vla.freeze_vision,
            "freeze_llm": self.vla.freeze_llm,
            "train_projector_only": self.vla.train_projector_only,
            "train": self.vla.train,
            "use_lora": self.vla.use_lora,
            "use_qlora": self.vla.use_qlora,
            "lora_r": self.vla.lora_r,
            "lora_alpha": self.vla.lora_alpha,
            "lora_dropout": self.vla.lora_dropout,
            "learning_rate": self.training.learning_rate,
            "epochs": self.training.epochs,
            "per_device_batch_size": self.training.per_device_batch_size,
            "gradient_accumulation_steps": self.vla.gradient_accumulation_steps,
            "validation_split": self.vla.validation_split,
            "max_seq_length": self.data.max_seq_length,
            "train_on_prompt": self.vla.train_on_prompt,
            "status": "planning_only",
            "notes": [
                "VLA SFT normalizes robot steps into multimodal messages with action tokens.",
                "Set vla.train=true to run the lightweight action-token behavior-cloning trainer.",
            ],
        }

    def _first_data_path(self) -> str:
        if self.data.sources:
            return self.data.sources[0].get("path", "")
        if self.data.manifest:
            try:
                sources, _ = self._resolve_sources(os.getcwd())
                return sources[0].get("path", "") if sources else ""
            except Exception:
                return self.data.manifest
        return ""

    def _stage_precedes(self, before: str, after: str) -> bool:
        stages = self.stages()
        return before in stages and after in stages and stages.index(before) < stages.index(after)

    @staticmethod
    def _resolve_path(path: str, base_dir: str) -> str:
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(base_dir, path))
