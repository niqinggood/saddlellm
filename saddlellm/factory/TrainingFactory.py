"""Top-level LLM training factory interface.

This module provides a project-oriented facade over SaddleLLM's lower-level
components.  It is intentionally conservative: methods plan and materialize
artifacts first, and only run training when explicitly requested.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class FactoryConfig:
    name: str = "saddlellm-factory"
    root_dir: str = "./llm_factory"
    domain: str = "research"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_path: Optional[str] = None
    scratch_models: List[str] = field(default_factory=lambda: ["qwen-tiny-160m", "qwen-300m"])
    token_multipliers: List[float] = field(default_factory=lambda: [0.01, 0.05])
    max_seq_length: int = 1024
    global_batch_size: int = 64
    num_gpus: int = 1
    gpu_memory_gb: float = 24.0
    prefer_backend: str = "auto"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FactoryStatus:
    root_dir: str
    exists: bool
    directories: Dict[str, bool]
    artifacts: Dict[str, bool]
    latest_manifest: Optional[str] = None
    latest_report: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


def _single_process_config(num_gpus: int = 1) -> Dict:
    """Return the smallest explicit launch configuration."""
    return {
        "strategy": "ddp" if num_gpus > 1 else "single",
        "num_gpus": num_gpus,
        "num_nodes": 1,
        "gradient_accumulation_steps": 1,
    }


def _make_training_config(
    *,
    project: str,
    experiment: str,
    stages: Sequence[str],
    model: Dict,
    data: Dict,
    training: Dict,
    distributed: Dict,
    output_dir: str,
    pretrain_mode: str = "continue",
    **stage_configs,
) -> Dict:
    """Build the plain dictionary consumed by TrainingOrchestrator."""
    config = {
        "project": project,
        "experiment": experiment,
        "model": dict(model),
        "pretrain_mode": pretrain_mode,
        "stages": list(stages),
        "data": dict(data),
        "training": dict(training),
        "distributed": dict(distributed),
        "logging": {"output_dir": output_dir, "backend": "local"},
        "eval": {"enabled": "eval" in stages},
    }
    config.update(stage_configs)
    return config


class LLMTrainingFactory:
    """Project facade for building domain LLM training workflows."""

    DIRS = {
        "raw_data": "data/raw",
        "processed_data": "data/processed",
        "eval": "eval",
        "tokenizer": "tokenizer",
        "experiments": "experiments",
        "models": "models",
        "reports": "reports",
        "configs": "configs",
    }

    def __init__(self, config: Optional[FactoryConfig] = None):
        self.config = config or FactoryConfig()

    @classmethod
    def for_domain(
        cls,
        domain: str,
        root_dir: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        **kwargs,
    ) -> "LLMTrainingFactory":
        return cls(FactoryConfig(
            domain=domain,
            root_dir=root_dir or f"./llm_factory_{domain}",
            base_model=base_model,
            **kwargs,
        ))

    def create_workspace(self) -> Dict[str, str]:
        paths = {}
        os.makedirs(self.config.root_dir, exist_ok=True)
        for name, rel in self.DIRS.items():
            path = os.path.join(self.config.root_dir, rel)
            os.makedirs(path, exist_ok=True)
            paths[name] = path
        self._save_json(os.path.join(self.config.root_dir, "factory_config.json"), self.config.to_dict())
        self.save_blueprint()
        return paths

    def blueprint(self) -> Dict:
        return {
            "name": self.config.name,
            "domain": self.config.domain,
            "objective": "LLM training factory for repeatable domain-model experiments",
            "default_base_model": self.config.base_model,
            "scratch_models": self.config.scratch_models,
            "pipeline": [
                "workspace",
                "dataset_manifest",
                "data_mix",
                "contamination_scan",
                "backend_parallel_plan",
                "tokenizer_train_eval",
                "scratch_pretrain_grid",
                "continued_pretrain",
                "sft",
                "preference_training",
                "rl_scaling",
                "eval_suite",
                "pretrain_report",
                "deploy_or_distill",
            ],
            "artifacts": {
                "dataset_manifest": "normalized data registry and source weights",
                "backend_plan": "distributed backend and parallelism plan",
                "configs": "training configs and experiment manifests",
                "reports": "pretrain reports, eval comparisons, contamination reports",
                "experiments": "scratch pretraining bundles and runs",
                "models": "trained model checkpoints",
            },
            "operating_rule": "dry-run plans first; real training only after data/eval/tokenizer checks pass",
        }

    def save_blueprint(self, path: Optional[str] = None) -> str:
        path = path or os.path.join(self.config.root_dir, "FACTORY_BLUEPRINT.md")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        blueprint = self.blueprint()
        lines = [
            f"# {blueprint['name']}",
            "",
            f"- Domain: `{blueprint['domain']}`",
            f"- Objective: {blueprint['objective']}",
            f"- Default base model: `{blueprint['default_base_model']}`",
            f"- Scratch models: `{', '.join(blueprint['scratch_models'])}`",
            "",
            "## Pipeline",
        ]
        lines.extend(f"{i + 1}. `{step}`" for i, step in enumerate(blueprint["pipeline"]))
        lines.extend(["", "## Operating Rule", "", blueprint["operating_rule"], ""])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self._save_json(os.path.join(self.config.root_dir, "factory_blueprint.json"), blueprint)
        return path

    def plan(self) -> Dict:
        from .DomainBuilder import DomainModelBuilder

        builder = DomainModelBuilder(
            domain=self.config.domain,
            base_model=self.config.base_model,
            output_dir=self.config.root_dir,
            tokenizer_name_or_path=self.config.tokenizer_path,
        )
        return {
            "architecture": builder.architecture_report(),
            "strategy": builder.training_strategy(prefer_scratch=True),
            "data_mix": builder.data_mix_plan(
                output_dir=os.path.join(self.config.root_dir, "data", "processed"),
                max_seq_length=self.config.max_seq_length,
            ),
            "eval_suite": builder.eval_suite(save_path=os.path.join(self.config.root_dir, "configs", "eval_suite.json")),
            "backend_plan_hint": self.backend_plan(model_params=300_000_000, save=False),
            "blueprint": self.blueprint(),
        }

    def import_dataset_manifest(
        self,
        manifest_path: str,
        dataset_dir: Optional[str] = None,
        selected_names: Optional[Sequence[str]] = None,
        role: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Dict:
        """Normalize a dataset registry into SaddleLLM source dictionaries."""
        from ..data.DatasetManifest import DatasetManifest

        self.create_workspace()
        if os.path.basename(manifest_path) == "dataset_info.json":
            manifest = DatasetManifest.from_llamafactory_json(
                manifest_path,
                dataset_dir=dataset_dir,
                selected_names=selected_names,
            )
        else:
            manifest = DatasetManifest.load(manifest_path)
        save_path = save_path or os.path.join(self.config.root_dir, "configs", "dataset_manifest.json")
        manifest.save(save_path)
        payload = manifest.to_dict()
        payload["summary"] = manifest.summary()
        payload["saddle_sources"] = manifest.to_saddle_sources(role=role)
        payload["source_weights"] = manifest.source_weights(role=role)
        payload["saved_to"] = save_path
        return payload

    def backend_plan(
        self,
        model_params: int,
        num_gpus: Optional[int] = None,
        gpu_memory_gb: Optional[float] = None,
        seq_length: Optional[int] = None,
        is_moe: bool = False,
        stage: str = "pretrain",
        save: bool = True,
        path: Optional[str] = None,
    ) -> Dict:
        """Create a backend and parallelism plan for a training stage."""
        from .FactoryBackendPlanner import FactoryBackendPlanner

        plan = FactoryBackendPlanner.recommend_parallelism(
            model_params=model_params,
            num_gpus=num_gpus if num_gpus is not None else self.config.num_gpus,
            gpu_memory_gb=gpu_memory_gb if gpu_memory_gb is not None else self.config.gpu_memory_gb,
            seq_length=seq_length if seq_length is not None else self.config.max_seq_length,
            global_batch_size=self.config.global_batch_size,
            is_moe=is_moe,
            prefer_backend=self.config.prefer_backend,
            training_stage=stage,
        )
        payload = {
            "plan": plan.to_dict(),
            "orchestrator_distributed": FactoryBackendPlanner.to_training_orchestrator_distributed(plan),
            "megatron_args": FactoryBackendPlanner.to_megatron_style_args(plan),
            "colossal_plugin": FactoryBackendPlanner.to_colossal_plugin_spec(plan),
        }
        if save:
            self.create_workspace()
            path = path or os.path.join(self.config.root_dir, "configs", f"{stage}_backend_plan.json")
            FactoryBackendPlanner.save_plan(path, plan)
            payload["saved_to"] = path
        return payload

    def model_blueprints(self, save_path: Optional[str] = None) -> Dict:
        """Generate model-first architecture blueprints for this factory domain."""
        from ..models.ModelBlueprint import ModelBlueprintLab

        blueprints = [
            ModelBlueprintLab.dense_gqa(name=f"{self.config.domain}-dense-gqa-300m"),
            ModelBlueprintLab.deepseek_style_moe(name=f"{self.config.domain}-deepseek-style-moe"),
            ModelBlueprintLab.minimax_style_long_context(name=f"{self.config.domain}-long-context"),
            ModelBlueprintLab.glm_style_reasoning(name=f"{self.config.domain}-reasoning"),
        ]
        comparison = ModelBlueprintLab.compare(blueprints)
        if save_path:
            ModelBlueprintLab.save_comparison(blueprints, save_path)
            comparison["saved_to"] = save_path
        return comparison

    def create_model_experiments(
        self,
        data_sources: Optional[Sequence[Dict]] = None,
        dry_run: bool = True,
        **overrides,
    ) -> Dict:
        """Create controlled model-architecture experiment plans."""
        from ..experiments.ModelExperimentPlanner import ModelExperimentConfig, ModelExperimentPlanner

        self.create_workspace()
        cfg = ModelExperimentConfig(
            domain=overrides.pop("domain", self.config.domain),
            output_dir=overrides.pop("output_dir", os.path.join(self.config.root_dir, "experiments", "model_architectures")),
            tokenizer_path=overrides.pop("tokenizer_path", self.config.tokenizer_path),
            data_sources=list(data_sources or []),
            seq_length=overrides.pop("seq_length", self.config.max_seq_length),
            global_batch_size=overrides.pop("global_batch_size", self.config.global_batch_size),
            num_gpus=overrides.pop("num_gpus", self.config.num_gpus),
            gpu_memory_gb=overrides.pop("gpu_memory_gb", self.config.gpu_memory_gb),
            **overrides,
        )
        bundle = ModelExperimentPlanner(cfg).run(dry_run=dry_run)
        return bundle.to_dict()

    def create_multimodal_plan(
        self,
        data_path: Optional[str] = None,
        image_root: Optional[str] = None,
        save: bool = True,
        **overrides,
    ) -> Dict:
        """Create a low-cost VLM/MLLM training plan."""
        from ..models.ModelBlueprint import ModelBlueprintLab
        from ..multimodal.VisionBackbones import list_vision_backbones

        self.create_workspace()
        output_dir = overrides.pop("output_dir", os.path.join(self.config.root_dir, "experiments", "multimodal"))
        blueprint = ModelBlueprintLab.llava_style_vlm(
            name=overrides.pop("name", f"{self.config.domain}-llava-style-vlm"),
            hidden_size=overrides.pop("hidden_size", 512),
            layers=overrides.pop("layers", 8),
            heads=overrides.pop("heads", 8),
            kv_heads=overrides.pop("kv_heads", 2),
            vocab_size=overrides.pop("vocab_size", 32000),
            seq_length=overrides.pop("seq_length", self.config.max_seq_length),
            vision_backbone=overrides.pop("vision_backbone", "tiny_patch"),
            projector=overrides.pop("projector", "mlp"),
            image_tokens=overrides.pop("image_tokens", 64),
        )
        max_steps = overrides.pop("max_steps", 1000)
        per_device_batch_size = overrides.pop("per_device_batch_size", 1)
        global_batch_size = overrides.pop("global_batch_size", self.config.global_batch_size)
        learning_rate = overrides.pop("learning_rate", 2e-4)
        epochs = overrides.pop("epochs", 1)
        config = _make_training_config(
            project="saddlellm-multimodal-factory",
            experiment=blueprint.name,
            stages=["mllm_sft", "eval"],
            model={
                "name_or_path": self.config.base_model,
                "tokenizer": self.config.tokenizer_path or self.config.base_model,
                "backend": "saddle",
            },
            data={
                "sources": [{"type": "local", "path": data_path, "format": "jsonl"}] if data_path else [],
                "output_dir": os.path.join(output_dir, "processed_data"),
                "max_seq_length": blueprint.max_position_embeddings,
            },
            training={
                "max_steps": max_steps,
                "per_device_batch_size": per_device_batch_size,
                "global_batch_size": global_batch_size,
                "learning_rate": learning_rate,
            },
            distributed=_single_process_config(self.config.num_gpus),
            output_dir=output_dir,
            multimodal={
                "enabled": True,
                "data_path": data_path,
                "image_root": image_root,
                "vision_backbone": blueprint.vision.backbone,
                "projector": blueprint.projector.kind,
                "image_token_id": blueprint.vocab_size,
                "freeze_vision": True,
                "freeze_llm": True,
                "train_projector_only": True,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "per_device_batch_size": per_device_batch_size,
                "max_seq_length": blueprint.max_position_embeddings,
            },
        )
        config["model_blueprint"] = blueprint.to_dict()
        payload = {
            "output_dir": output_dir,
            "blueprint": blueprint.to_dict(),
            "config": config,
            "vision_backbones": list_vision_backbones(),
            "notes": [
                "Plan starts with projector-only warmup.",
                "Use tiny_patch for smoke tests; switch to CLIP/SigLIP/DINOv2 for real VLM work.",
            ],
        }
        if save:
            os.makedirs(output_dir, exist_ok=True)
            payload["blueprint_path"] = blueprint.save(os.path.join(output_dir, "vlm_blueprint.json"))
            payload["config_path"] = self._save_json(os.path.join(output_dir, "config.json"), config)
            payload["plan_path"] = self._save_json(os.path.join(output_dir, "multimodal_plan.json"), payload)
        return payload

    def create_post_training_plan(
        self,
        data_path: str,
        stage: str = "sft",
        method: str = "lora",
        preference_method: str = "dpo",
        save: bool = True,
        **overrides,
    ) -> Dict:
        """Create a traditional text LLM post-training plan."""
        from ..data.PostTrainingData import PostTrainingDataAdapter
        from ..data.TrainingDataInspector import TrainingDataInspector

        self.create_workspace()
        stage = stage.lower()
        if stage in {"dpo", "orpo", "kto"}:
            preference_method = stage
            stage = "preference"
        if stage not in {"sft", "preference", "rlhf"}:
            raise ValueError(f"Unsupported post-training stage: {stage}")
        output_dir = overrides.pop("output_dir", os.path.join(self.config.root_dir, "experiments", f"post_training_{stage}"))
        task = preference_method if stage in {"preference", "rlhf"} else "sft"
        inspection_task = task if stage != "rlhf" else ("dpo" if preference_method == "dpo" else "rlhf")
        data_inspection = TrainingDataInspector.inspect_file(data_path, task=inspection_task).to_dict()
        normalized_path = os.path.join(output_dir, f"{stage}_normalized.jsonl")
        report = None
        if save and data_path and os.path.exists(data_path):
            report = PostTrainingDataAdapter.normalize_file(data_path, normalized_path, task=task)

        experiment = overrides.pop(
            "experiment",
            f"{self.config.domain}-{stage}-{preference_method if stage == 'preference' else method}",
        )
        base_model = overrides.pop("base_model", self.config.base_model)
        max_seq_length = overrides.pop("max_seq_length", self.config.max_seq_length)
        max_steps = overrides.pop("max_steps", 1000)
        per_device_batch_size = overrides.pop("per_device_batch_size", 1)
        global_batch_size = overrides.pop("global_batch_size", self.config.global_batch_size)
        learning_rate = overrides.pop("learning_rate", 2e-4 if stage == "sft" else 5e-6)
        epochs = overrides.pop("epochs", 3 if stage == "sft" else 1)
        dry_run = overrides.pop("dry_run", False)
        preflight_only = overrides.pop("preflight_only", False)
        lora_r = overrides.pop("lora_r", 16)
        lora_alpha = overrides.pop("lora_alpha", 32)
        lora_dropout = overrides.pop("lora_dropout", 0.05)
        target_modules = overrides.pop("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        beta = overrides.pop("beta", 0.1)
        training_data_path = normalized_path if report else data_path
        use_lora = method in {"lora", "qlora"}
        use_qlora = method == "qlora"

        stage_config = {
            "enabled": True,
            "data_path": training_data_path,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "per_device_batch_size": per_device_batch_size,
            "use_lora": use_lora,
            "use_qlora": use_qlora,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 0,
            "save_steps": 200,
        }
        if stage == "sft":
            stage_config.update(
                {
                    "max_seq_length": max_seq_length,
                    "lora": {
                        "r": lora_r,
                        "alpha": lora_alpha,
                        "dropout": lora_dropout,
                        "target_modules": target_modules,
                    },
                }
            )
        else:
            stage_config.update(
                {
                    "method": preference_method,
                    "beta": beta,
                    "max_length": max_seq_length,
                    "max_prompt_length": max(128, max_seq_length // 2),
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                }
            )

        config = _make_training_config(
            project="saddlellm-post-training-factory",
            experiment=experiment,
            stages=[stage, "eval"],
            model={
                "name_or_path": base_model,
                "tokenizer": self.config.tokenizer_path or base_model,
                "backend": "hf",
            },
            data={
                "sources": [{"type": "local", "path": training_data_path, "format": "jsonl"}],
                "output_dir": os.path.join(output_dir, "processed_data"),
                "max_seq_length": max_seq_length,
            },
            training={
                "max_steps": max_steps,
                "per_device_batch_size": per_device_batch_size,
                "global_batch_size": global_batch_size,
                "learning_rate": learning_rate,
                "warmup_steps": 0,
                "dry_run": dry_run,
                "preflight_only": preflight_only,
            },
            distributed=_single_process_config(self.config.num_gpus),
            output_dir=output_dir,
            **{stage: stage_config},
        )
        payload = {
            "output_dir": output_dir,
            "stage": stage,
            "method": preference_method if stage == "preference" else method,
            "config": config,
            "data_inspection": data_inspection,
            "normalized_data_path": normalized_path if report else None,
            "normalization_report": report.to_dict() if report else None,
            "notes": [
                "Traditional text LLM post-training plan.",
                "Data is normalized through PostTrainingDataAdapter before trainer use when save=True and the source file exists.",
            ],
        }
        if save:
            os.makedirs(output_dir, exist_ok=True)
            payload["config_path"] = self._save_json(os.path.join(output_dir, "config.json"), config)
            payload["plan_path"] = self._save_json(os.path.join(output_dir, "post_training_plan.json"), payload)
        return payload

    def create_preflight_plan(
        self,
        data_path: str,
        stage: str = "sft",
        method: str = "lora",
        preference_method: str = "dpo",
        save: bool = True,
        **overrides,
    ) -> Dict:
        """Create a no-training post-training preflight plan."""
        output_dir = overrides.pop(
            "output_dir",
            os.path.join(self.config.root_dir, "experiments", f"preflight_{stage}"),
        )
        payload = self.create_post_training_plan(
            data_path=data_path,
            stage=stage,
            method=method,
            preference_method=preference_method,
            save=False,
            output_dir=output_dir,
            dry_run=True,
            preflight_only=True,
            **overrides,
        )
        payload["preflight_only"] = True
        payload["ready"] = bool(payload.get("data_inspection", {}).get("ready", False))
        payload["blocking_errors"] = list(payload.get("data_inspection", {}).get("errors", []))
        payload["recommendations"] = list(payload.get("data_inspection", {}).get("recommendations", []))
        payload["training_estimate"] = self._estimate_from_config(
            payload["config"], stage=payload["stage"]
        )
        if save:
            os.makedirs(output_dir, exist_ok=True)
            payload["config_path"] = self._save_json(
                os.path.join(output_dir, "config.json"),
                payload["config"],
            )
            payload["plan_path"] = self._save_json(os.path.join(output_dir, "preflight_plan.json"), payload)
        return payload

    def _estimate_from_config(self, config: Dict, stage: str) -> Dict:
        from ..training.TrainingPlanEstimator import TrainingPlanEstimator

        training = config.get("training", {})
        distributed = config.get("distributed", {})
        stage_cfg = config.get(stage, {})
        if stage == "rlhf":
            stage_cfg = config.get("rlhf", {})
            max_seq_length = stage_cfg.get("max_length", config.get("data", {}).get("max_seq_length", self.config.max_seq_length))
        elif stage == "preference":
            stage_cfg = config.get("preference", {})
            max_seq_length = stage_cfg.get("max_length", config.get("data", {}).get("max_seq_length", self.config.max_seq_length))
        else:
            max_seq_length = stage_cfg.get("max_seq_length", config.get("data", {}).get("max_seq_length", self.config.max_seq_length))
        return TrainingPlanEstimator.estimate(
            per_device_batch_size=stage_cfg.get("per_device_batch_size", training.get("per_device_batch_size", 1)),
            gradient_accumulation_steps=stage_cfg.get(
                "gradient_accumulation_steps",
                distributed.get("gradient_accumulation_steps", 1),
            ),
            num_gpus=distributed.get("num_gpus", self.config.num_gpus),
            max_seq_length=max_seq_length,
            max_steps=training.get("max_steps", 0),
            epochs=stage_cfg.get("epochs", training.get("epochs")),
        ).to_dict()

    def create_mopd_plan(
        self,
        prompts_path: str,
        teachers: Optional[Sequence[Dict]] = None,
        chain_sft: bool = True,
        save: bool = True,
        **overrides,
    ) -> Dict:
        """Create an on-policy multi-teacher distillation plan.

        When chain_sft=True, the stages are mopd -> sft -> eval and SFT
        consumes the MOPD output JSONL produced by the preceding stage.
        """
        self.create_workspace()
        output_dir = overrides.pop("output_dir", os.path.join(self.config.root_dir, "experiments", "mopd"))
        stages = ["mopd", "sft", "eval"] if chain_sft else ["mopd", "eval"]
        experiment = overrides.pop("experiment", f"{self.config.domain}-mopd")
        base_model = overrides.pop("base_model", self.config.base_model)
        method = overrides.pop("method", "lora")
        lora_r = overrides.pop("lora_r", 16)
        lora_alpha = overrides.pop("lora_alpha", 32)
        lora_dropout = overrides.pop("lora_dropout", 0.05)
        max_seq_length = overrides.pop("max_seq_length", self.config.max_seq_length)
        max_steps = overrides.pop("max_steps", 1000)
        per_device_batch_size = overrides.pop("per_device_batch_size", 1)
        global_batch_size = overrides.pop("global_batch_size", self.config.global_batch_size)
        learning_rate = overrides.pop("learning_rate", 2e-4)
        epochs = overrides.pop("epochs", 1)
        config = _make_training_config(
            project="saddlellm-mopd-factory",
            experiment=experiment,
            stages=stages,
            model={
                "name_or_path": base_model,
                "tokenizer": self.config.tokenizer_path or base_model,
                "backend": "hf",
            },
            data={
                "sources": [{"type": "local", "path": prompts_path, "format": "jsonl"}] if prompts_path else [],
                "output_dir": os.path.join(output_dir, "processed_data"),
                "max_seq_length": max_seq_length,
            },
            training={
                "max_steps": max_steps,
                "per_device_batch_size": per_device_batch_size,
                "global_batch_size": global_batch_size,
                "learning_rate": learning_rate,
            },
            distributed=_single_process_config(self.config.num_gpus),
            output_dir=output_dir,
            mopd={
                "enabled": True,
                "prompts_path": prompts_path,
                "output_dir": os.path.join(output_dir, "mopd"),
                "teachers": list(teachers or []),
                "num_rollouts_per_prompt": overrides.pop("num_rollouts_per_prompt", 1),
                "max_new_tokens": overrides.pop("max_new_tokens", 1024),
                "student_temperature": overrides.pop("student_temperature", 0.9),
                "teacher_temperature": overrides.pop("teacher_temperature", 0.3),
                "aggregation": overrides.pop("aggregation", "best_score"),
                "dry_run": overrides.pop("dry_run", True),
            },
        )
        if chain_sft:
            config["sft"] = {
                "enabled": True,
                "data_path": "",
                "use_lora": method in {"lora", "qlora"},
                "use_qlora": method == "qlora",
                "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": lora_dropout},
                "epochs": epochs,
                "learning_rate": learning_rate,
                "per_device_batch_size": per_device_batch_size,
                "max_seq_length": max_seq_length,
            }
        payload = {
            "output_dir": output_dir,
            "stages": stages,
            "config": config,
            "notes": [
                "MOPD collects teacher supervision on student-generated rollouts.",
                "When chain_sft=True, SFT consumes the MOPD JSONL output from the prior stage.",
            ],
        }
        if save:
            os.makedirs(output_dir, exist_ok=True)
            payload["config_path"] = self._save_json(os.path.join(output_dir, "config.json"), config)
            payload["plan_path"] = self._save_json(os.path.join(output_dir, "mopd_plan.json"), payload)
        return payload

    def create_vla_plan(
        self,
        data_path: str,
        image_root: Optional[str] = None,
        save: bool = True,
        **overrides,
    ) -> Dict:
        """Create a vision-language-action behavior-cloning plan.

        The normalized output is compatible with the existing multimodal SFT
        planning path: messages contain the language instruction and assistant
        action tokens, while images/proprio/action metadata remain available for
        VLA-specific collators later.
        """
        from ..multimodal.VLA import VLAActionSpace, VLADataAdapter, VLATrainingPlanner

        self.create_workspace()
        output_dir = overrides.pop("output_dir", os.path.join(self.config.root_dir, "experiments", "vla"))
        action_space = VLAActionSpace(
            action_dim=overrides.pop("action_dim", 7),
            action_type=overrides.pop("action_type", "continuous"),
            bins=overrides.pop("action_bins", 256),
            min_value=overrides.pop("action_min_value", -1.0),
            max_value=overrides.pop("action_max_value", 1.0),
            include_gripper=overrides.pop("include_gripper", True),
            control_hz=overrides.pop("control_hz", 10.0),
            coordinate_frame=overrides.pop("coordinate_frame", "end_effector_delta"),
        )
        normalized_path = os.path.join(output_dir, "vla_normalized.jsonl")
        report = None
        if save and data_path and os.path.exists(data_path):
            report = VLADataAdapter.normalize_file(
                data_path,
                normalized_path,
                image_root=image_root,
                action_space=action_space,
            )
        image_token_id = overrides.pop("image_token_id", None)
        vision_backbone = overrides.pop("vision_backbone", "tiny_patch")
        projector = overrides.pop("projector", "mlp")
        freeze_vision = overrides.pop("freeze_vision", True)
        freeze_llm = overrides.pop("freeze_llm", False)
        train_projector_only = overrides.pop("train_projector_only", False)
        train_vla = overrides.pop("train", False)
        use_lora = overrides.pop("use_lora", True)
        use_qlora = overrides.pop("use_qlora", False)
        lora_r = overrides.pop("lora_r", 16)
        lora_alpha = overrides.pop("lora_alpha", 32)
        lora_dropout = overrides.pop("lora_dropout", 0.05)
        gradient_accumulation_steps = overrides.pop("gradient_accumulation_steps", 4)
        validation_split = overrides.pop("validation_split", 0.0)
        train_on_prompt = overrides.pop("train_on_prompt", False)

        experiment = overrides.pop("experiment", f"{self.config.domain}-vla-sft")
        base_model = overrides.pop("base_model", self.config.base_model)
        max_seq_length = overrides.pop("max_seq_length", self.config.max_seq_length)
        max_steps = overrides.pop("max_steps", 1000)
        per_device_batch_size = overrides.pop("per_device_batch_size", 1)
        global_batch_size = overrides.pop("global_batch_size", self.config.global_batch_size)
        learning_rate = overrides.pop("learning_rate", 2e-4)
        epochs = overrides.pop("epochs", 1)
        training_data_path = normalized_path if report else data_path
        multimodal_config = {
            "enabled": True,
            "data_path": training_data_path,
            "image_root": image_root,
            "image_token": "<image>",
            "image_token_id": image_token_id,
            "vision_backbone": vision_backbone,
            "projector": projector,
            "freeze_vision": freeze_vision,
            "freeze_llm": freeze_llm,
            "train_projector_only": train_projector_only,
        }
        vla_config = {
            "enabled": True,
            "data_path": training_data_path,
            "image_root": image_root,
            "output_dir": os.path.join(output_dir, "vla"),
            "action_space": {
                "action_dim": action_space.action_dim,
                "action_type": action_space.action_type,
                "bins": action_space.bins,
                "min_value": action_space.min_value,
                "max_value": action_space.max_value,
                "include_gripper": action_space.include_gripper,
                "control_hz": action_space.control_hz,
                "coordinate_frame": action_space.coordinate_frame,
            },
            "vision_backbone": vision_backbone,
            "projector": projector,
            "freeze_vision": freeze_vision,
            "freeze_llm": freeze_llm,
            "train_projector_only": train_projector_only,
            "train": train_vla,
            "use_lora": use_lora,
            "use_qlora": use_qlora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "validation_split": validation_split,
            "max_seq_length": max_seq_length,
            "train_on_prompt": train_on_prompt,
        }
        config = _make_training_config(
            project="saddlellm-vla-factory",
            experiment=experiment,
            stages=["vla_sft", "eval"],
            model={
                "name_or_path": base_model,
                "tokenizer": self.config.tokenizer_path or base_model,
                "backend": "hf",
            },
            data={
                "sources": [{"type": "local", "path": training_data_path, "format": "jsonl"}],
                "output_dir": os.path.join(output_dir, "processed_data"),
                "max_seq_length": max_seq_length,
            },
            training={
                "max_steps": max_steps,
                "per_device_batch_size": per_device_batch_size,
                "global_batch_size": global_batch_size,
                "learning_rate": learning_rate,
            },
            distributed=_single_process_config(self.config.num_gpus),
            output_dir=output_dir,
            multimodal=multimodal_config,
            vla=vla_config,
        )
        vla_plan = VLATrainingPlanner.create_plan(
            data_path=normalized_path if report else data_path,
            image_root=image_root,
            action_space=action_space,
            output_dir=output_dir,
        )
        payload = {
            "output_dir": output_dir,
            "stage": "vla_sft",
            "config": config,
            "vla_plan": vla_plan,
            "normalized_data_path": normalized_path if report else None,
            "normalization_report": report.to_dict() if report else None,
            "notes": [
                "VLA behavior cloning is represented as multimodal SFT over action tokens.",
                "Online/simulator RL should consume the same action_space metadata.",
            ],
        }
        if save:
            os.makedirs(output_dir, exist_ok=True)
            payload["config_path"] = self._save_json(os.path.join(output_dir, "config.json"), config)
            payload["vla_plan_path"] = self._save_json(os.path.join(output_dir, "vla_plan.json"), vla_plan)
            payload["plan_path"] = self._save_json(os.path.join(output_dir, "vla_factory_plan.json"), payload)
        return payload

    def save_plan(self, path: Optional[str] = None) -> str:
        self.create_workspace()
        path = path or os.path.join(self.config.root_dir, "configs", "factory_plan.json")
        return self._save_json(path, self.plan())

    def create_pretrain_experiments(
        self,
        corpus_sources: Optional[Sequence[Dict]] = None,
        sources_by_bucket: Optional[Dict[str, Sequence[Dict]]] = None,
        dry_run: bool = True,
        **overrides,
    ) -> Dict:
        from ..experiments.PretrainExperimentRunner import PretrainExperimentConfig, PretrainExperimentRunner

        self.create_workspace()
        cfg = PretrainExperimentConfig(
            domain=overrides.pop("domain", self.config.domain),
            output_dir=overrides.pop("output_dir", os.path.join(self.config.root_dir, "experiments", "scratch_pretrain")),
            model_names=overrides.pop("model_names", self.config.scratch_models),
            token_multipliers=overrides.pop("token_multipliers", self.config.token_multipliers),
            tokenizer_path=overrides.pop("tokenizer_path", self.config.tokenizer_path),
            corpus_sources=list(corpus_sources or []),
            sources_by_bucket=sources_by_bucket,
            global_batch_size=overrides.pop("global_batch_size", self.config.global_batch_size),
            seq_length=overrides.pop("seq_length", self.config.max_seq_length),
            **overrides,
        )
        bundle = PretrainExperimentRunner(cfg).run(dry_run=dry_run)
        return bundle.to_dict()

    def report(self, output_dir: Optional[str] = None) -> Dict:
        from ..experiments.PretrainReport import PretrainReport

        target = output_dir or self.config.root_dir
        result = PretrainReport.generate(target)
        report_dir = os.path.join(self.config.root_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        self._save_json(os.path.join(report_dir, "latest_pretrain_report.json"), result.to_dict())
        return result.to_dict()

    def status(self) -> Dict:
        root = self.config.root_dir
        directories = {
            name: os.path.isdir(os.path.join(root, rel))
            for name, rel in self.DIRS.items()
        }
        artifacts = {
            "factory_config": os.path.exists(os.path.join(root, "factory_config.json")),
            "blueprint": os.path.exists(os.path.join(root, "FACTORY_BLUEPRINT.md")),
            "plan": os.path.exists(os.path.join(root, "configs", "factory_plan.json")),
            "scratch_manifest": os.path.exists(os.path.join(root, "experiments", "scratch_pretrain", "manifest.json")),
            "latest_report": os.path.exists(os.path.join(root, "reports", "latest_pretrain_report.json")),
        }
        manifest = os.path.join(root, "experiments", "scratch_pretrain", "manifest.json")
        report = os.path.join(root, "reports", "latest_pretrain_report.json")
        return FactoryStatus(
            root_dir=root,
            exists=os.path.isdir(root),
            directories=directories,
            artifacts=artifacts,
            latest_manifest=manifest if os.path.exists(manifest) else None,
            latest_report=report if os.path.exists(report) else None,
        ).to_dict()

    def _save_json(self, path: str, data: Dict) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
