"""Plan controlled model-architecture experiments.

This planner creates comparable dry-run artifacts for model-first research:
same data, tokenizer, sequence length, batch budget, and train steps; different
architecture choices such as Dense GQA, MLA, MoE, MTP, and long-context.
"""
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class ModelExperimentConfig:
    domain: str = "research"
    output_dir: str = "./model_experiments"
    tokenizer_path: Optional[str] = None
    data_sources: List[Dict] = field(default_factory=list)
    source_weights: Optional[List[float]] = None
    seq_length: int = 1024
    max_steps: int = 1000
    global_batch_size: int = 64
    per_device_batch_size: int = 1
    learning_rate: float = 3e-4
    num_gpus: int = 1
    gpu_memory_gb: float = 24.0
    include_long_context: bool = False
    include_moe: bool = True
    include_mla: bool = True
    include_mtp: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelExperimentRun:
    run_id: str
    blueprint_name: str
    family: str
    output_dir: str
    blueprint_path: str
    recipe_path: str
    orchestrator_config_path: str
    analysis: Dict
    backend_plan: Dict = field(default_factory=dict)
    score: Dict = field(default_factory=dict)
    status: str = "planned"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["name"] = self.blueprint_name
        return data


@dataclass
class ModelExperimentBundle:
    output_dir: str
    config: Dict
    comparison_path: str
    manifest_path: str
    summary_path: str
    runs: List[ModelExperimentRun]

    def to_dict(self) -> Dict:
        return {
            "output_dir": self.output_dir,
            "config": self.config,
            "comparison_path": self.comparison_path,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "runs": [run.to_dict() for run in self.runs],
        }


class ModelExperimentPlanner:
    """Build dry-run bundles for architecture comparisons."""

    def __init__(self, config: Optional[ModelExperimentConfig] = None, blueprints: Optional[Sequence] = None):
        self.config = config or ModelExperimentConfig()
        self.blueprints = list(blueprints) if blueprints is not None else self.default_blueprints()

    def default_blueprints(self):
        from .ModelBlueprint import (
            AttentionBlueprint,
            FFNBlueprint,
            ModelBlueprint,
            ModelBlueprintLab,
            ObjectiveBlueprint,
        )

        cfg = self.config
        hidden = 512
        layers = 8
        heads = 8
        kv_heads = 2
        blueprints = [
            ModelBlueprintLab.dense_gqa(
                name=f"{cfg.domain}-dense-gqa-control",
                hidden_size=hidden,
                layers=layers,
                heads=heads,
                kv_heads=kv_heads,
                seq_length=cfg.seq_length,
            )
        ]
        if cfg.include_mla:
            blueprints.append(ModelBlueprint(
                name=f"{cfg.domain}-mla-latent",
                family="deepseek",
                hidden_size=hidden,
                num_layers=layers,
                max_position_embeddings=cfg.seq_length,
                attention=AttentionBlueprint(
                    kind="mla",
                    backend="sdpa",
                    mla_cache_mode="latent",
                    num_heads=heads,
                    num_kv_heads=kv_heads,
                    q_lora_rank=64,
                    kv_lora_rank=32,
                    rope_theta=1000000.0,
                ),
                ffn=FFNBlueprint(kind="swiglu", intermediate_size=hidden * 4),
                notes=["MLA latent-cache dense variant matched to dense control size range."],
            ))
        if cfg.include_moe:
            blueprints.append(ModelBlueprint(
                name=f"{cfg.domain}-moe-top2",
                family="deepseek",
                hidden_size=hidden,
                num_layers=layers,
                max_position_embeddings=cfg.seq_length,
                attention=AttentionBlueprint(kind="gqa", num_heads=heads, num_kv_heads=kv_heads),
                ffn=FFNBlueprint(
                    kind="moe",
                    intermediate_size=hidden * 4,
                    num_experts=8,
                    experts_per_token=2,
                    expert_intermediate_size=hidden * 2,
                    shared_expert=True,
                    aux_loss_free=True,
                    router_aux_loss_coef=0.0,
                ),
                notes=["Sparse top-2 MoE to compare active-parameter quality against dense control."],
            ))
        if cfg.include_mtp:
            blueprints.append(ModelBlueprint(
                name=f"{cfg.domain}-dense-gqa-mtp",
                family="qwen",
                hidden_size=hidden,
                num_layers=layers,
                max_position_embeddings=cfg.seq_length,
                attention=AttentionBlueprint(kind="gqa", num_heads=heads, num_kv_heads=kv_heads),
                ffn=FFNBlueprint(kind="swiglu", intermediate_size=hidden * 4),
                objective=ObjectiveBlueprint(multi_token_prediction=True, mtp_extra_tokens=2),
                notes=["Dense control plus MTP objective for sample-efficiency comparison."],
            ))
        if cfg.include_long_context:
            blueprints.append(ModelBlueprintLab.minimax_style_long_context(
                name=f"{cfg.domain}-long-context",
                hidden_size=hidden,
                layers=layers,
                heads=heads,
                seq_length=max(cfg.seq_length, 32768),
            ))
        return blueprints

    def build_bundle(self) -> ModelExperimentBundle:
        from .FactoryBackendPlanner import FactoryBackendPlanner
        from .ModelBlueprint import ModelBlueprintLab
        from .TrainingRecipe import TrainingRecipe

        os.makedirs(self.config.output_dir, exist_ok=True)
        config_path = self._save_json(os.path.join(self.config.output_dir, "experiment_config.json"), self.config.to_dict())
        comparison_path = ModelBlueprintLab.save_comparison(
            self.blueprints,
            os.path.join(self.config.output_dir, "blueprint_comparison.json"),
        )

        runs: List[ModelExperimentRun] = []
        for idx, bp in enumerate(self.blueprints):
            run_id = f"{idx:03d}-{bp.name}"
            run_dir = os.path.join(self.config.output_dir, "runs", run_id)
            os.makedirs(run_dir, exist_ok=True)
            blueprint_path = bp.save(os.path.join(run_dir, "blueprint.json"))

            recipe = self._recipe_for_blueprint(bp, run_dir)
            recipe_path = recipe.save(os.path.join(run_dir, "recipe.yaml"))
            orchestrator_cfg = recipe.compile(base_dir=run_dir, save_backend_plan=False)
            orchestrator_cfg["model"]["blueprint"] = bp.to_config_dict()
            orchestrator_cfg["data"]["source_weights"] = self.config.source_weights
            if not self.config.data_sources:
                orchestrator_cfg["stages"] = ["eval"]
                orchestrator_cfg.setdefault("notes", []).append("No data_sources supplied; architecture run is dry-run/planning only.")

            backend_plan = FactoryBackendPlanner.recommend_parallelism(
                model_params=bp.estimate_total_params(),
                num_gpus=self.config.num_gpus,
                gpu_memory_gb=self.config.gpu_memory_gb,
                seq_length=self.config.seq_length,
                global_batch_size=self.config.global_batch_size,
                is_moe="moe" in bp.architecture_axes().get("ffn_kinds", []),
                training_stage="pretrain",
            )
            orchestrator_cfg["distributed"].update(FactoryBackendPlanner.to_training_orchestrator_distributed(backend_plan))
            orchestrator_cfg["backend_plan"] = backend_plan.to_dict()

            orchestrator_config_path = self._save_json(os.path.join(run_dir, "orchestrator_config.json"), orchestrator_cfg)
            self._save_json(os.path.join(run_dir, "backend_plan.json"), {
                "plan": backend_plan.to_dict(),
                "orchestrator_distributed": FactoryBackendPlanner.to_training_orchestrator_distributed(backend_plan),
                "megatron_args": FactoryBackendPlanner.to_megatron_style_args(backend_plan),
                "colossal_plugin": FactoryBackendPlanner.to_colossal_plugin_spec(backend_plan),
            })
            analysis = bp.analyze()
            analysis["experiment_cost_profile"] = bp.cost_profile(
                token_budget=self._experiment_token_budget(),
                batch_size=self.config.global_batch_size,
                sequence_length=self.config.seq_length,
            )
            score = self._score_blueprint(bp, analysis, backend_plan.to_dict())
            runs.append(ModelExperimentRun(
                run_id=run_id,
                blueprint_name=bp.name,
                family=bp.family,
                output_dir=run_dir,
                blueprint_path=blueprint_path,
                recipe_path=recipe_path,
                orchestrator_config_path=orchestrator_config_path,
                analysis=analysis,
                backend_plan=backend_plan.to_dict(),
                score=score,
            ))

        summary_path = self._save_json(
            os.path.join(self.config.output_dir, "experiment_summary.json"),
            self._build_summary(runs),
        )
        manifest_path = self._save_json(os.path.join(self.config.output_dir, "manifest.json"), {
            "config_path": config_path,
            "comparison_path": comparison_path,
            "summary_path": summary_path,
            "runs": [run.to_dict() for run in runs],
        })
        return ModelExperimentBundle(
            output_dir=self.config.output_dir,
            config=self.config.to_dict(),
            comparison_path=comparison_path,
            manifest_path=manifest_path,
            summary_path=summary_path,
            runs=runs,
        )

    def run(self, dry_run: bool = True, limit: Optional[int] = None) -> ModelExperimentBundle:
        bundle = self.build_bundle()
        if dry_run:
            return bundle
        from .PretrainReport import PretrainReport
        from .TrainingOrchestrator import TrainingOrchestrator

        for run in bundle.runs[: limit or len(bundle.runs)]:
            with open(run.orchestrator_config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if not cfg.get("data", {}).get("sources") and "pretrain" in cfg.get("stages", []):
                run.status = "skipped_no_data"
                continue
            TrainingOrchestrator.from_dict(cfg).run()
            PretrainReport.generate(run.output_dir)
            run.status = "completed"
        self._save_json(bundle.manifest_path, bundle.to_dict())
        return bundle

    def _recipe_for_blueprint(self, bp, run_dir: str):
        from .TrainingRecipe import (
            RecipeBackendConfig,
            RecipeDataConfig,
            RecipeEvalConfig,
            RecipeLoggingConfig,
            RecipeModelConfig,
            RecipeTrainingConfig,
            TrainingRecipe,
        )

        return TrainingRecipe(
            project="saddlellm-model-experiment",
            experiment=bp.name,
            stage=["pretrain", "eval"],
            model=RecipeModelConfig(
                config="llama-tiny-150m",
                tokenizer=self.config.tokenizer_path,
                backend="saddle",
                pretrain_mode="scratch",
                params=bp.estimate_total_params(),
                is_moe="moe" in bp.architecture_axes().get("ffn_kinds", []),
            ),
            data=RecipeDataConfig(
                sources=list(self.config.data_sources),
                source_weights=self.config.source_weights,
                output_dir=os.path.join(run_dir, "processed_data"),
                max_seq_length=self.config.seq_length,
            ),
            backend=RecipeBackendConfig(
                strategy="auto",
                num_gpus=self.config.num_gpus,
                gpu_memory_gb=self.config.gpu_memory_gb,
            ),
            training=RecipeTrainingConfig(
                max_steps=self.config.max_steps,
                per_device_batch_size=self.config.per_device_batch_size,
                global_batch_size=self.config.global_batch_size,
                learning_rate=self.config.learning_rate,
                log_every_steps=max(1, min(50, self.config.max_steps)),
                save_every_steps=max(1, self.config.max_steps),
                eval_every_steps=max(1, self.config.max_steps),
            ),
            eval=RecipeEvalConfig(enabled=True, max_samples=100),
            logging=RecipeLoggingConfig(output_dir=run_dir, backend="local"),
        )

    def _experiment_token_budget(self) -> int:
        return int(max(1, self.config.max_steps) * max(1, self.config.global_batch_size) * max(1, self.config.seq_length))

    def _score_blueprint(self, bp, analysis: Dict, backend_plan: Dict) -> Dict:
        cost = analysis.get("experiment_cost_profile", analysis.get("cost_profile", {}))
        axes = analysis.get("architecture_axes", {})
        risk_count = len(analysis.get("risk_report", []))
        warnings = backend_plan.get("warnings", [])
        memory_used = (
            backend_plan.get("estimated_model_state_gb_per_gpu", 0.0)
            + backend_plan.get("estimated_activation_gb_per_gpu", 0.0)
        )
        memory_headroom = self.config.gpu_memory_gb - memory_used

        innovation = 0.0
        if axes.get("attention") == "mla":
            innovation += 18.0
        if axes.get("ffn") == "moe":
            innovation += 18.0
        if axes.get("objective") == "mtp":
            innovation += 12.0
        if axes.get("context") == "long":
            innovation += 10.0

        cost_penalty = min(35.0, cost.get("training_peta_flops", 0.0) * 0.02)
        cache_penalty = min(20.0, cost.get("kv_cache_gb", 0.0) * 0.35)
        warning_penalty = len(warnings) * 8.0
        risk_penalty = risk_count * 4.0
        memory_bonus = max(-20.0, min(15.0, memory_headroom * 1.5))

        feasibility = max(0.0, min(100.0, 82.0 + memory_bonus - warning_penalty - risk_penalty))
        smoke_priority = max(0.0, min(100.0, feasibility - innovation * 0.35 - cache_penalty))
        research_priority = max(0.0, min(100.0, feasibility * 0.55 + innovation - cost_penalty - cache_penalty * 0.5))
        if axes.get("attention") == "gqa" and axes.get("ffn") == "dense" and axes.get("objective") == "next_token":
            smoke_priority = min(100.0, smoke_priority + 15.0)

        if smoke_priority >= 75:
            recommendation = "smoke_first"
        elif research_priority >= 70:
            recommendation = "research_arm"
        elif feasibility < 45:
            recommendation = "defer_until_capacity"
        else:
            recommendation = "compare_after_control"

        return {
            "feasibility": round(feasibility, 2),
            "smoke_priority": round(smoke_priority, 2),
            "research_priority": round(research_priority, 2),
            "memory_headroom_gb": round(memory_headroom, 2),
            "innovation_points": round(innovation, 2),
            "cost_penalty": round(cost_penalty, 2),
            "cache_penalty": round(cache_penalty, 2),
            "recommendation": recommendation,
        }

    def _build_summary(self, runs: List[ModelExperimentRun]) -> Dict:
        from .OperatorBackends import list_attention_backends

        smoke_queue = sorted(runs, key=lambda run: run.score.get("smoke_priority", 0.0), reverse=True)
        research_queue = sorted(runs, key=lambda run: run.score.get("research_priority", 0.0), reverse=True)
        cheapest_queue = sorted(
            runs,
            key=lambda run: run.analysis.get("experiment_cost_profile", {}).get("training_peta_flops", 0.0),
        )
        warnings = []
        for run in runs:
            for warning in run.backend_plan.get("warnings", []):
                warnings.append({"run_id": run.run_id, "name": run.blueprint_name, "warning": warning})

        return {
            "domain": self.config.domain,
            "token_budget": self._experiment_token_budget(),
            "num_runs": len(runs),
            "recommended_smoke_order": [self._summary_item(run) for run in smoke_queue],
            "recommended_research_order": [self._summary_item(run) for run in research_queue],
            "lowest_cost_order": [self._summary_item(run) for run in cheapest_queue],
            "operator_backends": list_attention_backends(),
            "warnings": warnings,
            "next_actions": self._next_actions(smoke_queue, research_queue),
        }

    def _summary_item(self, run: ModelExperimentRun) -> Dict:
        cost = run.analysis.get("experiment_cost_profile", {})
        return {
            "run_id": run.run_id,
            "name": run.blueprint_name,
            "family": run.family,
            "recommendation": run.score.get("recommendation"),
            "smoke_priority": run.score.get("smoke_priority"),
            "research_priority": run.score.get("research_priority"),
            "training_peta_flops": round(cost.get("training_peta_flops", 0.0), 4),
            "kv_cache_gb": cost.get("kv_cache_gb", 0.0),
            "active_params_human": run.analysis.get("active_params_human"),
        }

    def _next_actions(self, smoke_queue: List[ModelExperimentRun], research_queue: List[ModelExperimentRun]) -> List[str]:
        actions = []
        if smoke_queue:
            actions.append(f"Start with {smoke_queue[0].blueprint_name} to validate tokenizer, data packing, loss, and checkpoint save/load.")
        if len(smoke_queue) > 1:
            actions.append(f"Then run {smoke_queue[1].blueprint_name} at the same token budget to establish a controlled delta.")
        if research_queue:
            actions.append(f"Use {research_queue[0].blueprint_name} as the first research bet after the dense control is stable.")
        actions.append("Do not scale context length or MoE experts until the dense control run has clean loss curves and evaluation reports.")
        return actions

    def _save_json(self, path: str, data: Dict) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
