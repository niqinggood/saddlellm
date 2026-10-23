"""Static validation for SaddleLLM training configs."""
from dataclasses import asdict, dataclass, field
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from ..framework.stages import BUILTIN_STAGE_NAMES


# Backward-compatible constant for callers that only need built-in names.
# Validation itself consults the live registry so installed plugins work.
VALID_STAGES = set(BUILTIN_STAGE_NAMES)


@dataclass
class TrainingConfigValidation:
    valid: bool
    stages: List[str]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_inspections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    training_estimates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    architecture_analysis: Dict[str, Any] = field(default_factory=dict)
    normalized_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingConfigValidator:
    """Validate a compiled TrainingOrchestrator config without starting training."""

    @classmethod
    def validate(
        cls,
        raw: Dict[str, Any],
        inspect_data: bool = True,
        *,
        stage_registry=None,
    ) -> TrainingConfigValidation:
        from ..agents.RealWorldOperators import _json_safe
        from .TrainingOrchestrator import TrainingOrchestrator
        from ..framework.stages import get_stage_registry

        issues: List[str] = []
        warnings: List[str] = []
        data_inspections: Dict[str, Dict[str, Any]] = {}
        training_estimates: Dict[str, Dict[str, Any]] = {}
        architecture_analysis: Dict[str, Any] = {}
        safe_raw = _json_safe(raw)
        if not isinstance(safe_raw, dict):
            safe_raw = {}

        try:
            cfg = TrainingOrchestrator._parse_config(raw)
        except Exception as exc:
            return TrainingConfigValidation(
                valid=False,
                stages=list(raw.get("stages", [])) if isinstance(raw.get("stages"), list) else [],
                issues=[f"Config parse failed: {exc}"],
                normalized_config=safe_raw,
            )

        stages = list(cfg.stages or [])
        try:
            from ..framework import PipelinePlan

            pipeline_plan = PipelinePlan.compile(
                stages, dependencies=cfg.pipeline.dependencies
            )
            stages = list(pipeline_plan.stages)
            cfg.stages = stages
        except Exception as exc:
            issues.append(f"Pipeline plan is invalid: {exc}")
        registry = stage_registry or get_stage_registry()
        available_stages = set(registry.names())
        unknown = [stage for stage in stages if stage not in available_stages]
        if unknown:
            issues.append(
                f"Unknown training stages: {unknown}. Available stages: "
                f"{sorted(available_stages)}"
            )

        unknown_rerun = sorted(set(cfg.pipeline.rerun) - set(stages))
        if unknown_rerun:
            issues.append(
                "pipeline.rerun contains stages not present in the run: "
                + ", ".join(unknown_rerun)
            )

        configured_plugins = {}
        for stage in stages:
            if stage in unknown:
                continue
            try:
                plugin = registry.create(stage)
                configured_plugins[stage] = plugin
            except Exception as exc:
                issues.append(f"Stage plugin {stage!r} is invalid: {exc}")

        from ..runtime.BackendConfig import validate_runtime_backends
        from ..compression.QuantizationStage import validate_quantization_stage

        try:
            validate_quantization_stage(cfg)
        except (ValueError, TypeError, OSError) as exc:
            issues.append(f"Quantization configuration is invalid: {exc}")
        if "decision" in stages:
            from ..decision.config import decision_plan

            try:
                decision = decision_plan(cfg, inspect_data=inspect_data)
                training_estimates["decision"] = decision
                if inspect_data and cfg.decision.enabled:
                    data_inspections["decision"] = {"counts": decision["counts"], "dataset_hashes": decision["dataset_hashes"]}
            except Exception as exc:
                issues.append(f"Decision training configuration is invalid: {exc}")
        try:
            validate_runtime_backends(cfg)
        except Exception as exc:
            issues.append(f"Runtime backend configuration is invalid: {exc}")
        if cfg.backends.training == "megatron":
            if cfg.sft.retention_kl_weight > 0:
                issues.append("SFT reference KL is not implemented by Megatron recipes.")
            if cfg.sft.replay_data_path or cfg.sft.replay_ratio or cfg.sft.replay_count is not None:
                issues.append("SFT replay requires a native or Hugging Face trainer; Megatron recipes must implement it explicitly.")
            warnings.append(
                "Megatron Bridge recipes/overrides own model, dataset and hyperparameters; "
                "native training/SFT settings are not translated. GPU execution requires Linux."
            )
            return TrainingConfigValidation(
                valid=not issues, stages=stages, issues=issues, warnings=warnings,
                architecture_analysis={"source": "megatron_bridge_recipe"},
                normalized_config=safe_raw,
            )

        if cfg.distributed.bf16 and cfg.distributed.fp16:
            issues.append("distributed.bf16 and distributed.fp16 cannot both be enabled.")
        if (
            cfg.distributed.zero_offload_to_cpu
            and cfg.distributed.zero_offload_to_nvme
        ):
            issues.append(
                "distributed.zero_offload_to_cpu and zero_offload_to_nvme "
                "are mutually exclusive."
            )
        if cfg.distributed.zero_offload_to_nvme and not str(
            cfg.distributed.zero_nvme_path
        ).strip():
            issues.append(
                "distributed.zero_nvme_path must not be empty when NVMe offload is enabled."
            )
        for field_name in (
            "zero_stage3_prefetch_bucket_size",
            "zero_reduce_bucket_size",
            "zero_allgather_bucket_size",
        ):
            value = getattr(cfg.distributed, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                issues.append(
                    f"distributed.{field_name} must be a non-negative integer."
                )
        if cfg.distributed.num_gpus < 1:
            issues.append("distributed.num_gpus must be >= 1.")
        if cfg.distributed.num_nodes < 1:
            issues.append("distributed.num_nodes must be >= 1.")
        if cfg.distributed.strategy not in {
            "single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"
        }:
            issues.append(f"Unsupported distributed.strategy: {cfg.distributed.strategy!r}.")
        if cfg.distributed.strategy != "single":
            unsupported = {
                stage
                for stage, plugin in configured_plugins.items()
                if not plugin.capabilities.supports_strategy(cfg.distributed.strategy)
            }
            if unsupported:
                issues.append(
                    f"Distributed strategy {cfg.distributed.strategy!r} is unsupported "
                    f"by stages: {sorted(unsupported)}."
                )
            if "pretrain" in stages and cfg.model_backend != "saddle":
                issues.append("Distributed pretraining is currently verified for model.backend='saddle' only.")
        if cfg.training.max_steps == 0 or cfg.training.max_steps < -1:
            issues.append("training.max_steps must be -1 (use stage epochs) or a positive integer.")
        if (
            isinstance(cfg.training.loss_chunk_size, bool)
            or not isinstance(cfg.training.loss_chunk_size, int)
            or cfg.training.loss_chunk_size < 0
        ):
            issues.append("training.loss_chunk_size must be a non-negative integer.")
        if (
            isinstance(cfg.training.dataloader_num_workers, bool)
            or not isinstance(cfg.training.dataloader_num_workers, int)
            or cfg.training.dataloader_num_workers < 0
        ):
            issues.append("training.dataloader_num_workers must be a non-negative integer.")
        prefetch_factor = cfg.training.dataloader_prefetch_factor
        if (
            prefetch_factor is not None
            and (
                isinstance(prefetch_factor, bool)
                or not isinstance(prefetch_factor, int)
                or prefetch_factor <= 0
            )
        ):
            issues.append(
                "training.dataloader_prefetch_factor must be null or a positive integer."
            )
        has_upstream_model = "pretrain" in stages

        if "pretrain" in stages and not cfg.data.sources:
            issues.append("pretrain stage requires data.sources.")
        if "pretrain" in stages:
            if cfg.pretrain_mode not in {"scratch", "continue"}:
                issues.append("pretrain_mode must be 'scratch' or 'continue'.")
            elif cfg.pretrain_mode == "continue":
                if not cfg.model_name_or_path:
                    issues.append(
                        "Continue pretraining requires model.name_or_path."
                    )
                architecture_analysis = {
                    "mode": "continue",
                    "source": "checkpoint",
                    "model_name_or_path": cfg.model_name_or_path,
                    "structure_inherited": True,
                }
            else:
                try:
                    blueprint = TrainingOrchestrator.resolve_architecture_blueprint(
                        cfg.model_config, cfg.model_blueprint
                    )
                    architecture_analysis = {
                        "mode": "scratch",
                        "source": (
                            "model.config + model.blueprint"
                            if cfg.model_blueprint
                            else "model.config preset"
                        ),
                        "name": blueprint.name,
                        "family": blueprint.family,
                        **blueprint.analyze(),
                    }
                except Exception as exc:
                    issues.append(f"Pretraining architecture is invalid: {exc}")
            from .PackingPolicy import resolve_packing_attention

            try:
                packing_backend = (
                    "megatron" if cfg.backends.training == "megatron" else cfg.model_backend
                )
                packing_policy = resolve_packing_attention(
                    cfg.data.packing_attention,
                    packing_backend,
                    cfg.data.pack_sequences,
                )
                if packing_policy == "concat":
                    warnings.append("Packed pretraining uses concat attention: earlier documents in a block are visible.")
            except ValueError as exc:
                issues.append(str(exc))
            if not 0 <= cfg.data.validation_split < 1:
                issues.append("data.validation_split must be in the range [0, 1).")
            elif cfg.data.validation_split == 0:
                warnings.append(
                    "Pretraining validation is disabled; validation PPL and best-checkpoint "
                    "selection will not be available."
                )
            elif cfg.training.eval_every_steps <= 0:
                issues.append(
                    "training.eval_every_steps must be positive when "
                    "data.validation_split is enabled."
                )
            if cfg.training.per_device_batch_size <= 0:
                issues.append("training.per_device_batch_size must be positive.")
            if cfg.training.global_batch_size <= 0:
                issues.append("training.global_batch_size must be positive.")
            if cfg.training.learning_rate <= 0:
                issues.append("training.learning_rate must be positive.")
            if not 0 <= cfg.training.min_lr <= cfg.training.learning_rate:
                issues.append(
                    "training.min_lr must be between 0 and training.learning_rate."
                )
            world_size = max(1, cfg.distributed.num_gpus * cfg.distributed.num_nodes)
            micro_global_batch = cfg.training.per_device_batch_size * world_size
            grad_accum = cfg.distributed.gradient_accumulation_steps
            if grad_accum is None:
                if (
                    micro_global_batch > 0
                    and cfg.training.global_batch_size % micro_global_batch != 0
                ):
                    issues.append(
                        "training.global_batch_size must be divisible by "
                        "per_device_batch_size * num_gpus * num_nodes when gradient "
                        "accumulation is derived automatically."
                    )
            elif grad_accum <= 0:
                issues.append(
                    "distributed.gradient_accumulation_steps must be positive."
                )
            elif micro_global_batch * grad_accum != cfg.training.global_batch_size:
                issues.append(
                    "Pretraining global_batch_size does not match per_device_batch_size "
                    "* num_gpus * num_nodes * gradient_accumulation_steps."
                )
            if (
                cfg.data.validation_split > 0
                and cfg.training.load_best_model_at_end
                and cfg.training.eval_every_steps > 0
                and cfg.training.save_every_steps
                % cfg.training.eval_every_steps
                != 0
            ):
                issues.append(
                    "training.save_every_steps must be a multiple of "
                    "training.eval_every_steps when load_best_model_at_end=true."
                )

        if cfg.eval.regression_suite:
            try:
                from ..evaluation.RegressionSuite import load_regression_suite

                suite = load_regression_suite(cfg.eval.regression_suite)
                if cfg.eval.checkpoint_candidates:
                    selection_task = next(
                        (task for task in suite["tasks"] if task["name"] == cfg.eval.selection_task),
                        None,
                    )
                    if selection_task is None or cfg.eval.selection_metric not in selection_task["metrics"]:
                        issues.append("Checkpoint selection task/metric must be declared in the regression suite.")
                    else:
                        metric_rule = selection_task["metrics"][cfg.eval.selection_metric]
                        if metric_rule["higher_is_better"] != cfg.eval.selection_higher_is_better:
                            issues.append("Checkpoint selection direction must match its regression-suite metric rule.")
                if inspect_data:
                    for task in suite["tasks"]:
                        if not os.path.isfile(task["data"]):
                            issues.append(f"Regression task {task['name']!r} data not found: {task['data']}")
            except (ValueError, OSError, TypeError) as exc:
                issues.append(f"Invalid regression suite: {exc}")
            if not cfg.eval.enabled or "eval" not in stages:
                issues.append("eval.regression_suite requires an enabled eval stage.")
            if not (cfg.eval.regression_baseline or cfg.model_name_or_path):
                issues.append("eval.regression_baseline is required without an original model.name_or_path.")

        if cfg.eval.gate.enabled:
            from ..evaluation.ReleaseGate import EvaluationReleaseGate

            issues.extend(EvaluationReleaseGate.validate_rules(cfg.eval.gate.rules))
            if not cfg.eval.enabled:
                issues.append("evaluation gate requires eval.enabled=true.")
            if "eval" not in stages:
                issues.append("evaluation gate requires an eval stage.")

        if "export" in stages or cfg.export.enabled:
            if not cfg.export.enabled:
                warnings.append("export stage is listed but export.enabled is false.")
            if str(cfg.export.format).lower() not in {"hf", "saddle", "native", "saddle-native", "saddle/native"}:
                issues.append("export.format must be 'hf' or 'saddle' (native aliases are accepted).")
            if cfg.export.device not in {"auto", "cpu", "cuda"}:
                issues.append("export.device must be one of: auto, cpu, cuda.")
            if cfg.export.dtype not in {"auto", "float32", "float16", "bfloat16"}:
                issues.append("export.dtype must be one of: auto, float32, float16, bfloat16.")
            has_model_source = bool(cfg.export.model_path or cfg.model_name_or_path or has_upstream_model
                                    or cfg.quantization.model_path or cfg.eval.checkpoint_candidates)
            has_model_source = has_model_source or any(
                stage in stages for stage in ("sft", "preference", "rlhf")
            )
            if not has_model_source:
                issues.append("Export requires export.model_path, model.name_or_path, or an upstream model stage.")
            export_index = stages.index("export") if "export" in stages else None
            eval_index = stages.index("eval") if "eval" in stages else None
            if (cfg.eval.gate.enabled or cfg.eval.regression_suite) and export_index is not None:
                if eval_index is None or eval_index > export_index:
                    issues.append("An enabled evaluation gate requires eval before export.")
            if cfg.export.require_gate:
                if not (cfg.eval.gate.enabled or cfg.eval.regression_suite):
                    issues.append("export.require_gate=true requires eval.gate.enabled=true or eval.regression_suite.")
                if export_index is not None and (eval_index is None or eval_index > export_index):
                    issues.append("export.require_gate=true requires eval before export.")

        if "operator" in stages:
            from saddle_ml.agent import SUPPORTED_AGENT_OPERATORS, normalize_agent_config
            from ..agents.AgentModelProvider import endpoint as _endpoint
            from ..agents.RealWorldOperators import SUPPORTED_OPERATORS

            supported_visual_operators = SUPPORTED_OPERATORS | SUPPORTED_AGENT_OPERATORS

            if not cfg.operator.enabled:
                issues.append("operator stage requires operator.enabled=true.")
            if not cfg.operator.label:
                issues.append("operator stage requires operator.label.")
            elif cfg.operator.label not in supported_visual_operators:
                issues.append(
                    f"Unsupported operator.label: {cfg.operator.label}. "
                    f"Available operators: {sorted(supported_visual_operators)}"
                )
            elif cfg.operator.label in SUPPORTED_AGENT_OPERATORS:
                try:
                    normalized_agent, migration_warnings, _ = normalize_agent_config(
                        cfg.operator.label,
                        cfg.operator.config,
                    )
                    safe_operator = safe_raw.get("operator")
                    if isinstance(safe_operator, dict):
                        safe_operator["config"] = normalized_agent
                    warnings.extend(f"operator config: {warning}" for warning in migration_warnings)
                    provider = normalized_agent["provider"]
                    if provider["type"] == "openai_compatible":
                        endpoint = _endpoint(
                            provider["baseUrl"],
                            allow_private_network=provider.get("allowPrivateNetwork", False),
                        )
                        hostname = (urlparse(endpoint).hostname or "").lower()
                        if hostname not in {"127.0.0.1", "localhost", "::1"} and not provider["credentialEnv"]:
                            issues.append("Remote Agent provider requires provider.credentialEnv.")
                        if not any(str(agent.get("model") or "") for agent in normalized_agent["agents"]):
                            issues.append("Agent operator requires provider.model or an agent-level model.")
                except Exception as exc:
                    issues.append(f"Agent operator config is invalid: {exc}")

        sft_can_use_mopd = "mopd" in stages and "sft" in stages and stages.index("mopd") < stages.index("sft")
        if "sft" in stages or cfg.sft.enabled:
            if cfg.sft.retention_kl_weight > 0 and cfg.distributed.strategy not in {"single", "ddp"}:
                issues.append("SFT reference KL currently supports single-device/DDP, not FSDP/DeepSpeed.")
            if not cfg.model_name_or_path and not has_upstream_model:
                issues.append("SFT requires model.name_or_path unless an upstream pretrain stage creates the model.")
            if not cfg.sft.enabled:
                warnings.append("sft stage is listed but sft.enabled is false.")
            if not cfg.sft.data_path and not sft_can_use_mopd:
                issues.append("SFT requires sft.data_path, unless an upstream MOPD stage materializes data.")
            if cfg.training.max_steps == -1 and cfg.sft.epochs <= 0:
                issues.append("SFT requires sft.epochs > 0 when training.max_steps is -1.")
            if cfg.sft.data_path:
                cls._inspect(data_inspections, issues, warnings, "sft", cfg.sft.data_path, "sft", inspect_data)
            elif sft_can_use_mopd:
                warnings.append("SFT has no data_path and will use the upstream MOPD output at runtime.")
            if cfg.sft.replay_data_path:
                cls._inspect(data_inspections, issues, warnings, "sft_replay", cfg.sft.replay_data_path, "sft", inspect_data)
            if cfg.sft.validation_data_path and cfg.sft.validation_split:
                issues.append(
                    "SFT accepts either sft.validation_data_path or "
                    "sft.validation_split, not both."
                )
            if cfg.sft.validation_data_path:
                if (
                    isinstance(cfg.sft.data_path, str)
                    and os.path.isfile(cfg.sft.data_path)
                    and os.path.isfile(cfg.sft.validation_data_path)
                    and os.path.abspath(cfg.sft.data_path)
                    == os.path.abspath(cfg.sft.validation_data_path)
                ):
                    issues.append("SFT training and validation files must be different.")
                cls._inspect(
                    data_inspections,
                    issues,
                    warnings,
                    "sft_validation",
                    cfg.sft.validation_data_path,
                    "sft",
                    inspect_data,
                )
                cls._audit_split(
                    data_inspections,
                    issues,
                    warnings,
                    "sft_split_leakage",
                    cfg.sft.data_path,
                    cfg.sft.validation_data_path,
                    inspect_data,
                )
            training_estimates["sft"] = cls._estimate(
                cfg.sft.per_device_batch_size,
                cfg.sft.gradient_accumulation_steps,
                cfg.distributed.num_gpus,
                cfg.sft.max_seq_length,
                cfg.training.max_steps,
                cfg.sft.epochs,
            )

        if "preference" in stages or cfg.preference.enabled:
            if not cfg.model_name_or_path and not has_upstream_model and "sft" not in stages:
                issues.append("Preference training requires model.name_or_path or an upstream sft/pretrain stage.")
            if not cfg.preference.enabled:
                warnings.append("preference stage is listed but preference.enabled is false.")
            if cfg.preference.method not in {"dpo", "orpo", "kto"}:
                issues.append(f"Unsupported preference.method: {cfg.preference.method}.")
            if cfg.preference.per_device_batch_size <= 0:
                issues.append("preference.per_device_batch_size must be > 0.")
            if cfg.training.max_steps == -1 and cfg.preference.epochs <= 0:
                issues.append("Preference training requires preference.epochs > 0 when training.max_steps is -1.")
            preference_actual_batch_size = cfg.preference.per_device_batch_size * max(1, cfg.distributed.num_gpus)
            if cfg.preference.method == "kto" and preference_actual_batch_size <= 1:
                issues.append(
                    "KTO preference training requires actual train batch size > 1; "
                    "increase preference.per_device_batch_size or distributed.num_gpus."
                )
            if not cfg.preference.data_path:
                issues.append("Preference training requires preference.data_path.")
            else:
                cls._inspect(
                    data_inspections,
                    issues,
                    warnings,
                    "preference",
                    cfg.preference.data_path,
                    cfg.preference.method,
                    inspect_data,
                )
            if cfg.preference.validation_data_path and cfg.preference.validation_split:
                issues.append(
                    "Preference training accepts either "
                    "preference.validation_data_path or "
                    "preference.validation_split, not both."
                )
            if cfg.preference.validation_data_path:
                if (
                    isinstance(cfg.preference.data_path, str)
                    and os.path.isfile(cfg.preference.data_path)
                    and os.path.isfile(cfg.preference.validation_data_path)
                    and os.path.abspath(cfg.preference.data_path)
                    == os.path.abspath(cfg.preference.validation_data_path)
                ):
                    issues.append(
                        "Preference training and validation files must be different."
                    )
                cls._inspect(
                    data_inspections,
                    issues,
                    warnings,
                    "preference_validation",
                    cfg.preference.validation_data_path,
                    cfg.preference.method,
                    inspect_data,
                )
                cls._audit_split(
                    data_inspections,
                    issues,
                    warnings,
                    "preference_split_leakage",
                    cfg.preference.data_path,
                    cfg.preference.validation_data_path,
                    inspect_data,
                )
            training_estimates["preference"] = cls._estimate(
                cfg.preference.per_device_batch_size,
                cfg.preference.gradient_accumulation_steps,
                cfg.distributed.num_gpus,
                cfg.preference.max_length,
                cfg.training.max_steps,
                cfg.preference.epochs,
            )

        if "rlhf" in stages or cfg.rlhf.enabled:
            if not cfg.model_name_or_path and not has_upstream_model and "sft" not in stages and "preference" not in stages:
                issues.append("RLHF requires model.name_or_path or an upstream preference/sft/pretrain stage.")
            if not cfg.rlhf.enabled:
                warnings.append("rlhf stage is listed but rlhf.enabled is false.")
            if cfg.training.max_steps == -1 and cfg.rlhf.epochs <= 0:
                issues.append("RLHF requires rlhf.epochs > 0 when training.max_steps is -1.")
            if cfg.rlhf.method not in {"dpo", "ppo"}:
                issues.append(f"Unsupported rlhf.method: {cfg.rlhf.method}.")
            if cfg.rlhf.method == "ppo":
                issues.append(
                    "PPO execution is not wired into the stable pipeline; an explicit "
                    "reward model, value model, rollout policy, and safety-tested reward "
                    "contract are required."
                )
            if not cfg.rlhf.data_path:
                issues.append("RLHF requires rlhf.data_path.")
            else:
                task = "dpo" if cfg.rlhf.method == "dpo" else "rlhf"
                cls._inspect(data_inspections, issues, warnings, "rlhf", cfg.rlhf.data_path, task, inspect_data)
            training_estimates["rlhf"] = cls._estimate(
                cfg.rlhf.per_device_batch_size,
                cfg.rlhf.gradient_accumulation_steps,
                cfg.distributed.num_gpus,
                cfg.rlhf.max_length,
                cfg.training.max_steps,
                cfg.rlhf.epochs,
            )

        if "mopd" in stages or cfg.mopd.enabled:
            if not cfg.mopd.enabled:
                warnings.append("mopd stage is listed but mopd.enabled is false.")
            if not cfg.mopd.prompts_path:
                issues.append("MOPD requires mopd.prompts_path.")
            if not cfg.mopd.teachers:
                warnings.append("MOPD has no teacher specs; dry-run can pass, but real distillation needs teachers.")
            if cfg.mopd.teacher_concurrency < 1:
                issues.append("mopd.teacher_concurrency must be at least 1.")

        if "world_model" in stages or cfg.world_model.enabled:
            if not cfg.world_model.enabled:
                warnings.append(
                    "world_model stage is listed but world_model.enabled is false."
                )
            backend_aliases = {
                "": "rssm",
                "native": "rssm",
                "gaussian": "rssm",
                "gaussian_rssm": "rssm",
                "categorical": "categorical_rssm",
                "discrete_rssm": "categorical_rssm",
            }
            backend = str(cfg.world_model.backend or "").strip().lower()
            backend = backend_aliases.get(backend, backend)
            if backend not in {"rssm", "categorical_rssm"}:
                issues.append(
                    "world_model.backend must be 'rssm' or 'categorical_rssm'."
                )
            action_type = str(
                cfg.world_model.model.get("action_type", "continuous")
            ).lower()
            if action_type not in {"continuous", "discrete"}:
                issues.append(
                    "world_model.model.action_type must be 'continuous' or 'discrete'."
                )
            if cfg.world_model.config_path:
                if not os.path.isfile(cfg.world_model.config_path):
                    issues.append(
                        "world_model.config_path does not exist: "
                        f"{cfg.world_model.config_path}"
                    )
            else:
                train_path = cfg.world_model.data.get("train_path")
                if not train_path:
                    issues.append(
                        "World-model training requires world_model.data.train_path."
                    )
                elif not os.path.isfile(train_path):
                    issues.append(
                        f"world_model data file does not exist: {train_path}"
                    )
                elif inspect_data:
                    try:
                        from ..world_models.WorldModelData import (
                            infer_world_model_dimensions,
                            load_world_model_trajectories,
                        )

                        trajectories = load_world_model_trajectories(train_path)
                        dimensions = infer_world_model_dimensions(
                            trajectories, action_type=action_type
                        )
                        data_inspections["world_model"] = {
                            "path": os.path.abspath(train_path),
                            "episodes": len(trajectories),
                            "transitions": sum(
                                len(item["actions"]) for item in trajectories
                            ),
                            **dimensions,
                        }
                    except ImportError as exc:
                        warnings.append(
                            "world_model data inspection needs training dependencies: "
                            f"{exc}"
                        )
                    except Exception as exc:
                        issues.append(f"world_model data: {exc}")

        if "eye_control" in stages or cfg.eye_control.enabled:
            if "eye_control" in stages and not cfg.eye_control.enabled:
                warnings.append(
                    "eye_control stage is listed but eye_control.enabled is false."
                )
            if cfg.eye_control.enabled and "eye_control" not in stages:
                issues.append(
                    "eye_control.enabled=true requires the eye_control stage."
                )
            try:
                from ..spatial.prosthetic_eye_control import (
                    EyePIDConfig,
                    EyePlantConfig,
                    EyeSafetyConfig,
                )
                from ..world_models.WorldModelInference import WorldModelPlannerConfig

                EyeSafetyConfig.from_dict(cfg.eye_control.safety)
                EyePIDConfig.from_dict(cfg.eye_control.pid)
                EyePlantConfig.from_dict(cfg.eye_control.plant)
                WorldModelPlannerConfig.from_dict(cfg.eye_control.planner)
            except Exception as exc:
                issues.append(f"eye_control config: {exc}")
            if cfg.eye_control.episodes <= 0:
                issues.append("eye_control.episodes must be positive.")
            if cfg.eye_control.steps <= 0:
                issues.append("eye_control.steps must be positive.")
            if not cfg.eye_control.output_path:
                issues.append("eye_control.output_path is required.")
            if cfg.eye_control.require_world_model:
                checkpoint = cfg.eye_control.checkpoint_path
                if not checkpoint and "world_model" not in stages:
                    issues.append(
                        "eye_control requires checkpoint_path or an upstream "
                        "world_model stage."
                    )
                elif (
                    not checkpoint
                    and "eye_control" in stages
                    and stages.index("world_model") > stages.index("eye_control")
                ):
                    issues.append("eye_control must execute after world_model.")
                elif checkpoint and not os.path.exists(checkpoint):
                    issues.append(
                        f"eye_control checkpoint does not exist: {checkpoint}"
                    )
            inspection = data_inspections.get("world_model", {})
            if inspection:
                if tuple(inspection.get("observation_shape", ())) != (9,):
                    issues.append(
                        "eye_control requires world-model observation_shape [9]."
                    )
                if inspection.get("action_dim") != 2:
                    issues.append("eye_control requires world-model action_dim 2.")
                if inspection.get("action_type") != "continuous":
                    issues.append(
                        "eye_control requires a continuous-action world model."
                    )

        if "vla_sft" in stages or cfg.vla.enabled:
            if not cfg.model_name_or_path and not has_upstream_model:
                issues.append("VLA SFT requires model.name_or_path unless an upstream pretrain stage creates the model.")
            if not cfg.vla.enabled:
                warnings.append("vla_sft stage is listed but vla.enabled is false.")
            effective_vla_max_steps = cfg.vla.max_steps if cfg.vla.max_steps > 0 else cfg.training.max_steps
            if effective_vla_max_steps == -1 and cfg.vla.epochs <= 0:
                issues.append("VLA SFT requires vla.epochs > 0 when max_steps is -1.")
            if not cfg.vla.data_path:
                issues.append("VLA SFT requires vla.data_path.")
            elif inspect_data:
                from ..multimodal.VLA import VLAActionSpace
                from ..multimodal.VLADataInspector import VLADataInspector

                action_space = VLAActionSpace(
                    action_dim=cfg.vla.action_dim,
                    action_type=cfg.vla.action_type,
                    bins=cfg.vla.action_bins,
                    min_value=cfg.vla.action_min_value,
                    max_value=cfg.vla.action_max_value,
                    include_gripper=cfg.vla.include_gripper,
                    control_hz=cfg.vla.control_hz,
                    coordinate_frame=cfg.vla.coordinate_frame,
                )
                report = VLADataInspector.inspect_file(
                    cfg.vla.data_path,
                    image_root=cfg.vla.image_root,
                    action_space=action_space,
                )
                data_inspections["vla_sft"] = report.to_dict()
                if report.errors:
                    issues.extend(f"vla_sft data: {error}" for error in report.errors)
                warnings.extend(f"vla_sft data: {warning}" for warning in report.warnings)
            if cfg.vla.action_dim <= 0:
                issues.append("vla.action_dim must be > 0.")
            if cfg.vla.action_min_value >= cfg.vla.action_max_value:
                issues.append("vla action min_value must be smaller than max_value.")
            training_estimates["vla_sft"] = cls._estimate(
                cfg.vla.per_device_batch_size,
                cfg.vla.gradient_accumulation_steps,
                cfg.distributed.num_gpus,
                cfg.vla.max_seq_length,
                cfg.vla.max_steps if cfg.vla.max_steps > 0 else cfg.training.max_steps,
                cfg.vla.epochs,
            )

        return TrainingConfigValidation(
            valid=not issues,
            stages=stages,
            issues=issues,
            warnings=warnings,
            data_inspections=data_inspections,
            training_estimates=training_estimates,
            architecture_analysis=architecture_analysis,
            normalized_config=safe_raw,
        )

    @staticmethod
    def _estimate(
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        num_gpus: int,
        max_seq_length: int,
        max_steps: int,
        epochs: Optional[int],
    ) -> Dict[str, Any]:
        from .TrainingPlanEstimator import TrainingPlanEstimator

        return TrainingPlanEstimator.estimate(
            per_device_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_gpus=num_gpus,
            max_seq_length=max_seq_length,
            max_steps=max_steps,
            epochs=epochs,
        ).to_dict()

    @staticmethod
    def _inspect(
        data_inspections: Dict[str, Dict[str, Any]],
        issues: List[str],
        warnings: List[str],
        stage: str,
        path: str,
        task: str,
        inspect_data: bool,
    ) -> None:
        if not inspect_data:
            return
        from ..data.TrainingDataInspector import TrainingDataInspector

        report = TrainingDataInspector.inspect_file(path, task=task)
        data_inspections[stage] = report.to_dict()
        if report.errors:
            issues.extend(f"{stage} data: {error}" for error in report.errors)
        warnings.extend(f"{stage} data: {warning}" for warning in report.warnings)

    @staticmethod
    def _audit_split(
        data_inspections: Dict[str, Dict[str, Any]],
        issues: List[str],
        warnings: List[str],
        stage: str,
        train_path: Optional[str],
        validation_path: Optional[str],
        inspect_data: bool,
    ) -> None:
        if (
            not inspect_data
            or not isinstance(train_path, str)
            or not isinstance(validation_path, str)
            or not os.path.isfile(train_path)
            or not os.path.isfile(validation_path)
        ):
            return
        from ..data.TrainingDataInspector import audit_training_splits

        report = audit_training_splits(train_path, validation_path)
        data_inspections[stage] = report
        issues.extend(f"{stage}: {issue}" for issue in report["issues"])
        warnings.extend(f"{stage}: {warning}" for warning in report["warnings"])


def validate_training_config(
    raw: Dict[str, Any],
    inspect_data: bool = True,
    *,
    stage_registry=None,
) -> Dict[str, Any]:
    return TrainingConfigValidator.validate(
        raw,
        inspect_data=inspect_data,
        stage_registry=stage_registry,
    ).to_dict()
