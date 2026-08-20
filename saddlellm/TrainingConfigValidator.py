"""Static validation for SaddleLLM training configs."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


VALID_STAGES = {
    "tokenizer",
    "pretrain",
    "sft",
    "preference",
    "rlhf",
    "mopd",
    "mllm_sft",
    "vision_alignment",
    "vla_sft",
    "eval",
    "export",
    "operator",
}


@dataclass
class TrainingConfigValidation:
    valid: bool
    stages: List[str]
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_inspections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    training_estimates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    normalized_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrainingConfigValidator:
    """Validate a compiled TrainingOrchestrator config without starting training."""

    @classmethod
    def validate(cls, raw: Dict[str, Any], inspect_data: bool = True) -> TrainingConfigValidation:
        from saddle_ml.agent.operator import _json_safe
        from .TrainingOrchestrator import TrainingOrchestrator

        issues: List[str] = []
        warnings: List[str] = []
        data_inspections: Dict[str, Dict[str, Any]] = {}
        training_estimates: Dict[str, Dict[str, Any]] = {}
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
        unknown = [stage for stage in stages if stage not in VALID_STAGES]
        if unknown:
            issues.append(f"Unknown training stages: {unknown}. Available stages: {sorted(VALID_STAGES)}")

        if cfg.distributed.bf16 and cfg.distributed.fp16:
            issues.append("distributed.bf16 and distributed.fp16 cannot both be enabled.")
        if cfg.distributed.num_gpus < 1:
            issues.append("distributed.num_gpus must be >= 1.")
        if cfg.distributed.num_nodes < 1:
            issues.append("distributed.num_nodes must be >= 1.")
        if cfg.distributed.strategy not in {
            "single", "ddp", "fsdp", "deepspeed_zero2", "deepspeed_zero3"
        }:
            issues.append(f"Unsupported distributed.strategy: {cfg.distributed.strategy!r}.")
        if cfg.distributed.strategy != "single":
            unsupported = set(stages) - {"pretrain"}
            if unsupported:
                issues.append(
                    "Distributed execution currently supports pretrain only; "
                    f"unsupported stages: {sorted(unsupported)}."
                )
            if cfg.model_backend != "saddle":
                issues.append("Distributed pretraining is currently verified for model.backend='saddle' only.")
        if cfg.training.max_steps == 0 or cfg.training.max_steps < -1:
            issues.append("training.max_steps must be -1 (use stage epochs) or a positive integer.")
        has_upstream_model = "pretrain" in stages

        if "pretrain" in stages and not cfg.data.sources:
            issues.append("pretrain stage requires data.sources.")

        if cfg.eval.gate.enabled:
            from .ReleaseGate import EvaluationReleaseGate

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
            has_model_source = bool(cfg.export.model_path or cfg.model_name_or_path or has_upstream_model)
            has_model_source = has_model_source or any(
                stage in stages for stage in ("sft", "preference", "rlhf")
            )
            if not has_model_source:
                issues.append("Export requires export.model_path, model.name_or_path, or an upstream model stage.")
            export_index = stages.index("export") if "export" in stages else None
            eval_index = stages.index("eval") if "eval" in stages else None
            if cfg.eval.gate.enabled and export_index is not None:
                if eval_index is None or eval_index > export_index:
                    issues.append("An enabled evaluation gate requires eval before export.")
            if cfg.export.require_gate:
                if not cfg.eval.gate.enabled:
                    issues.append("export.require_gate=true requires eval.gate.enabled=true.")
                if export_index is not None and (eval_index is None or eval_index > export_index):
                    issues.append("export.require_gate=true requires eval before export.")

        if "operator" in stages:
            from saddle_ml.agent import SUPPORTED_AGENT_OPERATORS, normalize_agent_config
            from .AgentModelProvider import endpoint as _endpoint
            from .RealWorldOperators import SUPPORTED_OPERATORS

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
                warnings.append("PPO is planned only; current orchestrator has no PPO trainer integration.")
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
                from .VLA import VLAActionSpace
                from .VLADataInspector import VLADataInspector

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
        from .TrainingDataInspector import TrainingDataInspector

        report = TrainingDataInspector.inspect_file(path, task=task)
        data_inspections[stage] = report.to_dict()
        if report.errors:
            issues.extend(f"{stage} data: {error}" for error in report.errors)
        warnings.extend(f"{stage} data: {warning}" for warning in report.warnings)


def validate_training_config(raw: Dict[str, Any], inspect_data: bool = True) -> Dict[str, Any]:
    return TrainingConfigValidator.validate(raw, inspect_data=inspect_data).to_dict()
