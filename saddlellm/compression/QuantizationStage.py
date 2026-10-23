"""Optional post-training quantization in the canonical stage scheduler."""

from __future__ import annotations

import gc
import json
import sys

# ruff: noqa: N999 -- Public modules follow the repository's naming convention.
from dataclasses import dataclass
from pathlib import Path

from .QuantizationPipeline import QuantizationConfig, plan_quantization

MODEL_STAGES = ("rlhf", "preference", "sft", "pretrain", "mllm_sft", "vision_alignment")


@dataclass
class QuantizationStageConfig(QuantizationConfig):
    model_path: str | None = None
    output_dir: str | None = None
    merge_lora: bool = True

    def __post_init__(self):
        super().__post_init__()
        if type(self.merge_lora) is not bool:
            raise TypeError("quantization.merge_lora must be boolean")
        for name in ("model_path", "output_dir"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (str, Path)) or not str(value).strip()):
                raise ValueError(f"quantization.{name} must be a nonempty path")

    def function_config(self):
        return QuantizationConfig(ptq=self.ptq, qat=self.qat)


def quantization_enabled(config):
    settings = config.quantization
    if type(settings.ptq.enabled) is not bool or type(settings.qat.enabled) is not bool:
        raise ValueError("ptq.enabled and qat.enabled must be booleans")
    if settings.ptq.enabled and settings.qat.enabled:
        raise ValueError("Enable either ptq or qat; they use different numerical/export formats")
    return settings.ptq.enabled or settings.qat.enabled


def _require_dependency(config, stage, upstream):
    dependencies = config.pipeline.dependencies
    if dependencies is None:
        return
    pending = list(dependencies.get(stage, ()))
    seen = set()
    while pending:
        parent = pending.pop()
        if parent == upstream:
            return
        if parent not in seen:
            seen.add(parent)
            pending.extend(dependencies.get(parent, ()))
    raise ValueError(f"pipeline.dependencies must make {stage} depend on {upstream} for safe resume/rerun")


def validate_quantization_stage(config):
    """Check settings without rejecting completed outputs before resume.

    Execution independently requires a new/empty destination. The state store
    restores completed stages only after static configuration validation.
    """
    if not quantization_enabled(config):
        return
    stages = config.stages
    if "quantize" not in stages:
        raise ValueError("Enabled quantization requires a quantize stage in stages")
    index = stages.index("quantize")
    for stage in MODEL_STAGES:
        if stage in stages:
            if stages.index(stage) > index:
                raise ValueError(f"quantize must run after {stage}")
            _require_dependency(config, "quantize", stage)
    for stage in ("eval", "export"):
        if stage in stages:
            if stages.index(stage) < index:
                raise ValueError(f"quantize must run before {stage}")
            _require_dependency(config, stage, "quantize")
    if "eval" in stages and "export" in stages and (config.eval.gate.enabled or config.eval.regression_suite or config.export.require_gate):
        _require_dependency(config, "export", "eval")
    if config.eval.adapter:
        raise ValueError("eval.adapter cannot overlay the quantized model; merge adapters before quantization")
    settings = config.quantization
    if not settings.model_path and not config.model_name_or_path and not any(stage in stages[:index] for stage in MODEL_STAGES):
        raise ValueError("Quantization requires quantization.model_path, model.name_or_path, or an upstream model stage")
    output = settings.output_dir or str(Path(config.logging.output_dir) / "quantized")
    source = settings.model_path or config.model_name_or_path or "<upstream>"
    if settings.ptq.enabled:
        from .CalibratedPTQ import validate_ptq_config

        validate_ptq_config(settings.ptq)
        if config.model_backend == "saddle" and not settings.model_path:
            raise ValueError("AWQ/GPTQ currently require an HF checkpoint; use QAT for a native Saddle model")
    else:
        from .QuantizationAwareTraining import validate_qat_config

        validate_qat_config(settings.qat, source, output, allow_existing_output=True)


def latest_language_model_path(orchestrator, *, include_quantization=True):
    results = orchestrator._stage_results
    if include_quantization:
        result = results.get("quantize", {})
        if result.get("status") in {"completed", "planned"} and result.get("model_path"):
            return result["model_path"]
    # Respect either preference/RLHF execution order. The fallback preserves
    # callers which inject conventional stage results directly before eval.
    ordered = tuple(reversed(orchestrator.config.stages)) + MODEL_STAGES
    return next((results.get(stage, {}).get("model_path") for stage in ordered
                 if stage in MODEL_STAGES and results.get(stage, {}).get("model_path")), None) or orchestrator.config.model_name_or_path


def _prepare_adapter(source, destination, route, quantized_output):
    from ..evaluation.ModelIdentity import (
        compute_model_identity,
        model_identities_match,
    )
    from ..runtime.ModelExporter import ModelExporter, ModelExportRequest

    marker = destination / "quantization_preparation.json"
    adapter = json.loads((Path(source).expanduser() / "adapter_config.json").read_text(encoding="utf-8"))
    base_reference = adapter.get("base_model_name_or_path") if isinstance(adapter, dict) else None
    if isinstance(base_reference, str) and Path(base_reference).expanduser().is_dir():
        base = Path(base_reference).expanduser().resolve()
        for target in (destination.resolve(), quantized_output.resolve()):
            if target == base or base in target.parents or target in base.parents:
                raise ValueError("Quantization and merged outputs must be separate from the adapter's base checkpoint")
    source_identity = compute_model_identity(source)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        try:
            prepared = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Existing merged checkpoint has no verified preparation record; choose a new quantization output_dir") from exc
        if (not isinstance(prepared, dict) or prepared.get("schema_version") != 1
                or prepared.get("dtype") != "float32"
                or not model_identities_match(prepared.get("source_model_identity"), source_identity)
                or not model_identities_match(prepared.get("model_identity"), compute_model_identity(destination))):
            raise ValueError("Merged checkpoint or adapter/base contents changed; choose a new quantization output_dir")
        return
    ModelExporter.export(ModelExportRequest(
        model_path=str(source), output_dir=str(destination), merge_lora=True,
        device=route.device, dtype="float32", local_files_only=route.local_files_only,
        trust_remote_code=route.trust_remote_code,
    ))
    marker.write_text(json.dumps({
        "schema_version": 1, "dtype": "float32",
        "source_model_identity": source_identity,
        "model_identity": compute_model_identity(destination),
    }, indent=2) + "\n", encoding="utf-8")


def run_quantization_stage(orchestrator):
    from .QuantizationPipeline import apply_quantization

    config = orchestrator.config
    settings = config.quantization
    if not quantization_enabled(config):
        orchestrator._stage_results["quantize"] = {
            "status": "skipped", "enabled": False, "reason": "PTQ and QAT are disabled",
        }
        return
    source = settings.model_path or latest_language_model_path(orchestrator, include_quantization=False)
    if not source:
        raise ValueError("Quantization has no upstream model checkpoint")
    output = settings.output_dir or str(Path(orchestrator._output_dir) / "quantized")
    function_config = settings.function_config()
    dry_run = config.training.dry_run or config.training.preflight_only
    plan_quantization(source, config=function_config, output_dir=output)
    if not dry_run:
        orchestrator._model = None
        orchestrator._tokenizer = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    original_source = source
    adapter = (Path(source).expanduser() / "adapter_config.json").is_file()
    prepared_source = None
    if adapter:
        if not settings.merge_lora:
            raise ValueError("Quantization requires a full float checkpoint; enable quantization.merge_lora for PEFT adapters")
        destination = Path(output).expanduser().resolve()
        prepared_source = str(destination.with_name(destination.name + "-float"))
        if not dry_run:
            route = settings.ptq if settings.ptq.enabled else settings.qat
            _prepare_adapter(source, Path(prepared_source), route, destination)
            source = prepared_source
    result = apply_quantization(source, config=function_config, output_dir=output, dry_run=dry_run)
    orchestrator._stage_results["quantize"] = {
        **result.report, "source_model": str(original_source),
        "prepared_source": prepared_source,
        "model_path": str(Path(output).expanduser().resolve()),
    }
    # Downstream stages load the saved artifact without retaining a duplicate.
    del result


__all__ = ["QuantizationStageConfig", "run_quantization_stage", "validate_quantization_stage"]
