"""Opt-in function entry point for calibrated PTQ and backend-specific QAT.

The disabled path preserves its input and imports no tensor/model runtimes.
PTQ and QAT are alternative numerical routes, not implicit sequential passes.
"""

from __future__ import annotations

# ruff: noqa: N999 -- Matches the repository's public module naming convention.
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from os import PathLike
from pathlib import Path
from typing import Any

from .CalibratedPTQ import PTQConfig, validate_ptq_config
from .QuantizationAwareTraining import QATConfig, validate_qat_config


def _section(value, config_type, name):
    if value is None:
        return config_type()
    if isinstance(value, config_type):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a configuration mapping or {config_type.__name__}")
    unknown = set(value) - {item.name for item in fields(config_type)}
    if unknown:
        raise ValueError(f"Unknown {name} configuration keys: {sorted(unknown)}")
    return config_type(**dict(value))


@dataclass
class QuantizationConfig:
    """Two disabled-by-default switches with their own route parameters."""

    ptq: PTQConfig = field(default_factory=PTQConfig)
    qat: QATConfig = field(default_factory=QATConfig)

    def __post_init__(self):
        self.ptq = _section(self.ptq, PTQConfig, "ptq")
        self.qat = _section(self.qat, QATConfig, "qat")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None = None):
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise TypeError("quantization config must be a mapping")
        unknown = set(value) - {"ptq", "qat"}
        if unknown:
            raise ValueError(f"Unknown quantization configuration keys: {sorted(unknown)}")
        return cls(**dict(value))


@dataclass
class QuantizationResult:
    model: Any
    tokenizer: Any
    report: dict[str, Any]
    output_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return report metadata without trying to serialize model objects."""
        return dict(self.report)


def _config(value) -> QuantizationConfig:
    return value if isinstance(value, QuantizationConfig) else QuantizationConfig.from_dict(value)


def _source(value) -> str:
    if isinstance(value, (str, PathLike)):
        if not str(value).strip():
            raise ValueError("model reference must not be empty")
        return str(value)
    if value is None:
        raise ValueError("enabled quantization requires a model or checkpoint reference")
    return "<in-memory>"


def _output(source, output_dir, config):
    if not isinstance(output_dir, (str, PathLike)) or not str(output_dir).strip():
        raise ValueError("enabled quantization requires a new output_dir")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("quantization output_dir must be absent or empty")
    teacher = config.qat.teacher_checkpoint if config.qat.enabled else None
    for reference in (source, teacher):
        if reference and reference != "<in-memory>":
            original = Path(reference).expanduser().resolve()
            if destination == original or original in destination.parents:
                raise ValueError("quantization output_dir must be outside input/teacher checkpoints")
    data_sources = (config.qat.train_data, config.qat.validation_data) if config.qat.enabled else (config.ptq.calibration_path, config.ptq.calibration_data)
    for reference in data_sources:
        if isinstance(reference, (str, PathLike)):
            data = Path(reference).expanduser().resolve()
            if destination == data or destination in data.parents:
                raise ValueError("quantization output_dir must not contain its source data")
    return str(destination)


def plan_quantization(model_or_path=None, *, config=None, output_dir=None) -> dict[str, Any]:
    """Validate the chosen route before loading models or optional dependencies."""
    settings = _config(config)
    if type(settings.ptq.enabled) is not bool or type(settings.qat.enabled) is not bool:
        raise ValueError("ptq.enabled and qat.enabled must be booleans")
    if settings.ptq.enabled and settings.qat.enabled:
        raise ValueError("Enable either ptq or qat: AWQ/GPTQ and this QAT profile use different numerical/export formats")
    if not settings.ptq.enabled and not settings.qat.enabled:
        return {"operation": "quantization", "status": "skipped", "enabled": False,
                "route": None, "model_loaded": False, "output_dir": None}
    source = _source(model_or_path)
    destination = _output(source, output_dir, settings)
    if settings.ptq.enabled:
        details = validate_ptq_config(settings.ptq)
        route = "ptq"
    else:
        details = validate_qat_config(settings.qat, source=source, output_dir=destination)
        route = "qat"
    return {**details, "operation": "quantization", "status": "planned", "enabled": True,
            "route": route, "source_model": source, "output_dir": destination,
            "model_loaded": False}


def apply_quantization(model_or_path=None, tokenizer=None, *, config=None,
                       output_dir=None, dry_run: bool = False) -> QuantizationResult:
    """Apply a configured route, or return the original input when both are off.

    ``config={"ptq": {"enabled": True, "method": "awq", ...}}`` selects
    calibration-based PTQ. ``config={"qat": {"enabled": True, ...}}`` trains
    and converts the explicitly named QAT backend. Model inputs may be local
    checkpoint/Hub references or in-memory models with a supplied tokenizer.
    """
    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a boolean")
    settings = _config(config)
    plan = plan_quantization(model_or_path, config=settings, output_dir=output_dir)
    if not plan["enabled"] or dry_run:
        return QuantizationResult(model_or_path, tokenizer, plan)
    is_reference = isinstance(model_or_path, (str, PathLike))
    model = None if is_reference else model_or_path
    source = str(model_or_path) if is_reference else "<in-memory>"
    if settings.ptq.enabled:
        from .CalibratedPTQ import run_calibrated_ptq

        result = run_calibrated_ptq(source, plan["output_dir"], settings.ptq, model=model, tokenizer=tokenizer)
    else:
        from .QuantizationAwareTraining import run_qat

        result = run_qat(source, plan["output_dir"], settings.qat, model=model, tokenizer=tokenizer)
    report = {**result.report, "operation": "quantization", "enabled": True,
              "route": plan["route"], "status": "completed", "output_dir": result.output_dir,
              "model_loaded": True}
    return QuantizationResult(result.model, result.tokenizer, report, result.output_dir)


__all__ = [
    "QuantizationConfig",
    "QuantizationResult",
    "apply_quantization",
    "plan_quantization",
]
