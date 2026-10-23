"""Runtime compatibility helpers for SaddleLLM post-training backends.

The Hugging Face post-training stack changes constructor names frequently.
This module centralizes version reporting and safe keyword adaptation so SFT
and preference trainers do not each grow their own brittle compatibility code.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
from importlib import metadata
from typing import Any, Dict, List, Mapping

from packaging.specifiers import SpecifierSet


POST_TRAINING_REQUIREMENTS: Dict[str, str] = {
    "torch": ">=2.1,<3.0",
    "transformers": ">=4.57.6,<5.0",
    "trl": ">=0.26.2,<0.27",
    "peft": ">=0.18.1,<0.19",
    "datasets": ">=3.6,<4.0",
    "accelerate": ">=1.12,<2.0",
}

_PEFT_OPTIONAL_BACKEND_WARNINGS: List[str] | None = None


def installed_post_training_versions() -> Dict[str, str | None]:
    """Return installed versions without importing large ML packages."""
    versions: Dict[str, str | None] = {}
    for package in POST_TRAINING_REQUIREMENTS:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def post_training_runtime_report() -> Dict[str, Any]:
    """Describe whether the tested post-training dependency set is active."""
    versions = installed_post_training_versions()
    issues = []
    for package, requirement in POST_TRAINING_REQUIREMENTS.items():
        version = versions[package]
        if version is None:
            issues.append(f"{package} is not installed (required {requirement})")
            continue
        if version not in SpecifierSet(requirement):
            issues.append(f"{package}=={version} is outside the tested range {requirement}")
    return {
        "compatible": not issues,
        "versions": versions,
        "requirements": dict(POST_TRAINING_REQUIREMENTS),
        "issues": issues,
        "install_command": "python -m pip install -r requirements-posttrain.txt",
    }


def format_post_training_runtime_error(feature: str, error: BaseException) -> str:
    """Create an actionable error for lazy TRL/Transformers import failures."""
    report = post_training_runtime_report()
    issue_text = "; ".join(report["issues"]) or "trainer import failed in an otherwise supported version set"
    return (
        f"{feature} is unavailable because the post-training runtime is incompatible: "
        f"{issue_text}. Install the isolated tested stack with "
        f"`{report['install_command']}`. Original error: {error}"
    )


def callable_accepts_var_kwargs(callable_obj: Any) -> bool:
    """Whether a callable explicitly accepts arbitrary keyword arguments."""
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(callable_obj).parameters.values()
    )


def supported_kwargs(callable_obj: Any, values: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop unsupported/None kwargs unless the callable exposes ``**kwargs``.

    Some deprecated TRL trainers wrap their real constructor as ``*args,
    **kwargs``. Treating the visible parameter names as an allow-list would
    accidentally remove every useful argument, so wrappers receive all
    non-None values.
    """
    cleaned = {key: value for key, value in values.items() if value is not None}
    if callable_accepts_var_kwargs(callable_obj):
        return cleaned
    accepted = set(inspect.signature(callable_obj).parameters)
    return {key: value for key, value in cleaned.items() if key in accepted}


def instantiate_supported(config_class: Any, values: Mapping[str, Any]):
    """Instantiate a version-varying config class with supported keywords."""
    return config_class(**supported_kwargs(config_class.__init__, values))


def metric_snapshot(metrics: Any) -> Dict[str, object]:
    """Return a compact JSON-safe copy of trainer metrics."""

    snapshot: Dict[str, object] = {}
    for key, value in dict(metrics or {}).items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)
        snapshot[str(key)] = value
    return snapshot


def latest_eval_metrics(log_history: Any) -> Dict[str, object]:
    """Return the latest evaluation event emitted by a Trainer."""

    for event in reversed(list(log_history or [])):
        if any(str(key).startswith("eval_") for key in event):
            return metric_snapshot(
                {
                    key: value
                    for key, value in event.items()
                    if str(key).startswith("eval_") or key in {"epoch", "step"}
                }
            )
    return {}


def latest_train_metrics(log_history: Any) -> Dict[str, object]:
    """Return the latest per-step training event emitted by a Trainer.

    ``Trainer.train().metrics['train_loss']`` is cumulative and can look
    artificially small after an exact resume because Transformers divides the
    newly accumulated loss by the restored global step.  Keep that upstream
    metric for provenance, but expose the latest logged step separately so a
    resumed run remains interpretable.
    """

    for event in reversed(list(log_history or [])):
        if "loss" in event and not any(
            str(key).startswith("eval_") for key in event
        ):
            return metric_snapshot(event)
    return {}


def resolve_trainer_resume_checkpoint(
    resume_from_checkpoint: Any,
    checkpoint_root: str,
) -> str | None:
    """Resolve ``True`` to the latest concrete Hugging Face checkpoint path."""

    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint is True:
        from transformers.trainer_utils import get_last_checkpoint

        root = os.path.abspath(os.path.expanduser(checkpoint_root))
        checkpoint = get_last_checkpoint(root) if os.path.isdir(root) else None
        if checkpoint is None:
            raise FileNotFoundError(
                "resume_from_checkpoint=true but no checkpoint-* exists in "
                f"{root}"
            )
        return os.path.abspath(checkpoint)
    if isinstance(resume_from_checkpoint, (str, os.PathLike)):
        checkpoint = os.path.abspath(
            os.path.expanduser(os.fspath(resume_from_checkpoint))
        )
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(
                f"resume checkpoint directory does not exist: {checkpoint}"
            )
        return checkpoint
    raise TypeError("resume_from_checkpoint must be false, true, or a checkpoint path")


def validate_resume_training_args(checkpoint: str | None, training_args: Any) -> None:
    """Reject a resumed run whose effective train batch size has changed.

    Transformers only warns about this mismatch and continues, but the data
    cursor then no longer represents an exact continuation.  Other scheduling
    fields such as ``max_steps`` are intentionally allowed to grow.
    """

    if not checkpoint:
        return
    state_path = os.path.join(checkpoint, "trainer_state.json")
    if not os.path.isfile(state_path):
        raise ValueError(
            "Exact resume requires trainer_state.json in the checkpoint: "
            f"{checkpoint}"
        )
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    saved_batch_size = state.get("train_batch_size")
    current_batch_size = getattr(training_args, "train_batch_size", None)
    if (
        saved_batch_size is not None
        and current_batch_size is not None
        and int(saved_batch_size) != int(current_batch_size)
    ):
        raise ValueError(
            "Exact resume train_batch_size mismatch: checkpoint saved "
            f"{saved_batch_size}, current configuration resolves to "
            f"{current_batch_size}. Keep batch size/world size unchanged or "
            "start a weights-only new run."
        )


def resolve_post_training_precision(precision: str, device: str):
    """Resolve an explicit SFT/preference precision into torch settings.

    Mixed-precision flags require CUDA. ``auto`` keeps the historical behavior:
    BF16 where supported, otherwise FP16 on CUDA and FP32 elsewhere.
    """

    import torch

    normalized = str(precision or "auto").strip().lower()
    if normalized not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("precision must be one of: auto, bf16, fp16, fp32")
    if normalized == "auto":
        if device == "cuda":
            normalized = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            normalized = "fp32"
    if normalized in {"bf16", "fp16"} and device != "cuda":
        raise RuntimeError(
            f"precision={normalized} requires CUDA for post-training; "
            "use precision=fp32 on this device"
        )
    if normalized == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "precision=bf16 was requested, but the current CUDA device does not support BF16"
        )
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[normalized]
    return normalized, dtype, normalized == "bf16", normalized == "fp16"


def stabilize_peft_optional_backends() -> List[str]:
    """Disable installed-but-broken optional PEFT quantization dispatchers.

    PEFT probes AutoAWQ by package presence and imports it while dispatching
    ordinary LoRA layers. A stale system AutoAWQ can therefore break an
    otherwise unquantized model. If the optional backend cannot actually be
    imported, mark only that dispatcher unavailable and keep core LoRA active.
    """
    global _PEFT_OPTIONAL_BACKEND_WARNINGS
    if _PEFT_OPTIONAL_BACKEND_WARNINGS is not None:
        return list(_PEFT_OPTIONAL_BACKEND_WARNINGS)

    warnings: List[str] = []
    if importlib.util.find_spec("awq") is not None:
        try:
            importlib.import_module("awq.modules.linear")
        except Exception as exc:
            try:
                import peft.import_utils as peft_import_utils
                import peft.tuners.lora.awq as peft_awq

                unavailable = lambda: False
                peft_import_utils.is_auto_awq_available = unavailable
                peft_awq.is_auto_awq_available = unavailable
                try:
                    autoawq_version = metadata.version("autoawq")
                except metadata.PackageNotFoundError:
                    autoawq_version = "unknown"
                warnings.append(
                    f"Ignored unusable optional AutoAWQ=={autoawq_version} "
                    f"({type(exc).__name__}) so normal LoRA remains available."
                )
            except Exception as patch_error:
                warnings.append(
                    "AutoAWQ is installed but unusable, and PEFT's optional dispatcher could not be isolated: "
                    f"{type(patch_error).__name__}."
                )
    _PEFT_OPTIONAL_BACKEND_WARNINGS = warnings
    return list(warnings)
