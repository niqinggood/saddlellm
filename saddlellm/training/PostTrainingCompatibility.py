"""Runtime compatibility helpers for SaddleLLM post-training backends.

The Hugging Face post-training stack changes constructor names frequently.
This module centralizes version reporting and safe keyword adaptation so SFT
and preference trainers do not each grow their own brittle compatibility code.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
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
