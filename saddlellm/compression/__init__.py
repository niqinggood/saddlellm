"""Model compression APIs with lazy implementation imports.

The canonical implementations live in this package. Legacy top-level modules
remain as compatibility shims for existing SaddleLLM integrations.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PTQConfig": (".CalibratedPTQ", "PTQConfig"),
    "load_calibrated_ptq": (".CalibratedPTQ", "load_calibrated_ptq"),
    "QATConfig": (".QuantizationAwareTraining", "QATConfig"),
    "load_qat": (".QuantizationAwareTraining", "load_qat"),
    "QuantizationConfig": (".QuantizationPipeline", "QuantizationConfig"),
    "QuantizationStageConfig": (".QuantizationStage", "QuantizationStageConfig"),
    "QuantizationResult": (".QuantizationPipeline", "QuantizationResult"),
    "apply_quantization": (".QuantizationPipeline", "apply_quantization"),
    "plan_quantization": (".QuantizationPipeline", "plan_quantization"),
    "RecoveryConfig": (".RecoveryTrainer", "RecoveryConfig"),
    "run_recovery": (".RecoveryTrainer", "run_recovery"),
    "validate_recovery_config": (".RecoveryTrainer", "validate_recovery_config"),
    "LIGHTWEIGHT_RECIPES": (".lightweight", "LIGHTWEIGHT_RECIPES"),
    "Lightweight": (".lightweight", "Lightweight"),
    "PRUNING_MANIFEST_FILENAME": (".pruner", "PRUNING_MANIFEST_FILENAME"),
    "ModelPruner": (".pruner", "ModelPruner"),
    "PruningMethod": (".pruner", "PruningMethod"),
    "QUANTIZATION_MANIFEST_FILENAME": (
        ".quantizer",
        "QUANTIZATION_MANIFEST_FILENAME",
    ),
    "QUANTIZED_STATE_FILENAME": (".quantizer", "QUANTIZED_STATE_FILENAME"),
    "ModelQuantizer": (".quantizer", "ModelQuantizer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_path, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_path, package=__name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
