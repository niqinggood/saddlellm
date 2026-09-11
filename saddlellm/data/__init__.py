"""Dataset collection, cleaning, mixing, and text-processing APIs."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DataMixConfig": (".pipeline", "DataMixConfig"),
    "DataMixer": (".pipeline", "DataMixer"),
    "DataPipeline": (".pipeline", "DataPipeline"),
    "PipelineConfig": (".pipeline", "PipelineConfig"),
    "DataSourceHandler": (".text_processing", "DataSourceHandler"),
    "TextCleaner": (".text_processing", "TextCleaner"),
    "TextProcessor": (".text_processing", "TextProcessor"),
    "load_text_config": (".text_processing", "load_config"),
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
