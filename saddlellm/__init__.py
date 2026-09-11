"""Public SaddleLLM API.

Implementation modules are imported only when their public symbols are
accessed. Keeping the export registry in one place prevents the package API
and lazy-import behavior from drifting apart.
"""

from importlib import import_module
from typing import Any

from ._exports import LAZY_IMPORTS as _LAZY_IMPORT_MAP

__version__ = "2.35"


__all__ = ["__version__", *_LAZY_IMPORT_MAP]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORT_MAP:
        mod_path, attr = _LAZY_IMPORT_MAP[name]
        module = import_module(mod_path, package=__package__)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_IMPORT_MAP))
