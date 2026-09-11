"""Small, import-safe helpers for capability-specific dependencies."""
from __future__ import annotations

from importlib import metadata


class OptionalDependencyError(ImportError):
    """Raised when a requested capability extra is not installed."""


def require_distribution(distribution: str, *, extra: str, capability: str) -> str:
    """Return an installed version or raise with the exact extra to install."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError as exc:
        raise OptionalDependencyError(
            f"{capability} requires {distribution}; install `saddlellm[{extra}]`."
        ) from exc


__all__ = ["OptionalDependencyError", "require_distribution"]
