"""Compatibility import for :mod:`saddlellm.compression.pruner`.

New code should import from :mod:`saddlellm.compression`.
"""

from .compression.pruner import (
    PRUNING_MANIFEST_FILENAME,
    ModelPruner,
    PruningMethod,
)

__all__ = ["PRUNING_MANIFEST_FILENAME", "ModelPruner", "PruningMethod"]
