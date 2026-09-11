"""Compatibility import for :mod:`saddlellm.compression.lightweight`.

New code should import from :mod:`saddlellm.compression`.
"""

from .compression.lightweight import LIGHTWEIGHT_RECIPES, Lightweight

__all__ = ["LIGHTWEIGHT_RECIPES", "Lightweight"]
