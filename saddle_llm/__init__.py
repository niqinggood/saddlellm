"""Backward-compatible alias for the canonical :mod:`saddlellm` package.

New code should import ``saddlellm``. Public top-level symbols and canonical
subpackages are forwarded; historical flat implementation-module paths are
not part of the compatibility contract.
"""

from __future__ import annotations

import saddlellm as _canonical

__version__ = _canonical.__version__
__all__ = _canonical.__all__
__path__ = _canonical.__path__


def __getattr__(name: str):
    return getattr(_canonical, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_canonical)))
