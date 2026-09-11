"""Compatibility shim for build frontends that still discover ``setup.py``.

All package metadata and dependencies live in ``pyproject.toml``. Build and
installation flows must use ``python -m build`` and ``python -m pip`` so the
declared build-system version is honored.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
