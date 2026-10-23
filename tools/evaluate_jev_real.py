"""Backward-compatible entry point for SaddleLLM typed decision evaluation.

Existing frozen experiment commands keep their arguments and output format.
The general implementation lives in ``saddlellm.decision.Evaluation``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saddlellm.decision.Evaluation import evaluate_typed, main

__all__ = ["evaluate_typed", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
