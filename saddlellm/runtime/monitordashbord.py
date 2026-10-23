"""Backward-compatible entry point for the unified training dashboard.

The historical module name is intentionally retained so existing uvicorn
commands keep working. New code should use :mod:`saddlellm.runtime.WebUI` or
``saddlellm dashboard``.
"""
from __future__ import annotations

import os

from .WebUI import collect_run_status, create_dashboard_app


app = create_dashboard_app(os.environ.get("SADDLELLM_OUTPUT_DIR", "."))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7860)


__all__ = ["app", "collect_run_status", "create_dashboard_app"]
