"""Small, local training-progress artifacts shared by Trainer-based stages."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from transformers import TrainerCallback


def _json_values(values: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in dict(values or {}).items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        if isinstance(value, (str, int, float, bool, type(None))):
            result[str(key)] = value
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class LocalProgressCallback(TrainerCallback):
    """Write a tiny live status file and append-only metric history.

    The files are deliberately plain JSON/JSONL so the built-in dashboard,
    shell scripts, and third-party tools can all consume the same artifacts.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir).expanduser()
        self.status_path = self.output_dir / "training_progress.json"
        self.metrics_path = self.output_dir / "training_metrics.jsonl"
        self.started_at: Optional[str] = None
        self.latest: Dict[str, Any] = {}

    @staticmethod
    def _is_main_process(state: Any) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _write(
        self,
        state: Any,
        status: str,
        logs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self._is_main_process(state):
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if logs:
            self.latest.update(_json_values(logs))
        step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        payload = {
            "schema_version": 1,
            "status": status,
            "global_step": step,
            "max_steps": max_steps,
            "progress": (step / max_steps) if max_steps > 0 else None,
            "epoch": getattr(state, "epoch", None),
            "latest": self.latest,
            "started_at": self.started_at,
            "updated_at": _now(),
        }
        if status in {"completed", "failed"}:
            payload["finished_at"] = payload["updated_at"]
        temporary = self.status_path.with_name(
            f".{self.status_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str)
                + "\n",
                encoding="utf-8",
            )
            for attempt in range(5):
                try:
                    os.replace(temporary, self.status_path)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    # Windows virus scanners and live dashboard readers can
                    # briefly hold the destination open. Keep the retry
                    # bounded so a genuine permission problem still fails.
                    time.sleep(0.01 * (2**attempt))
        finally:
            temporary.unlink(missing_ok=True)

    def on_train_begin(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        previous = {}
        if step > 0 and self.status_path.is_file():
            try:
                previous = json.loads(self.status_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                previous = {}
        self.started_at = previous.get("started_at") or _now()
        if step == 0 and self._is_main_process(state):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text("", encoding="utf-8")
        self._write(state, "running")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = _json_values(logs)
        values.setdefault("step", int(getattr(state, "global_step", 0) or 0))
        if self._is_main_process(state):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(values, ensure_ascii=False, default=str) + "\n")
        self._write(state, "running", values)
        return control

    def on_save(self, args, state, control, **kwargs):
        self._write(state, "running")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._write(state, "completed")
        return control


__all__ = ["LocalProgressCallback"]
