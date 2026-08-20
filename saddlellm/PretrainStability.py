"""Stability callbacks and checkpoint checks for scratch pretraining."""
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class StabilityConfig:
    enabled: bool = True
    output_dir: str = "./stability"
    loss_window: int = 50
    spike_threshold: float = 2.5
    stagnation_window: int = 200
    stagnation_min_improvement: float = 0.01
    stop_on_nan: bool = True
    log_jsonl: bool = True


@dataclass
class StabilityEvent:
    step: int
    severity: str
    kind: str
    message: str
    metrics: Dict

    def to_dict(self) -> Dict:
        return asdict(self)


try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers is an install dependency.
    TrainerCallback = object


class PretrainStabilityCallback(TrainerCallback):
    """HuggingFace Trainer callback for basic pretraining health checks."""

    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
        self.loss_history: List[float] = []
        self.events: List[StabilityEvent] = []
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.events_path = os.path.join(self.config.output_dir, "stability_events.jsonl")
        self.summary_path = os.path.join(self.config.output_dir, "stability_summary.json")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.config.enabled or not getattr(state, "is_world_process_zero", True) or not logs:
            return control
        step = int(getattr(state, "global_step", 0) or 0)
        loss = logs.get("loss")
        if loss is None:
            return control
        try:
            loss_val = float(loss)
        except Exception:
            return control

        metrics = {k: self._jsonable(v) for k, v in logs.items()}
        event = self._check_loss(step, loss_val, metrics)
        if event:
            self._record(event)
            if event.kind == "loss_nan_inf" and self.config.stop_on_nan:
                control.should_training_stop = True
        self._write_summary()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        self._write_summary()
        return control

    def summary(self) -> Dict:
        finite_losses = [x for x in self.loss_history if math.isfinite(x)]
        return {
            "checks": len(self.loss_history),
            "events": [e.to_dict() for e in self.events],
            "event_count": len(self.events),
            "last_loss": finite_losses[-1] if finite_losses else None,
            "min_loss": min(finite_losses) if finite_losses else None,
            "trend": self._trend(finite_losses),
        }

    def _check_loss(self, step: int, loss: float, metrics: Dict) -> Optional[StabilityEvent]:
        self.loss_history.append(loss)
        if math.isnan(loss) or math.isinf(loss):
            return StabilityEvent(
                step=step,
                severity="critical",
                kind="loss_nan_inf",
                message="loss is NaN/Inf; lower learning rate, inspect data, or disable mixed precision",
                metrics=metrics,
            )

        if len(self.loss_history) > max(10, self.config.loss_window // 2):
            window = [x for x in self.loss_history[-self.config.loss_window - 1:-1] if math.isfinite(x)]
            if window:
                avg = sum(window) / len(window)
                if avg > 0 and loss > avg * self.config.spike_threshold:
                    return StabilityEvent(
                        step=step,
                        severity="warning",
                        kind="loss_spike",
                        message=f"loss spike detected: {loss:.4f} vs moving average {avg:.4f}",
                        metrics={**metrics, "moving_avg_loss": avg},
                    )

        n = self.config.stagnation_window
        if n > 0 and len(self.loss_history) >= n * 2:
            prev = [x for x in self.loss_history[-2 * n:-n] if math.isfinite(x)]
            recent = [x for x in self.loss_history[-n:] if math.isfinite(x)]
            if prev and recent:
                prev_avg = sum(prev) / len(prev)
                recent_avg = sum(recent) / len(recent)
                improvement = (prev_avg - recent_avg) / max(prev_avg, 1e-8)
                if improvement < self.config.stagnation_min_improvement:
                    return StabilityEvent(
                        step=step,
                        severity="info",
                        kind="loss_stagnation",
                        message=f"loss improved only {improvement:.2%} over last {n} logs",
                        metrics={**metrics, "prev_avg_loss": prev_avg, "recent_avg_loss": recent_avg},
                    )
        return None

    def _record(self, event: StabilityEvent):
        self.events.append(event)
        if self.config.log_jsonl:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def _write_summary(self):
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

    def _trend(self, losses: List[float]) -> str:
        if len(losses) < 4:
            return "unknown"
        width = max(1, len(losses) // 4)
        start = sum(losses[:width]) / width
        end = sum(losses[-width:]) / width
        if end < start * 0.9:
            return "decreasing"
        if end > start * 1.1:
            return "increasing"
        return "flat"

    def _jsonable(self, value):
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)


class CheckpointInspector:
    """Validate that a Trainer checkpoint has the files needed for resume."""

    REQUIRED_FILES = ("trainer_state.json", "training_args.bin")
    ONE_OF_MODEL_FILES = ("model.safetensors", "pytorch_model.bin")
    ONE_OF_OPTIMIZER_FILES = ("optimizer.pt", "optimizer.bin")

    @classmethod
    def inspect(cls, checkpoint_dir: str) -> Dict:
        missing = []
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            return {"ok": False, "checkpoint_dir": checkpoint_dir, "missing": ["checkpoint_dir"], "warnings": []}
        for name in cls.REQUIRED_FILES:
            if not os.path.exists(os.path.join(checkpoint_dir, name)):
                missing.append(name)
        if not any(os.path.exists(os.path.join(checkpoint_dir, name)) for name in cls.ONE_OF_MODEL_FILES):
            missing.append("model weights")
        warnings = []
        if not any(os.path.exists(os.path.join(checkpoint_dir, name)) for name in cls.ONE_OF_OPTIMIZER_FILES):
            warnings.append("optimizer state missing; resume may restart optimizer statistics")
        return {
            "ok": not missing,
            "checkpoint_dir": checkpoint_dir,
            "missing": missing,
            "warnings": warnings,
        }
