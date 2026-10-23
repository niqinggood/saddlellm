"""Held-out temperature fitting and descriptive multiclass probability metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _inputs(logits: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("logits must have shape [nonempty samples, at least two choices]")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    if targets.shape != (values.shape[0],) or targets.dtype.kind not in "iu":
        raise ValueError("labels must be one integer per sample")
    if np.any(targets < 0) or np.any(targets >= values.shape[1]):
        raise ValueError("label index is outside the candidate set")
    return values, targets.astype(np.int64, copy=False)


def probabilities(logits: Any, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("logits must be a nonempty finite matrix")
    values = (values - values.max(axis=1, keepdims=True)) / temperature
    exponentials = np.exp(values)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    values = (logits - logits.max(axis=1, keepdims=True)) / temperature
    log_normalizer = np.log(np.exp(values).sum(axis=1))
    return float(np.mean(log_normalizer - values[np.arange(len(labels)), labels]))


def fit_temperature(logits: Any, labels: Any) -> dict[str, Any]:
    """Fit one scalar on a dedicated calibration split, never on test labels.

    Minimization is deterministic and bounded to T in [0.01, 100]. The
    uncalibrated T=1 candidate is retained, so calibration NLL cannot worsen.
    Improvement on this split does not guarantee improvement on new data.
    """
    values, targets = _inputs(logits, labels)
    left, right = math.log(0.01), math.log(100.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1, f2 = (_nll(values, targets, math.exp(x)) for x in (x1, x2))
    for _ in range(80):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = _nll(values, targets, math.exp(x1))
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = _nll(values, targets, math.exp(x2))
    candidates = [1.0, 0.01, 100.0, math.exp((left + right) / 2.0)]
    temperature = min(candidates, key=lambda value: _nll(values, targets, value))
    return {
        "method": "scalar_temperature_nll",
        "temperature": temperature,
        "samples": len(targets),
        "classes": values.shape[1],
        "nll_before": _nll(values, targets, 1.0),
        "nll_after": _nll(values, targets, temperature),
        "temperature_bounds": [0.01, 100.0],
        "at_bound": temperature in (0.01, 100.0),
        "scope": "Only the supplied held-out task/candidate distribution",
    }


def probability_metrics(
    logits: Any,
    labels: Any,
    *,
    temperature: float = 1.0,
    bins: int = 15,
    thresholds: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> dict[str, Any]:
    """Report top-label ECE and Brier=sum over classes, mean over examples."""
    values, targets = _inputs(logits, labels)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    if any(not math.isfinite(t) or t < 0 or t > 1 for t in thresholds):
        raise ValueError("coverage thresholds must lie in [0, 1]")
    probs = probabilities(values, temperature)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predictions == targets
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    reliability = []
    ece = 0.0
    bucket_ids = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    for index in range(bins):
        selected = bucket_ids == index
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean_probability = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(targets) * abs(accuracy - mean_probability)
        reliability.append({
            "lower": index / bins, "upper": (index + 1) / bins,
            "count": count, "accuracy": accuracy, "mean_probability": mean_probability,
        })
    class_metrics = []
    for index in range(values.shape[1]):
        true_positive = int(((predictions == index) & (targets == index)).sum())
        support = int((targets == index).sum())
        predicted = int((predictions == index).sum())
        denominator = support + predicted
        class_metrics.append({
            "index": index, "support": support,
            "f1": 2 * true_positive / denominator if denominator else 0.0,
            "recall": true_positive / support if support else None,
        })
    selective = []
    for threshold in thresholds:
        selected = confidence >= threshold
        count = int(selected.sum())
        selective.append({
            "threshold": threshold, "accepted": count, "coverage": count / len(targets),
            "error_rate": float((~correct[selected]).mean()) if count else None,
            "accuracy": float(correct[selected].mean()) if count else None,
        })
    return {
        "samples": len(targets), "classes": values.shape[1],
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean([row["f1"] for row in class_metrics])),
        "nll": _nll(values, targets, temperature),
        "multiclass_brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
        "brier_definition": "mean_examples(sum_classes((probability - one_hot)^2))",
        "top_label_ece": float(ece), "ece_bins": bins,
        "temperature": temperature, "reliability": reliability,
        "selective_risk": selective, "per_class": class_metrics,
        "interpretation": "Descriptive held-out metrics, not a correctness guarantee per decision",
    }
