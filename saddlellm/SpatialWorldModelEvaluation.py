"""Metrics for action-conditioned spatial world-model rollouts."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from .WorldModelData import WorldModelTrajectoryDataset, load_world_model_trajectories
from .WorldModelInference import WorldModelRuntime


@torch.no_grad()
def evaluate_spatial_world_model(
    checkpoint: str,
    data_path: str,
    device: str = "auto",
    sequence_length: int = 20,
    max_windows: Optional[int] = None,
    group_key: Optional[str] = None,
    group_value: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate prior-only multi-step prediction on offline spatial sequences."""

    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive")
    runtime = WorldModelRuntime.from_pretrained(checkpoint, device=device)
    model = runtime.model
    occupancy_channels = int(
        getattr(model.config, "spatial_occupancy_channels", 0)
    )
    if occupancy_channels <= 0:
        raise ValueError("Checkpoint does not contain a spatial occupancy head")
    trajectories = load_world_model_trajectories(data_path)
    if (group_key is None) != (group_value is None):
        raise ValueError("group_key and group_value must be provided together")
    if group_key is not None:
        trajectories = [
            trajectory
            for trajectory in trajectories
            if str(_nested_value(trajectory.get("metadata", {}), group_key))
            == str(group_value)
        ]
        if not trajectories:
            raise ValueError(
                f"No trajectories matched {group_key}={group_value!r}"
            )
    dataset = WorldModelTrajectoryDataset(
        trajectories,
        sequence_length=sequence_length,
        stride=sequence_length,
        action_type=model.config.action_type,
        action_dim=model.config.action_dim,
        normalize_images=False,
        pad_short_trajectories=True,
    )

    confusion = np.zeros((occupancy_channels, occupancy_channels), dtype=np.int64)
    copy_last_confusion = np.zeros_like(confusion)
    reward_absolute_error = 0.0
    reward_count = 0
    ego_squared_error = 0.0
    ego_absolute_error = 0.0
    ego_count = 0
    collision_probabilities = []
    collision_targets = []
    valid_transitions = 0
    windows = min(len(dataset), max_windows or len(dataset))

    for index in range(windows):
        item = dataset[index]
        observations = item["observations"]
        actions = item["actions"]
        mask = item["mask"].bool()
        state = runtime.encode_observation(observations[0])
        imagination = runtime.rollout(state, actions, deterministic=True)
        if imagination.occupancy_logits is None:
            raise ValueError("Checkpoint did not emit occupancy predictions")
        logits = imagination.occupancy_logits[0].detach().cpu()
        target = observations[1:, :occupancy_channels].argmax(dim=1)
        predicted = logits.argmax(dim=1)
        copy_last = observations[0, :occupancy_channels].argmax(dim=0).expand_as(
            target
        )
        for true_class in range(occupancy_channels):
            for predicted_class in range(occupancy_channels):
                confusion[true_class, predicted_class] += int(
                    (
                        (target[mask] == true_class)
                        & (predicted[mask] == predicted_class)
                    ).sum()
                )
                copy_last_confusion[true_class, predicted_class] += int(
                    (
                        (target[mask] == true_class)
                        & (copy_last[mask] == predicted_class)
                    ).sum()
                )

        rewards = imagination.rewards[0].detach().cpu()
        reward_absolute_error += float(
            torch.abs(rewards[mask] - item["rewards"][mask]).sum()
        )
        reward_count += int(mask.sum())
        valid_transitions += int(mask.sum())

        if imagination.ego_motion is not None and "ego_motions" in item:
            predicted_ego = imagination.ego_motion[0].detach().cpu()[mask]
            target_ego = item["ego_motions"][mask]
            difference = predicted_ego - target_ego
            ego_squared_error += float(torch.square(difference).sum())
            ego_absolute_error += float(torch.abs(difference).sum())
            ego_count += int(difference.numel())

        if (
            imagination.collision_probability is not None
            and "collisions" in item
        ):
            collision_probabilities.append(
                imagination.collision_probability[0].detach().cpu()[mask].numpy()
            )
            collision_targets.append(item["collisions"][mask].numpy())

    occupancy_metrics = _occupancy_metrics(confusion)
    copy_last_metrics = _occupancy_metrics(copy_last_confusion)
    result: Dict[str, Any] = {
        "status": "completed",
        "checkpoint": checkpoint,
        "data_path": data_path,
        "device": str(runtime.device),
        "windows": windows,
        "valid_transitions": valid_transitions,
        "horizon": sequence_length,
        "group_filter": (
            {"key": group_key, "value": group_value}
            if group_key is not None
            else None
        ),
        "occupancy": {
            **occupancy_metrics,
            "copy_last_baseline": copy_last_metrics,
            "mean_iou_gain": (
                occupancy_metrics["mean_iou"] - copy_last_metrics["mean_iou"]
            ),
        },
        "reward": {
            "mae": reward_absolute_error / max(reward_count, 1),
        },
        "ego_motion": (
            {
                "mae": ego_absolute_error / ego_count,
                "rmse": (ego_squared_error / ego_count) ** 0.5,
            }
            if ego_count
            else None
        ),
    }
    if collision_probabilities:
        probabilities = np.concatenate(collision_probabilities).astype(np.float64)
        targets = np.concatenate(collision_targets).astype(np.float64)
        predictions = probabilities >= 0.5
        positives = targets >= 0.5
        true_positive = int(np.logical_and(predictions, positives).sum())
        false_positive = int(np.logical_and(predictions, ~positives).sum())
        false_negative = int(np.logical_and(~predictions, positives).sum())
        result["collision"] = {
            "brier": float(np.mean(np.square(probabilities - targets))),
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "positive_rate": float(positives.mean()),
            "predicted_positive_rate": float(predictions.mean()),
            "ece": _expected_calibration_error(probabilities, targets),
        }
    else:
        result["collision"] = None
    return result


def _expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, bins: int = 10
) -> float:
    total = max(len(probabilities), 1)
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        upper_inclusive = index == bins - 1
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if upper_inclusive
            else probabilities < edges[index + 1]
        )
        if not selected.any():
            continue
        confidence = float(probabilities[selected].mean())
        frequency = float(targets[selected].mean())
        error += float(selected.sum()) / total * abs(confidence - frequency)
    return error


def _occupancy_metrics(confusion: np.ndarray) -> Dict[str, Any]:
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - intersection
    per_class_iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
    return {
        "accuracy": float(intersection.sum() / max(int(confusion.sum()), 1)),
        "mean_iou": float(per_class_iou.mean()),
        "per_class_iou": per_class_iou.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def _nested_value(value: Any, path: str) -> Any:
    for part in str(path).split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


__all__ = ["evaluate_spatial_world_model"]
