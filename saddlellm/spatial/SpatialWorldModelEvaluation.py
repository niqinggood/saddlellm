"""Metrics for action-conditioned spatial world-model rollouts."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from ..world_models.WorldModelData import WorldModelTrajectoryDataset, load_world_model_trajectories
from ..world_models.WorldModelInference import WorldModelRuntime


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
    termination_probabilities = []
    termination_targets = []
    horizon_confusions = np.zeros(
        (sequence_length, occupancy_channels, occupancy_channels), dtype=np.int64
    )
    horizon_copy_last_confusions = np.zeros_like(horizon_confusions)
    horizon_transition_counts = np.zeros(sequence_length, dtype=np.int64)
    horizon_reward_absolute_error = np.zeros(sequence_length, dtype=np.float64)
    horizon_ego_squared_error = np.zeros(sequence_length, dtype=np.float64)
    horizon_ego_absolute_error = np.zeros(sequence_length, dtype=np.float64)
    horizon_ego_count = np.zeros(sequence_length, dtype=np.int64)
    horizon_collision_probabilities = [[] for _ in range(sequence_length)]
    horizon_collision_targets = [[] for _ in range(sequence_length)]
    horizon_termination_probabilities = [[] for _ in range(sequence_length)]
    horizon_termination_targets = [[] for _ in range(sequence_length)]
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
        _accumulate_confusion(confusion, target[mask], predicted[mask])
        _accumulate_confusion(copy_last_confusion, target[mask], copy_last[mask])

        rewards = imagination.rewards[0].detach().cpu()
        reward_errors = torch.abs(rewards - item["rewards"])
        reward_absolute_error += float(reward_errors[mask].sum())
        reward_count += int(mask.sum())
        valid_transitions += int(mask.sum())

        done_probabilities = (1.0 - imagination.continuation[0].detach().cpu()).clamp(
            0.0, 1.0
        )
        done_targets = item["dones"]
        termination_probabilities.append(done_probabilities[mask].numpy())
        termination_targets.append(done_targets[mask].numpy())

        valid_steps = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        for step in valid_steps:
            horizon_transition_counts[step] += 1
            horizon_reward_absolute_error[step] += float(reward_errors[step])
            _accumulate_confusion(
                horizon_confusions[step], target[step], predicted[step]
            )
            _accumulate_confusion(
                horizon_copy_last_confusions[step], target[step], copy_last[step]
            )
            horizon_termination_probabilities[step].append(
                float(done_probabilities[step])
            )
            horizon_termination_targets[step].append(float(done_targets[step]))

        if imagination.ego_motion is not None and "ego_motions" in item:
            predicted_ego = imagination.ego_motion[0].detach().cpu()
            target_ego = item["ego_motions"]
            difference = predicted_ego[mask] - target_ego[mask]
            ego_squared_error += float(torch.square(difference).sum())
            ego_absolute_error += float(torch.abs(difference).sum())
            ego_count += int(difference.numel())
            for step in valid_steps:
                step_difference = predicted_ego[step] - target_ego[step]
                horizon_ego_squared_error[step] += float(
                    torch.square(step_difference).sum()
                )
                horizon_ego_absolute_error[step] += float(
                    torch.abs(step_difference).sum()
                )
                horizon_ego_count[step] += int(step_difference.numel())

        if (
            imagination.collision_probability is not None
            and "collisions" in item
        ):
            predicted_collision = (
                imagination.collision_probability[0].detach().cpu()
            )
            target_collision = item["collisions"]
            collision_probabilities.append(
                predicted_collision[mask].numpy()
            )
            collision_targets.append(target_collision[mask].numpy())
            for step in valid_steps:
                horizon_collision_probabilities[step].append(
                    float(predicted_collision[step])
                )
                horizon_collision_targets[step].append(float(target_collision[step]))

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
        result["collision"] = _binary_probability_metrics(probabilities, targets)
    else:
        result["collision"] = None
    result["termination"] = _binary_probability_metrics(
        np.concatenate(termination_probabilities).astype(np.float64),
        np.concatenate(termination_targets).astype(np.float64),
    )
    result["horizon_curve"] = _build_horizon_curve(
        horizon_transition_counts=horizon_transition_counts,
        horizon_confusions=horizon_confusions,
        horizon_copy_last_confusions=horizon_copy_last_confusions,
        horizon_reward_absolute_error=horizon_reward_absolute_error,
        horizon_ego_squared_error=horizon_ego_squared_error,
        horizon_ego_absolute_error=horizon_ego_absolute_error,
        horizon_ego_count=horizon_ego_count,
        horizon_collision_probabilities=horizon_collision_probabilities,
        horizon_collision_targets=horizon_collision_targets,
        horizon_termination_probabilities=horizon_termination_probabilities,
        horizon_termination_targets=horizon_termination_targets,
    )
    result["rollout_drift"] = _rollout_drift(result["horizon_curve"])
    return result


def _build_horizon_curve(
    *,
    horizon_transition_counts: np.ndarray,
    horizon_confusions: np.ndarray,
    horizon_copy_last_confusions: np.ndarray,
    horizon_reward_absolute_error: np.ndarray,
    horizon_ego_squared_error: np.ndarray,
    horizon_ego_absolute_error: np.ndarray,
    horizon_ego_count: np.ndarray,
    horizon_collision_probabilities: list,
    horizon_collision_targets: list,
    horizon_termination_probabilities: list,
    horizon_termination_targets: list,
) -> list:
    curve = []
    for index, transition_count in enumerate(horizon_transition_counts):
        count = int(transition_count)
        if count <= 0:
            continue
        occupancy = _occupancy_metrics(horizon_confusions[index])
        copy_last = _occupancy_metrics(horizon_copy_last_confusions[index])
        ego_count = int(horizon_ego_count[index])
        collision = (
            _binary_probability_metrics(
                np.asarray(horizon_collision_probabilities[index], dtype=np.float64),
                np.asarray(horizon_collision_targets[index], dtype=np.float64),
            )
            if horizon_collision_probabilities[index]
            else None
        )
        termination = _binary_probability_metrics(
            np.asarray(horizon_termination_probabilities[index], dtype=np.float64),
            np.asarray(horizon_termination_targets[index], dtype=np.float64),
        )
        curve.append(
            {
                "step": index + 1,
                "valid_transitions": count,
                "occupancy": {
                    "accuracy": occupancy["accuracy"],
                    "mean_iou": occupancy["mean_iou"],
                    "per_class_iou": occupancy["per_class_iou"],
                    "copy_last_baseline": {
                        "accuracy": copy_last["accuracy"],
                        "mean_iou": copy_last["mean_iou"],
                    },
                    "mean_iou_gain": (
                        occupancy["mean_iou"] - copy_last["mean_iou"]
                    ),
                },
                "reward": {
                    "mae": float(horizon_reward_absolute_error[index] / count)
                },
                "ego_motion": (
                    {
                        "mae": float(horizon_ego_absolute_error[index] / ego_count),
                        "rmse": float(
                            (horizon_ego_squared_error[index] / ego_count) ** 0.5
                        ),
                    }
                    if ego_count
                    else None
                ),
                "collision": collision,
                "termination": termination,
            }
        )
    return curve


def _rollout_drift(curve: list) -> Dict[str, Any]:
    if not curve:
        return {
            "first_step": None,
            "last_step": None,
            "occupancy_mean_iou_delta": None,
            "reward_mae_delta": None,
            "last_step_mean_iou_gain": None,
        }
    first = curve[0]
    last = curve[-1]
    return {
        "first_step": first["step"],
        "last_step": last["step"],
        "occupancy_mean_iou_delta": (
            last["occupancy"]["mean_iou"] - first["occupancy"]["mean_iou"]
        ),
        "reward_mae_delta": last["reward"]["mae"] - first["reward"]["mae"],
        "last_step_mean_iou_gain": last["occupancy"]["mean_iou_gain"],
    }


def _binary_probability_metrics(
    probabilities: np.ndarray, targets: np.ndarray
) -> Dict[str, Any]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probabilities.shape != targets.shape or probabilities.size == 0:
        raise ValueError("Binary probability metrics require equally sized non-empty arrays")
    predictions = probabilities >= 0.5
    positives = targets >= 0.5
    true_positive = int(np.logical_and(predictions, positives).sum())
    true_negative = int(np.logical_and(~predictions, ~positives).sum())
    false_positive = int(np.logical_and(predictions, ~positives).sum())
    false_negative = int(np.logical_and(~predictions, positives).sum())
    return {
        "count": int(probabilities.size),
        "positive_count": int(positives.sum()),
        "brier": float(np.mean(np.square(probabilities - targets))),
        "accuracy": (true_positive + true_negative) / probabilities.size,
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "positive_rate": float(positives.mean()),
        "predicted_positive_rate": float(predictions.mean()),
        "ece": _expected_calibration_error(probabilities, targets),
    }


def _accumulate_confusion(
    confusion: np.ndarray, targets: torch.Tensor, predictions: torch.Tensor
) -> None:
    classes = int(confusion.shape[0])
    target_values = targets.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    predicted_values = (
        predictions.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    )
    counts = np.bincount(
        target_values * classes + predicted_values,
        minlength=classes * classes,
    ).reshape(classes, classes)
    confusion += counts


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
