"""Trajectory loading and sequence windows for world-model training."""

from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WorldModelDataError(ValueError):
    """Raised when an offline trajectory cannot form valid transitions."""


def load_world_model_trajectories(path: str) -> List[Dict[str, Any]]:
    """Load JSON/JSONL, NPZ, or PyTorch trajectory data.

    A trajectory contains ``observations`` with length ``T + 1`` and ``actions``
    with length ``T``.  Optional ``rewards`` and ``dones`` also have length
    ``T``.  JSONL may instead contain transition rows with ``observation``,
    ``action`` and ``next_observation``; rows are grouped by ``episode_id``.
    """

    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"World-model data file not found: {path}")
    extension = os.path.splitext(path)[1].lower()
    if extension == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise WorldModelDataError(
                        f"Invalid JSON on line {line_number} of {path}: {exc}"
                    ) from exc
        payload: Any = records
    elif extension == ".json":
        with open(path, "r", encoding="utf-8-sig") as file:
            payload = json.load(file)
    elif extension == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
    elif extension in {".pt", ".pth"}:
        payload = _torch_load_local_data(path)
    else:
        raise WorldModelDataError(
            "Unsupported world-model data file. Use .json, .jsonl, .npz, .pt, or .pth"
        )
    trajectories = normalize_world_model_trajectories(payload)
    if not trajectories:
        raise WorldModelDataError(f"No usable trajectories found in {path}")
    return trajectories


def normalize_world_model_trajectories(payload: Any) -> List[Dict[str, Any]]:
    """Normalize trajectory records or transition rows to one canonical schema."""

    if isinstance(payload, dict):
        wrapped = _first_present(payload, ("trajectories", "episodes", "data"))
        if isinstance(wrapped, (list, tuple)):
            records = list(wrapped)
        else:
            records = [payload]
    elif isinstance(payload, (list, tuple)):
        records = list(payload)
    else:
        raise WorldModelDataError(
            f"Expected a trajectory dictionary/list, got {type(payload).__name__}"
        )
    if not records:
        return []
    if not all(isinstance(record, dict) for record in records):
        raise WorldModelDataError("Every world-model record must be a dictionary")

    trajectories: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if _looks_like_transition(record):
            transitions.append(record)
            continue
        try:
            trajectories.append(_canonicalize_trajectory(record, index))
        except WorldModelDataError as exc:
            raise WorldModelDataError(f"Trajectory {index}: {exc}") from exc
    if transitions:
        trajectories.extend(_transitions_to_trajectories(transitions))
    return trajectories


def split_world_model_trajectories(
    trajectories: Sequence[Dict[str, Any]],
    validation_split: float,
    seed: int = 42,
    group_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split complete episodes, optionally holding out whole scene groups.

    ``group_key`` is read from trajectory metadata and may use dotted paths,
    for example ``"map_id"`` or ``"scene.id"``.  Holding out complete maps is
    important for spatial world models because random episode splitting from a
    single map substantially overstates generalization.
    """

    items = list(trajectories)
    if validation_split <= 0 or len(items) < 2:
        return items, []
    if validation_split >= 1:
        raise ValueError("validation_split must be smaller than 1")
    validation_indices = set()
    if group_key:
        groups: "OrderedDict[str, List[int]]" = OrderedDict()
        for index, item in enumerate(items):
            value = _nested_metadata_value(item.get("metadata", {}), group_key)
            if value is None:
                raise WorldModelDataError(
                    f"Trajectory {index} metadata is missing split group {group_key!r}"
                )
            groups.setdefault(str(value), []).append(index)
        if len(groups) < 2:
            raise WorldModelDataError(
                f"Grouped validation requires at least two distinct {group_key!r} values"
            )
        group_names = list(groups)
        random.Random(seed).shuffle(group_names)
        target_size = max(1, int(round(len(items) * validation_split)))
        for name in group_names:
            if validation_indices and len(validation_indices) >= target_size:
                break
            validation_indices.update(groups[name])
        if len(validation_indices) == len(items):
            last_group = group_names[-1]
            validation_indices.difference_update(groups[last_group])
    else:
        indices = list(range(len(items)))
        random.Random(seed).shuffle(indices)
        validation_size = max(1, int(round(len(items) * validation_split)))
        validation_size = min(validation_size, len(items) - 1)
        validation_indices = set(indices[:validation_size])
    train = [item for index, item in enumerate(items) if index not in validation_indices]
    validation = [item for index, item in enumerate(items) if index in validation_indices]
    return train, validation


def infer_world_model_dimensions(
    trajectories: Sequence[Dict[str, Any]],
    action_type: str = "continuous",
) -> Dict[str, Any]:
    """Infer observation shape and continuous width/discrete vocabulary size."""

    if not trajectories:
        raise WorldModelDataError("Cannot infer dimensions from an empty trajectory list")
    observation_shape = tuple(np.asarray(trajectories[0]["observations"]).shape[1:])
    if not observation_shape:
        raise WorldModelDataError("Observations must have at least one feature dimension")
    for index, trajectory in enumerate(trajectories[1:], 1):
        current_shape = tuple(np.asarray(trajectory["observations"]).shape[1:])
        if current_shape != observation_shape:
            raise WorldModelDataError(
                f"Trajectory {index} has observation shape {current_shape}; expected {observation_shape}"
            )

    normalized_action_type = str(action_type).lower()
    if normalized_action_type == "discrete":
        maximum = -1
        for index, trajectory in enumerate(trajectories):
            actions = np.asarray(trajectory["actions"])
            if actions.ndim == 2 and actions.shape[-1] == 1:
                actions = actions[:, 0]
            if actions.ndim != 1:
                raise WorldModelDataError(
                    f"Trajectory {index}: discrete actions must be scalar ids"
                )
            if actions.size:
                if np.min(actions) < 0:
                    raise WorldModelDataError("Discrete action ids cannot be negative")
                maximum = max(maximum, int(np.max(actions)))
        action_dim = maximum + 1
    elif normalized_action_type == "continuous":
        actions = np.asarray(trajectories[0]["actions"])
        action_dim = 1 if actions.ndim == 1 else int(np.prod(actions.shape[1:]))
        for index, trajectory in enumerate(trajectories[1:], 1):
            current = np.asarray(trajectory["actions"])
            current_dim = 1 if current.ndim == 1 else int(np.prod(current.shape[1:]))
            if current_dim != action_dim:
                raise WorldModelDataError(
                    f"Trajectory {index} has action width {current_dim}; expected {action_dim}"
                )
    else:
        raise ValueError("action_type must be 'continuous' or 'discrete'")
    if action_dim <= 0:
        raise WorldModelDataError("Could not infer a positive action dimension")
    return {
        "observation_shape": observation_shape,
        "action_dim": action_dim,
        "action_type": normalized_action_type,
    }


class WorldModelTrajectoryDataset(Dataset):
    """Create fixed-size, padded transition windows from offline episodes."""

    def __init__(
        self,
        trajectories: Sequence[Dict[str, Any]],
        sequence_length: int = 32,
        stride: Optional[int] = None,
        action_type: str = "continuous",
        action_dim: Optional[int] = None,
        normalize_images: bool = True,
        pad_short_trajectories: bool = True,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        self.trajectories = normalize_world_model_trajectories(list(trajectories))
        if not self.trajectories:
            raise WorldModelDataError("WorldModelTrajectoryDataset needs at least one trajectory")
        self.sequence_length = int(sequence_length)
        self.stride = int(stride or sequence_length)
        self.action_type = str(action_type).lower()
        if self.action_type not in {"continuous", "discrete"}:
            raise ValueError("action_type must be 'continuous' or 'discrete'")
        self.normalize_images = bool(normalize_images)
        self.pad_short_trajectories = bool(pad_short_trajectories)
        dimensions = infer_world_model_dimensions(self.trajectories, self.action_type)
        self.observation_shape = dimensions["observation_shape"]
        inferred_action_dim = dimensions["action_dim"]
        if action_dim is not None:
            if self.action_type == "discrete" and action_dim < inferred_action_dim:
                raise WorldModelDataError(
                    f"action_dim={action_dim} cannot represent observed action id "
                    f"{inferred_action_dim - 1}"
                )
            if self.action_type == "continuous" and action_dim != inferred_action_dim:
                raise WorldModelDataError(
                    f"action_dim={action_dim} does not match data width {inferred_action_dim}"
                )
        self.action_dim = int(action_dim or inferred_action_dim)
        self._windows = self._build_windows()
        if not self._windows:
            raise WorldModelDataError(
                "No sequence windows were produced; enable padding or reduce sequence_length"
            )

    @classmethod
    def from_file(
        cls,
        path: str,
        sequence_length: int = 32,
        stride: Optional[int] = None,
        action_type: str = "continuous",
        action_dim: Optional[int] = None,
        normalize_images: bool = True,
        pad_short_trajectories: bool = True,
    ) -> "WorldModelTrajectoryDataset":
        return cls(
            load_world_model_trajectories(path),
            sequence_length=sequence_length,
            stride=stride,
            action_type=action_type,
            action_dim=action_dim,
            normalize_images=normalize_images,
            pad_short_trajectories=pad_short_trajectories,
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        trajectory_index, start, valid_length = self._windows[index]
        trajectory = self.trajectories[trajectory_index]
        stop = start + valid_length

        observations = self._observation_tensor(trajectory["observations"])
        actions = self._action_tensor(trajectory["actions"])
        rewards = torch.as_tensor(trajectory["rewards"], dtype=torch.float32).reshape(-1)
        dones = torch.as_tensor(trajectory["dones"], dtype=torch.float32).reshape(-1)

        observation_window = observations[start : stop + 1]
        action_window = actions[start:stop]
        reward_window = rewards[start:stop]
        done_window = dones[start:stop]
        auxiliary_windows: Dict[str, torch.Tensor] = {}
        for key in ("ego_motions", "collisions"):
            if key in trajectory:
                values = torch.as_tensor(trajectory[key], dtype=torch.float32)
                auxiliary_windows[key] = values[start:stop]
        padding = self.sequence_length - valid_length
        if padding:
            final_observation = observation_window[-1:].expand(
                padding, *observation_window.shape[1:]
            )
            observation_window = torch.cat([observation_window, final_observation], dim=0)
            action_window = torch.cat(
                [action_window, torch.zeros((padding, *action_window.shape[1:]), dtype=action_window.dtype)],
                dim=0,
            )
            reward_window = torch.cat([reward_window, torch.zeros(padding)], dim=0)
            done_window = torch.cat([done_window, torch.ones(padding)], dim=0)
            for key, values in list(auxiliary_windows.items()):
                auxiliary_windows[key] = torch.cat(
                    [
                        values,
                        torch.zeros(
                            (padding, *values.shape[1:]), dtype=values.dtype
                        ),
                    ],
                    dim=0,
                )
        mask = torch.cat(
            [torch.ones(valid_length, dtype=torch.float32), torch.zeros(padding, dtype=torch.float32)]
        )
        result = {
            "observations": observation_window,
            "actions": action_window,
            "rewards": reward_window,
            "dones": done_window,
            "mask": mask,
            "episode_id": str(trajectory.get("episode_id", trajectory_index)),
            "start_step": torch.tensor(start, dtype=torch.long),
        }
        result.update(auxiliary_windows)
        return result

    def summary(self) -> Dict[str, Any]:
        transitions = sum(len(trajectory["actions"]) for trajectory in self.trajectories)
        valid_lengths = [window[2] for window in self._windows]
        return {
            "episodes": len(self.trajectories),
            "transitions": int(transitions),
            "windows": len(self),
            "sequence_length": self.sequence_length,
            "stride": self.stride,
            "observation_shape": list(self.observation_shape),
            "action_dim": self.action_dim,
            "action_type": self.action_type,
            "padded_windows": sum(length < self.sequence_length for length in valid_lengths),
            "auxiliary_targets": [
                key
                for key in ("ego_motions", "collisions")
                if all(key in trajectory for trajectory in self.trajectories)
            ],
        }

    def _build_windows(self) -> List[Tuple[int, int, int]]:
        windows: List[Tuple[int, int, int]] = []
        for trajectory_index, trajectory in enumerate(self.trajectories):
            transitions = len(trajectory["actions"])
            if transitions <= 0:
                continue
            if transitions < self.sequence_length:
                if self.pad_short_trajectories:
                    windows.append((trajectory_index, 0, transitions))
                continue
            starts = list(range(0, transitions - self.sequence_length + 1, self.stride))
            final_start = transitions - self.sequence_length
            if not starts or starts[-1] != final_start:
                starts.append(final_start)
            windows.extend(
                (trajectory_index, start, self.sequence_length) for start in starts
            )
        return windows

    def _observation_tensor(self, values: Any) -> torch.Tensor:
        array = _as_numpy(values)
        try:
            tensor = torch.as_tensor(array, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise WorldModelDataError(
                "Observations must be numeric arrays; image paths must be decoded before training"
            ) from exc
        if len(self.observation_shape) == 3 and self.normalize_images:
            if np.issubdtype(array.dtype, np.integer) or (
                tensor.numel() and float(tensor.detach().max()) > 1.0
            ):
                tensor = tensor / 255.0
        return tensor

    def _action_tensor(self, values: Any) -> torch.Tensor:
        array = _as_numpy(values)
        if self.action_type == "discrete":
            tensor = torch.as_tensor(array, dtype=torch.long)
            if tensor.ndim == 2 and tensor.shape[-1] == 1:
                tensor = tensor[:, 0]
            if tensor.ndim != 1:
                raise WorldModelDataError("Discrete actions must be scalar ids")
            return tensor
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor.flatten(start_dim=1)


def _canonicalize_trajectory(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    observations = _first_present(record, ("observations", "obs", "states"))
    actions = _first_present(record, ("actions",))
    if observations is None or actions is None:
        raise WorldModelDataError(
            "Expected 'observations' and 'actions', or transition rows with 'next_observation'"
        )
    observations_array = _as_numpy(observations)
    actions_array = _as_numpy(actions)
    if actions_array.ndim == 0:
        actions_array = actions_array.reshape(1)
    transitions = int(actions_array.shape[0])
    if transitions < 1:
        raise WorldModelDataError("A trajectory must contain at least one action")

    next_observations = _first_present(record, ("next_observations", "next_obs", "next_states"))
    if observations_array.ndim == 0:
        raise WorldModelDataError("observations must include a time dimension")
    if observations_array.shape[0] == transitions and next_observations is not None:
        next_array = _as_numpy(next_observations)
        if next_array.shape[0] != transitions:
            raise WorldModelDataError("next_observations must have the same length as actions")
        observations_array = np.concatenate([observations_array[:1], next_array], axis=0)
    if observations_array.shape[0] != transitions + 1:
        raise WorldModelDataError(
            f"observations length must equal actions length + 1 "
            f"({observations_array.shape[0]} vs {transitions + 1})"
        )

    rewards = _first_present(record, ("rewards", "reward"))
    dones = _first_present(record, ("dones", "terminals", "is_terminal", "done"))
    rewards_array = _transition_vector(rewards, transitions, default=0.0, name="rewards")
    dones_array = _transition_vector(dones, transitions, default=0.0, name="dones")
    normalized = {
        "observations": observations_array,
        "actions": actions_array,
        "rewards": rewards_array,
        "dones": dones_array,
        "episode_id": record.get("episode_id", record.get("id", index)),
        "metadata": record.get("metadata", {}),
    }
    collisions = _first_present(record, ("collisions", "collision", "costs"))
    if collisions is not None:
        normalized["collisions"] = _transition_vector(
            collisions, transitions, default=0.0, name="collisions"
        )
    ego_motions = _first_present(
        record, ("ego_motions", "ego_motion", "pose_deltas")
    )
    if ego_motions is not None:
        normalized["ego_motions"] = _transition_matrix(
            ego_motions, transitions, name="ego_motions"
        )
    return normalized


def _transitions_to_trajectories(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for index, record in enumerate(records):
        episode_id = str(record.get("episode_id", record.get("trajectory_id", "episode_0")))
        copied = dict(record)
        copied["_input_order"] = index
        groups.setdefault(episode_id, []).append(copied)

    trajectories = []
    for episode_id, rows in groups.items():
        rows.sort(key=lambda row: (row.get("step_id", row.get("t", row["_input_order"])), row["_input_order"]))
        segment: List[Dict[str, Any]] = []
        segment_index = 0
        for row in rows:
            segment.append(row)
            done = bool(_first_present(row, ("done", "terminal", "is_terminal")) or False)
            if done:
                trajectories.append(
                    _transition_segment_to_trajectory(episode_id, segment_index, segment)
                )
                segment = []
                segment_index += 1
        if segment:
            trajectories.append(
                _transition_segment_to_trajectory(episode_id, segment_index, segment)
            )
    return trajectories


def _transition_segment_to_trajectory(
    episode_id: str,
    segment_index: int,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    observations = [_first_present(rows[0], ("observation", "obs", "state"))]
    actions = []
    rewards = []
    dones = []
    for row in rows:
        next_observation = _first_present(row, ("next_observation", "next_obs", "next_state"))
        action = _first_present(row, ("action", "actions"))
        if next_observation is None or action is None:
            raise WorldModelDataError(
                "Transition records require observation, action, and next_observation"
            )
        observations.append(next_observation)
        actions.append(action)
        rewards.append(float(_first_present(row, ("reward", "rewards")) or 0.0))
        dones.append(float(bool(_first_present(row, ("done", "terminal", "is_terminal")) or False)))
    suffix = f"/{segment_index}" if segment_index else ""
    return _canonicalize_trajectory(
        {
            "episode_id": f"{episode_id}{suffix}",
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        },
        segment_index,
    )


def _looks_like_transition(record: Dict[str, Any]) -> bool:
    return any(key in record for key in ("next_observation", "next_obs", "next_state")) and not any(
        key in record for key in ("observations", "next_observations")
    )


def _transition_vector(
    values: Any,
    length: int,
    default: float,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.full(length, default, dtype=np.float32)
    array = _as_numpy(values)
    if array.ndim == 0:
        if length != 1:
            raise WorldModelDataError(f"{name} must have length {length}")
        array = array.reshape(1)
    array = array.reshape(-1)
    if array.shape[0] != length:
        raise WorldModelDataError(
            f"{name} length must match actions ({array.shape[0]} vs {length})"
        )
    return array.astype(np.float32, copy=False)


def _transition_matrix(values: Any, length: int, name: str) -> np.ndarray:
    array = _as_numpy(values)
    if array.ndim == 1:
        if length != 1:
            raise WorldModelDataError(f"{name} must have {length} rows")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != length:
        raise WorldModelDataError(
            f"{name} must have shape [actions, features] ({array.shape} vs {length} rows)"
        )
    return array.astype(np.float32, copy=False)


def _nested_metadata_value(metadata: Any, path: str) -> Any:
    value = metadata
    for part in str(path).split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _torch_load_local_data(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
