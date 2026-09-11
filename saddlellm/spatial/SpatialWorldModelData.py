"""Build native RSSM trajectories from top-down maps and planned routes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .SpatialPerception import MapExtractionConfig, TopDownMapExtractor
from .SpatialPlanner import (
    GridPathPlanner,
    OccupancyGrid,
    RouteCandidate,
    SpatialPlannerConfig,
)
from .SpatialWorldModel import (
    SpatialObservationEncoder,
    SpatialTensorObservationEncoder,
)


@dataclass
class SpatialTrajectoryConfig:
    """Feature/action/reward contract for spatial RSSM training data."""

    observation_size: int = 16
    action_dim: int = 2
    max_steps: int = 64
    progress_reward: float = 4.0
    clearance_reward: float = 0.05
    step_penalty: float = 0.01
    goal_reward: float = 1.0

    def __post_init__(self) -> None:
        minimum = len(SpatialObservationEncoder.feature_names)
        if self.observation_size < minimum:
            raise ValueError(f"observation_size must be at least {minimum}")
        if self.action_dim < 2:
            raise ValueError("action_dim must be at least 2")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.clearance_reward < 0:
            raise ValueError("clearance_reward cannot be negative")
        if self.step_penalty < 0:
            raise ValueError("step_penalty cannot be negative")


class SpatialTrajectoryBuilder:
    """Convert a collision-free route into one canonical world-model episode.

    Every observation uses ``spatial_features_v1``. The first point changes at
    each transition while the map and goal stay fixed, so the RSSM can learn
    action-conditioned progress, clearance reward, and episode termination.
    """

    def __init__(self, config: Optional[SpatialTrajectoryConfig] = None) -> None:
        self.config = config or SpatialTrajectoryConfig()
        self.encoder = SpatialObservationEncoder()

    def build(
        self,
        grid: OccupancyGrid,
        route: RouteCandidate,
        episode_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        points = _sample_route_points(route.points, self.config.max_steps + 1)
        if len(points) < 2:
            raise ValueError("A spatial trajectory needs at least two distinct points")
        goal = points[-1]
        planner = GridPathPlanner()
        clearance = planner.clearance_map(grid)
        observations = [
            self.encoder.encode(
                grid,
                point,
                goal,
                (self.config.observation_size,),
                clearance=clearance,
            ).tolist()
            for point in points
        ]
        actions: List[List[float]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        for index, (current, following) in enumerate(zip(points, points[1:])):
            dx = float(following[0] - current[0])
            dy = float(following[1] - current[1])
            norm = max(math.hypot(dx, dy), 1.0)
            action = [0.0] * self.config.action_dim
            action[0] = dx / norm
            action[1] = dy / norm
            actions.append(action)

            progress = _remaining_distance(grid, current, goal) - _remaining_distance(
                grid, following, goal
            )
            clearance_ratio = min(
                float(clearance[following[1], following[0]])
                / max(grid.width, grid.height, 1),
                1.0,
            )
            terminal = index == len(points) - 2
            reward = (
                self.config.progress_reward * progress
                + self.config.clearance_reward * clearance_ratio
                - self.config.step_penalty
                + (self.config.goal_reward if terminal else 0.0)
            )
            rewards.append(float(reward))
            dones.append(terminal)

        episode_metadata: Dict[str, Any] = {
            "schema": self.encoder.schema,
            "map": {
                "width": grid.width,
                "height": grid.height,
                "resolution": grid.resolution,
            },
            "route": {
                "label": route.label,
                "length": route.length,
                "risk": route.risk,
                "minimum_clearance": route.minimum_clearance,
            },
        }
        if metadata:
            episode_metadata.update(metadata)
        return {
            "episode_id": str(episode_id),
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "metadata": episode_metadata,
        }


@dataclass
class SpatialSequenceConfig:
    """Temporal local-map contract for ``spatial_sequence_v2`` episodes."""

    crop_size: int = 16
    sensor_radius: int = 6
    action_dim: int = 2
    max_steps: int = 64
    collision_probability: float = 0.15
    progress_reward: float = 4.0
    clearance_reward: float = 0.05
    step_penalty: float = 0.01
    goal_reward: float = 1.0
    collision_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.crop_size < 8:
            raise ValueError("crop_size must be at least 8")
        if self.sensor_radius <= 0:
            raise ValueError("sensor_radius must be positive")
        if self.action_dim < 2:
            raise ValueError("action_dim must be at least 2")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.collision_probability <= 1.0:
            raise ValueError("collision_probability must be in [0, 1]")
        for name in (
            "clearance_reward",
            "step_penalty",
            "goal_reward",
            "collision_penalty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


class SpatialSequenceTrajectoryBuilder:
    """Build partial-observation trajectories with failures and recovery.

    The builder reveals only a sensor-radius disk around the current agent and
    persists that belief over time.  Optional blocked-cell probes add explicit
    collision transitions; the following expert movement acts as recovery.
    """

    def __init__(self, config: Optional[SpatialSequenceConfig] = None) -> None:
        self.config = config or SpatialSequenceConfig()
        self.encoder = SpatialTensorObservationEncoder(
            crop_size=self.config.crop_size,
            sensor_radius=self.config.sensor_radius,
        )

    def build(
        self,
        grid: OccupancyGrid,
        route: RouteCandidate,
        episode_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        generator = rng or np.random.default_rng(0)
        route_points = _sample_route_points(route.points, self.config.max_steps + 1)
        if len(route_points) < 2:
            raise ValueError("A spatial sequence needs at least two distinct points")
        goal = route_points[-1]
        belief = self.encoder.new_belief(grid)
        clearance = GridPathPlanner().clearance_map(grid)
        current = route_points[0]
        first_delta = (
            route_points[1][0] - current[0], route_points[1][1] - current[1]
        )
        heading = math.atan2(first_delta[1], first_delta[0])
        observations = [
            self.encoder.encode(
                grid, current, goal, belief=belief, heading=heading
            ).tolist()
        ]
        actions: List[List[float]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        collisions: List[float] = []
        ego_motions: List[List[float]] = []

        def append_transition(
            action_dx: float,
            action_dy: float,
            following: Tuple[int, int],
            collided: bool,
            terminal: bool,
        ) -> None:
            nonlocal current, heading
            action_heading = math.atan2(action_dy, action_dx)
            heading_delta = _wrap_angle(action_heading - heading)
            norm = max(math.hypot(action_dx, action_dy), 1.0)
            action = [0.0] * self.config.action_dim
            action[0] = action_dx / norm
            action[1] = action_dy / norm
            actions.append(action)
            actual_dx = following[0] - current[0]
            actual_dy = following[1] - current[1]
            ego_motions.append(
                [
                    actual_dx / max(grid.width - 1, 1),
                    actual_dy / max(grid.height - 1, 1),
                    heading_delta / math.pi,
                ]
            )
            progress = _remaining_distance(grid, current, goal) - _remaining_distance(
                grid, following, goal
            )
            clearance_ratio = min(
                float(clearance[following[1], following[0]])
                / max(grid.width, grid.height, 1),
                1.0,
            )
            reward = (
                self.config.progress_reward * progress
                + self.config.clearance_reward * clearance_ratio
                - self.config.step_penalty
                - (self.config.collision_penalty if collided else 0.0)
                + (self.config.goal_reward if terminal else 0.0)
            )
            rewards.append(float(reward))
            dones.append(bool(terminal))
            collisions.append(float(collided))
            current = following
            heading = action_heading
            observations.append(
                self.encoder.encode(
                    grid, current, goal, belief=belief, heading=heading
                ).tolist()
            )

        for route_index, target in enumerate(route_points[1:]):
            if len(actions) >= self.config.max_steps:
                break
            obstacle = _adjacent_blocked_cell(grid, current, generator)
            remaining_route_steps = len(route_points) - route_index - 1
            room_for_probe = len(actions) + remaining_route_steps < self.config.max_steps
            if (
                obstacle is not None
                and room_for_probe
                and generator.random() < self.config.collision_probability
            ):
                append_transition(
                    obstacle[0] - current[0],
                    obstacle[1] - current[1],
                    current,
                    collided=True,
                    terminal=False,
                )
            terminal = route_index == len(route_points) - 2
            append_transition(
                target[0] - current[0],
                target[1] - current[1],
                target,
                collided=False,
                terminal=terminal,
            )

        if not dones or not dones[-1]:
            dones[-1] = True
            rewards[-1] += self.config.goal_reward

        episode_metadata: Dict[str, Any] = {
            "schema": self.encoder.schema,
            "observation_shape": list(self.encoder.observation_shape),
            "sensor_radius": self.config.sensor_radius,
            "map": {
                "width": grid.width,
                "height": grid.height,
                "resolution": grid.resolution,
            },
            "route": {
                "label": route.label,
                "length": route.length,
                "risk": route.risk,
                "minimum_clearance": route.minimum_clearance,
            },
        }
        if metadata:
            episode_metadata.update(metadata)
        return {
            "episode_id": str(episode_id),
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "collisions": collisions,
            "ego_motions": ego_motions,
            "metadata": episode_metadata,
        }


def generate_spatial_world_model_dataset(
    image_path: str,
    output_path: str,
    episodes: int = 64,
    routes_per_pair: int = 2,
    seed: int = 42,
    minimum_distance: float = 0.25,
    map_config: Optional[MapExtractionConfig] = None,
    planner_config: Optional[SpatialPlannerConfig] = None,
    trajectory_config: Optional[SpatialTrajectoryConfig] = None,
) -> Dict[str, Any]:
    """Generate deterministic offline navigation episodes from one map image."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if routes_per_pair <= 0 or routes_per_pair > 6:
        raise ValueError("routes_per_pair must be between 1 and 6")
    if not 0.0 <= minimum_distance <= math.sqrt(2.0):
        raise ValueError("minimum_distance must be in [0, sqrt(2)]")
    source = Path(image_path)
    if not source.is_file():
        raise FileNotFoundError(f"Spatial image not found: {source}")

    grid = TopDownMapExtractor(map_config or MapExtractionConfig()).extract(str(source))
    config = (
        replace(planner_config, route_count=routes_per_pair)
        if planner_config is not None
        else SpatialPlannerConfig(
            route_count=routes_per_pair,
            clearance_weight=0.1,
            diversity_weight=2.0,
        )
    )
    planner = GridPathPlanner(config)
    builder = SpatialTrajectoryBuilder(trajectory_config)
    free_yx = np.argwhere(grid.cells == grid.FREE)
    if len(free_yx) < 2:
        raise ValueError("The extracted map needs at least two free cells")

    rng = np.random.default_rng(seed)
    generated: List[Dict[str, Any]] = []
    sampled_pairs = set()
    attempts = 0
    maximum_attempts = max(episodes * 50, 200)
    while len(generated) < episodes and attempts < maximum_attempts:
        attempts += 1
        chosen = rng.choice(len(free_yx), size=2, replace=False)
        start_y, start_x = free_yx[int(chosen[0])]
        goal_y, goal_x = free_yx[int(chosen[1])]
        start = (int(start_x), int(start_y))
        goal = (int(goal_x), int(goal_y))
        pair = (start, goal)
        if pair in sampled_pairs:
            continue
        sampled_pairs.add(pair)
        if _remaining_distance(grid, start, goal) < minimum_distance:
            continue
        try:
            routes = planner.plan(grid, start, goal, route_count=routes_per_pair)
        except ValueError:
            continue
        for route_index, route in enumerate(routes):
            if len(generated) >= episodes:
                break
            episode_id = f"spatial-{len(generated):05d}"
            generated.append(
                builder.build(
                    grid,
                    route,
                    episode_id,
                    metadata={
                        "source_image": source.name,
                        "start": list(start),
                        "goal": list(goal),
                        "route_index": route_index,
                    },
                )
            )
    if len(generated) < episodes:
        raise ValueError(
            f"Only generated {len(generated)} of {episodes} requested episodes; "
            "use a larger connected map or lower minimum_distance"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for episode in generated:
            stream.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    transitions = sum(len(episode["actions"]) for episode in generated)
    return {
        "status": "completed",
        "output": str(destination.resolve()),
        "source_image": str(source.resolve()),
        "episodes": len(generated),
        "transitions": transitions,
        "observation_shape": [builder.config.observation_size],
        "action_dim": builder.config.action_dim,
        "schema": builder.encoder.schema,
        "map": grid.to_dict(include_cells=False),
        "seed": seed,
    }


def generate_spatial_sequence_dataset(
    image_paths: Union[str, Sequence[str]],
    output_path: str,
    episodes: int = 64,
    routes_per_pair: int = 2,
    seed: int = 42,
    minimum_distance: float = 0.25,
    map_config: Optional[MapExtractionConfig] = None,
    planner_config: Optional[SpatialPlannerConfig] = None,
    sequence_config: Optional[SpatialSequenceConfig] = None,
) -> Dict[str, Any]:
    """Generate v2 temporal episodes across one or more top-down maps."""

    sources = [image_paths] if isinstance(image_paths, str) else list(image_paths)
    sources = [Path(value) for value in sources]
    if not sources:
        raise ValueError("At least one spatial image is required")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Spatial image not found: {source}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if routes_per_pair <= 0 or routes_per_pair > 6:
        raise ValueError("routes_per_pair must be between 1 and 6")
    if not 0.0 <= minimum_distance <= math.sqrt(2.0):
        raise ValueError("minimum_distance must be in [0, sqrt(2)]")

    extractor = TopDownMapExtractor(map_config or MapExtractionConfig())
    grids = [extractor.extract(str(source)) for source in sources]
    config = (
        replace(planner_config, route_count=routes_per_pair)
        if planner_config is not None
        else SpatialPlannerConfig(
            route_count=routes_per_pair,
            clearance_weight=0.1,
            diversity_weight=2.0,
        )
    )
    planner = GridPathPlanner(config)
    builder = SpatialSequenceTrajectoryBuilder(sequence_config)
    rng = np.random.default_rng(seed)
    generated: List[Dict[str, Any]] = []
    attempts = 0
    maximum_attempts = max(episodes * 80, 400)
    sampled_pairs = set()

    while len(generated) < episodes and attempts < maximum_attempts:
        attempts += 1
        map_index = (attempts - 1) % len(grids)
        grid = grids[map_index]
        source = sources[map_index]
        free_yx = np.argwhere(grid.cells == grid.FREE)
        if len(free_yx) < 2:
            continue
        chosen = rng.choice(len(free_yx), size=2, replace=False)
        start_y, start_x = free_yx[int(chosen[0])]
        goal_y, goal_x = free_yx[int(chosen[1])]
        start = (int(start_x), int(start_y))
        goal = (int(goal_x), int(goal_y))
        pair = (map_index, start, goal)
        if pair in sampled_pairs:
            continue
        sampled_pairs.add(pair)
        if _remaining_distance(grid, start, goal) < minimum_distance:
            continue
        try:
            routes = planner.plan(grid, start, goal, route_count=routes_per_pair)
        except ValueError:
            continue
        for route_index, route in enumerate(routes):
            if len(generated) >= episodes:
                break
            episode_id = f"spatial-v2-{len(generated):05d}"
            generated.append(
                builder.build(
                    grid,
                    route,
                    episode_id,
                    metadata={
                        "source_image": source.name,
                        "map_id": f"{map_index}:{source.stem}",
                        "start": list(start),
                        "goal": list(goal),
                        "route_index": route_index,
                    },
                    rng=rng,
                )
            )
    if len(generated) < episodes:
        raise ValueError(
            f"Only generated {len(generated)} of {episodes} requested v2 episodes; "
            "use larger connected maps or lower minimum_distance"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for episode in generated:
            stream.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    transitions = sum(len(episode["actions"]) for episode in generated)
    collisions = sum(sum(episode["collisions"]) for episode in generated)
    return {
        "status": "completed",
        "output": str(destination.resolve()),
        "source_images": [str(source.resolve()) for source in sources],
        "maps": len(sources),
        "episodes": len(generated),
        "transitions": transitions,
        "collisions": int(collisions),
        "observation_shape": list(builder.encoder.observation_shape),
        "action_dim": builder.config.action_dim,
        "schema": builder.encoder.schema,
        "seed": seed,
    }


def _sample_route_points(
    points: Sequence[Sequence[int]], maximum_points: int
) -> List[tuple]:
    normalized = [(int(point[0]), int(point[1])) for point in points]
    if len(normalized) > maximum_points:
        indices = np.linspace(0, len(normalized) - 1, maximum_points).round().astype(int)
        normalized = [normalized[int(index)] for index in indices]
    distinct = []
    for point in normalized:
        if not distinct or point != distinct[-1]:
            distinct.append(point)
    return distinct


def _remaining_distance(
    grid: OccupancyGrid, point: Sequence[int], goal: Sequence[int]
) -> float:
    dx = (float(goal[0]) - float(point[0])) / max(grid.width - 1, 1)
    dy = (float(goal[1]) - float(point[1])) / max(grid.height - 1, 1)
    return math.hypot(dx, dy)


def _adjacent_blocked_cell(
    grid: OccupancyGrid,
    point: Sequence[int],
    rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    candidates = []
    for dx, dy in (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ):
        candidate = (int(point[0]) + dx, int(point[1]) + dy)
        if grid.in_bounds(candidate) and grid.cell(candidate) == grid.BLOCKED:
            candidates.append(candidate)
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


__all__ = [
    "SpatialTrajectoryConfig",
    "SpatialTrajectoryBuilder",
    "generate_spatial_world_model_dataset",
    "SpatialSequenceConfig",
    "SpatialSequenceTrajectoryBuilder",
    "generate_spatial_sequence_dataset",
]
