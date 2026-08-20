"""Coordinator joining visual semantics, maps, planning, and learned dynamics."""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .SpatialPerception import (
    SpatialAnalysis,
    TopDownMapExtractor,
)
from .SpatialPlanner import (
    GridPathPlanner,
    GridPoint,
    OccupancyGrid,
    RouteCandidate,
)
from ._WorldModel import RSSMState


@dataclass
class SpatialPlanResult:
    image_path: str
    grid: OccupancyGrid
    analysis: SpatialAnalysis
    start: GridPoint
    goal: GridPoint
    routes: List[RouteCandidate]
    instruction: str = ""
    world_model: Optional[Dict[str, Any]] = None

    def to_dict(self, include_grid: bool = False) -> Dict[str, Any]:
        routes = []
        for index, route in enumerate(self.routes):
            data = route.to_dict()
            data["id"] = index + 1
            data["source_points"] = [
                list(self.grid.scale_to_source(point)) for point in route.points
            ]
            routes.append(data)
        result = {
            "image_path": os.path.abspath(self.image_path),
            "instruction": self.instruction,
            "map": self.grid.to_dict(include_cells=include_grid),
            "analysis": self.analysis.to_dict(),
            "start": list(self.start),
            "goal": list(self.goal),
            "start_source": list(self.grid.scale_to_source(self.start)),
            "goal_source": list(self.grid.scale_to_source(self.goal)),
            "routes": routes,
        }
        if self.world_model is not None:
            result["world_model"] = self.world_model
        return result


class SpatialObservationEncoder:
    """Encode a navigation map and task into a flat RSSM observation.

    ``spatial_features_v1`` is intentionally compact and deterministic so the
    exact same representation can be used by dataset conversion, training,
    API inference, and route re-scoring.  A spatial RSSM checkpoint must have
    a one-dimensional observation shape with at least 16 values.
    """

    schema = "spatial_features_v1"
    feature_names = (
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "delta_x",
        "delta_y",
        "direct_distance",
        "heading_sin",
        "heading_cos",
        "free_ratio",
        "blocked_ratio",
        "unknown_ratio",
        "start_clearance",
        "goal_clearance",
        "map_aspect",
        "cell_resolution",
    )

    def encode(
        self,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
        observation_shape: Sequence[int],
        clearance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        shape = tuple(int(value) for value in observation_shape)
        if len(shape) != 1 or shape[0] < len(self.feature_names):
            raise ValueError(
                "Spatial RSSM scoring requires a flat observation_shape with "
                f"at least {len(self.feature_names)} values"
            )
        if not grid.in_bounds(start) or not grid.in_bounds(goal):
            raise ValueError("Spatial observation points must be inside the map")
        if clearance is None:
            clearance = GridPathPlanner().clearance_map(grid)
        width_scale = max(grid.width - 1, 1)
        height_scale = max(grid.height - 1, 1)
        dx = (goal[0] - start[0]) / width_scale
        dy = (goal[1] - start[1]) / height_scale
        heading = math.atan2(dy, dx)
        counts = np.bincount(grid.cells.reshape(-1), minlength=3).astype(np.float32)
        ratios = counts / max(float(counts.sum()), 1.0)
        clearance_scale = max(grid.width, grid.height, 1)
        features = np.asarray(
            [
                start[0] / width_scale,
                start[1] / height_scale,
                goal[0] / width_scale,
                goal[1] / height_scale,
                dx,
                dy,
                math.hypot(dx, dy),
                math.sin(heading),
                math.cos(heading),
                ratios[grid.FREE],
                ratios[grid.BLOCKED],
                ratios[grid.UNKNOWN],
                float(clearance[start[1], start[0]]) / clearance_scale,
                float(clearance[goal[1], goal[0]]) / clearance_scale,
                grid.width / max(grid.height, 1),
                math.log1p(grid.resolution),
            ],
            dtype=np.float32,
        )
        output = np.zeros(shape[0], dtype=np.float32)
        output[: features.size] = features
        return output.reshape(shape)

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "minimum_size": len(self.feature_names),
            "features": list(self.feature_names),
        }


class SpatialBeliefMap:
    """Persistent partially observed occupancy belief for one navigation run."""

    def __init__(self, grid: OccupancyGrid) -> None:
        self.cells = np.full_like(grid.cells, grid.UNKNOWN, dtype=np.uint8)
        self.resolution = float(grid.resolution)
        self.source_size = grid.source_size

    def observe(
        self,
        ground_truth: OccupancyGrid,
        center: GridPoint,
        radius: int,
    ) -> "SpatialBeliefMap":
        if radius <= 0:
            raise ValueError("belief observation radius must be positive")
        if self.cells.shape != ground_truth.cells.shape:
            raise ValueError("belief and ground-truth map shapes must match")
        cx, cy = center
        y0, y1 = max(0, cy - radius), min(ground_truth.height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(ground_truth.width, cx + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        visible = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        target = self.cells[y0:y1, x0:x1]
        source = ground_truth.cells[y0:y1, x0:x1]
        target[visible] = source[visible]
        return self

    def as_grid(self) -> OccupancyGrid:
        return OccupancyGrid(
            self.cells.copy(),
            resolution=self.resolution,
            source_size=self.source_size,
        )

    def copy(self) -> "SpatialBeliefMap":
        duplicate = object.__new__(SpatialBeliefMap)
        duplicate.cells = self.cells.copy()
        duplicate.resolution = self.resolution
        duplicate.source_size = self.source_size
        return duplicate


class SpatialTensorObservationEncoder:
    """Encode a local probabilistic map as a compact CHW observation.

    The first three channels are mutually exclusive occupancy classes and are
    the targets used by the dedicated future-occupancy head.  Remaining
    channels retain navigation context without requiring the network to infer
    coordinates from a single marker.
    """

    schema = "spatial_sequence_v2"
    channel_names = (
        "free",
        "blocked",
        "unknown",
        "clearance",
        "agent",
        "goal",
        "dynamic",
        "goal_delta_x",
        "goal_delta_y",
        "heading_sin",
        "heading_cos",
    )

    def __init__(self, crop_size: int = 16, sensor_radius: int = 6) -> None:
        if crop_size < 8:
            raise ValueError("crop_size must be at least 8")
        if sensor_radius <= 0:
            raise ValueError("sensor_radius must be positive")
        self.crop_size = int(crop_size)
        self.sensor_radius = int(sensor_radius)

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return len(self.channel_names), self.crop_size, self.crop_size

    def new_belief(self, grid: OccupancyGrid) -> SpatialBeliefMap:
        return SpatialBeliefMap(grid)

    def encode(
        self,
        grid: OccupancyGrid,
        current: GridPoint,
        goal: GridPoint,
        observation_shape: Optional[Sequence[int]] = None,
        belief: Optional[SpatialBeliefMap] = None,
        heading: float = 0.0,
        dynamic: Optional[np.ndarray] = None,
        update_belief: bool = True,
    ) -> np.ndarray:
        expected = self.observation_shape
        shape = tuple(int(value) for value in (observation_shape or expected))
        if shape != expected:
            raise ValueError(
                f"{self.schema} requires observation_shape={expected}, got {shape}"
            )
        if not grid.in_bounds(current) or not grid.in_bounds(goal):
            raise ValueError("Spatial tensor points must be inside the map")
        if belief is not None and update_belief:
            belief.observe(grid, current, self.sensor_radius)
        cells = belief.cells if belief is not None else grid.cells
        if dynamic is not None and np.asarray(dynamic).shape != grid.cells.shape:
            raise ValueError("dynamic occupancy must match the map shape")

        size = self.crop_size
        center = size // 2
        origin_x = current[0] - center
        origin_y = current[1] - center
        local = np.full((size, size), grid.UNKNOWN, dtype=np.uint8)
        local_dynamic = np.zeros((size, size), dtype=np.float32)
        for local_y in range(size):
            source_y = origin_y + local_y
            if not 0 <= source_y < grid.height:
                continue
            for local_x in range(size):
                source_x = origin_x + local_x
                if not 0 <= source_x < grid.width:
                    continue
                local[local_y, local_x] = cells[source_y, source_x]
                if dynamic is not None:
                    local_dynamic[local_y, local_x] = float(
                        np.asarray(dynamic)[source_y, source_x]
                    )

        output = np.zeros(expected, dtype=np.float32)
        output[0] = local == grid.FREE
        output[1] = local == grid.BLOCKED
        output[2] = local == grid.UNKNOWN
        local_grid = OccupancyGrid(local, resolution=grid.resolution)
        clearance = GridPathPlanner().clearance_map(local_grid)
        output[3] = np.clip(clearance / max(float(size), 1.0), 0.0, 1.0)
        output[4, center, center] = 1.0

        goal_x = goal[0] - origin_x
        goal_y = goal[1] - origin_y
        if not (0 <= goal_x < size and 0 <= goal_y < size):
            delta_x = goal[0] - current[0]
            delta_y = goal[1] - current[1]
            scale = max(abs(delta_x), abs(delta_y), 1)
            goal_x = center + int(round(delta_x / scale * max(center - 1, 1)))
            goal_y = center + int(round(delta_y / scale * max(center - 1, 1)))
        goal_x = min(max(int(goal_x), 0), size - 1)
        goal_y = min(max(int(goal_y), 0), size - 1)
        output[5, goal_y, goal_x] = 1.0
        output[6] = np.clip(local_dynamic, 0.0, 1.0)
        normalized_goal_x = np.clip(
            (goal[0] - current[0]) / max(grid.width - 1, 1), -1.0, 1.0
        )
        normalized_goal_y = np.clip(
            (goal[1] - current[1]) / max(grid.height - 1, 1), -1.0, 1.0
        )
        output[7].fill((normalized_goal_x + 1.0) / 2.0)
        output[8].fill((normalized_goal_y + 1.0) / 2.0)
        output[9].fill((math.sin(float(heading)) + 1.0) / 2.0)
        output[10].fill((math.cos(float(heading)) + 1.0) / 2.0)
        return output

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "observation_shape": list(self.observation_shape),
            "channels": list(self.channel_names),
            "sensor_radius": self.sensor_radius,
        }


class WorldModelRouteScorer:
    """Use a trained SaddleLLM world model to rescore geometric routes."""

    def __init__(
        self,
        runtime: Any,
        action_encoder: Optional[Callable[[RouteCandidate], Any]] = None,
        discount: float = 0.99,
        max_steps: int = 64,
        geometric_cost_weight: float = 0.05,
        risk_weight: float = 1.0,
        learned_collision_weight: float = 5.0,
    ) -> None:
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if learned_collision_weight < 0:
            raise ValueError("learned_collision_weight cannot be negative")
        self.runtime = runtime
        self.action_encoder = action_encoder
        self.discount = float(discount)
        self.max_steps = int(max_steps)
        self.geometric_cost_weight = float(geometric_cost_weight)
        self.risk_weight = float(risk_weight)
        self.learned_collision_weight = float(learned_collision_weight)

    def score(self, state: RSSMState, routes: List[RouteCandidate]) -> None:
        for route in routes:
            actions = (
                self.action_encoder(route)
                if self.action_encoder is not None
                else self._continuous_actions(route)
            )
            imagination = self.runtime.rollout(state, actions, deterministic=True)
            rewards = imagination.rewards[0]
            continuation = imagination.continuation[0]
            horizon = rewards.shape[0]
            discounts = torch.pow(
                rewards.new_tensor(self.discount),
                torch.arange(horizon, device=rewards.device),
            )
            survival = torch.cumprod(
                torch.cat(
                    [
                        torch.ones(1, device=rewards.device, dtype=rewards.dtype),
                        continuation[:-1],
                    ]
                ),
                dim=0,
            )
            predicted_return = float((rewards * discounts * survival).sum().cpu())
            collision_probability = getattr(
                imagination, "collision_probability", None
            )
            learned_collision_cost = (
                float(collision_probability[0].sum().cpu())
                if collision_probability is not None
                else 0.0
            )
            route.predicted_return = predicted_return
            route.combined_score = (
                predicted_return
                - self.geometric_cost_weight * route.cost
                - self.risk_weight * route.risk
                - self.learned_collision_weight * learned_collision_cost
            )
        routes.sort(
            key=lambda route: route.combined_score
            if route.combined_score is not None
            else -float("inf"),
            reverse=True,
        )
        for index, route in enumerate(routes):
            route.label = "world-model-best" if index == 0 else "world-model-alternative"

    def _continuous_actions(self, route: RouteCandidate) -> torch.Tensor:
        config = self.runtime.model.config
        if config.action_type != "continuous" or config.action_dim < 2:
            raise ValueError(
                "Automatic route scoring requires a continuous world model with action_dim >= 2; "
                "provide action_encoder for other action spaces"
            )
        points = _downsample_points(route.points, self.max_steps + 1)
        actions = torch.zeros(max(len(points) - 1, 1), config.action_dim)
        for index, (first, second) in enumerate(zip(points, points[1:])):
            dx = float(second[0] - first[0])
            dy = float(second[1] - first[1])
            norm = max(math.hypot(dx, dy), 1.0)
            actions[index, 0] = dx / norm
            actions[index, 1] = dy / norm
        return actions


class SpatialWorldModelCoordinator:
    """Plan routes from a top-down image with optional VLM and RSSM scoring."""

    def __init__(
        self,
        extractor: Optional[TopDownMapExtractor] = None,
        planner: Optional[GridPathPlanner] = None,
        analyzer: Optional[Any] = None,
        route_scorer: Optional[WorldModelRouteScorer] = None,
        snap_radius: int = 24,
        allow_perspective: bool = False,
    ) -> None:
        self.extractor = extractor or TopDownMapExtractor()
        self.planner = planner or GridPathPlanner()
        self.analyzer = analyzer
        self.route_scorer = route_scorer
        self.snap_radius = int(snap_radius)
        self.allow_perspective = bool(allow_perspective)

    def plan_image(
        self,
        image_path: str,
        start: Union[str, Sequence[float]],
        goal: Union[str, Sequence[float]],
        instruction: str = "",
        route_count: Optional[int] = None,
        world_state: Optional[RSSMState] = None,
    ) -> SpatialPlanResult:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Spatial image not found: {image_path}")
        grid = self.extractor.extract(image_path)
        analysis = (
            self.analyzer.analyze(image_path, instruction)
            if self.analyzer is not None
            else SpatialAnalysis(
                image_type="top_down",
                summary="Geometry extracted from image luminance.",
                confidence=0.5,
                caveats=[
                    "No vision-language analyzer was configured; semantic labels are unavailable."
                ],
            )
        )
        if analysis.image_type == "perspective" and not self.allow_perspective:
            raise ValueError(
                "A single perspective image cannot define a complete navigation map. "
                "Use a top-down map or enable an explicitly approximate workflow."
            )
        return self.plan_grid(
            grid,
            analysis,
            start=start,
            goal=goal,
            instruction=instruction,
            route_count=route_count,
            world_state=world_state,
            image_path=image_path,
        )

    def plan_grid(
        self,
        grid: OccupancyGrid,
        analysis: SpatialAnalysis,
        start: Union[str, Sequence[float]],
        goal: Union[str, Sequence[float]],
        instruction: str = "",
        route_count: Optional[int] = None,
        world_state: Optional[RSSMState] = None,
        image_path: str = "",
    ) -> SpatialPlanResult:
        """Plan on an already constructed world state.

        Keeping this operation separate from :meth:`plan_image` lets agents
        analyze an observation once, retain the resulting state, and produce
        multiple plans without decoding or interpreting the image again.
        """

        if not isinstance(grid, OccupancyGrid):
            raise TypeError("grid must be an OccupancyGrid")
        if not isinstance(analysis, SpatialAnalysis):
            raise TypeError("analysis must be a SpatialAnalysis")
        if analysis.image_type == "perspective" and not self.allow_perspective:
            raise ValueError(
                "A single perspective image cannot define a complete navigation map. "
                "Use a top-down map or enable an explicitly approximate workflow."
            )
        start_point = self._resolve_point(start, grid, analysis)
        goal_point = self._resolve_point(goal, grid, analysis)
        start_point = self._snap_to_traversable(grid, start_point)
        goal_point = self._snap_to_traversable(grid, goal_point)
        routes = self.planner.plan(
            grid,
            start_point,
            goal_point,
            route_count=route_count,
        )
        if self.route_scorer is not None:
            if world_state is None:
                raise ValueError("world_state is required when route_scorer is configured")
            self.route_scorer.score(world_state, routes)
        return SpatialPlanResult(
            image_path=image_path,
            grid=grid,
            analysis=analysis,
            start=start_point,
            goal=goal_point,
            routes=routes,
            instruction=instruction,
        )

    def _resolve_point(
        self,
        value: Union[str, Sequence[float]],
        grid: OccupancyGrid,
        analysis: SpatialAnalysis,
    ) -> GridPoint:
        if not isinstance(value, str):
            return grid.scale_from_source(value)
        normalized = value.strip().lower()
        matches = [
            entity for entity in analysis.entities if entity.name.strip().lower() == normalized
        ]
        if not matches:
            raise ValueError(f"No spatial entity named {value!r} was found")
        x, y = matches[0].center
        return (
            min(max(int(round(x / 1000.0 * grid.width)), 0), grid.width - 1),
            min(max(int(round(y / 1000.0 * grid.height)), 0), grid.height - 1),
        )

    def _snap_to_traversable(
        self, grid: OccupancyGrid, point: GridPoint
    ) -> GridPoint:
        allow_unknown = self.planner.config.allow_unknown
        if grid.traversable(point, allow_unknown):
            return point
        queue = deque([(point, 0)])
        visited = {point}
        while queue:
            current, distance = queue.popleft()
            if distance >= self.snap_radius:
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in visited or not grid.in_bounds(neighbor):
                    continue
                if grid.traversable(neighbor, allow_unknown):
                    return neighbor
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        raise ValueError(
            f"No traversable cell was found within {self.snap_radius} cells of {point}"
        )


def _downsample_points(
    points: Sequence[GridPoint], maximum_points: int
) -> List[GridPoint]:
    if len(points) <= maximum_points:
        return list(points)
    indices = torch.linspace(0, len(points) - 1, maximum_points).round().long().tolist()
    return [points[index] for index in indices]
