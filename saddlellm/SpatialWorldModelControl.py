"""Closed-loop latent MPC for SaddleLLM spatial world models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .SpatialPlanner import (
    GridPathPlanner,
    GridPoint,
    OccupancyGrid,
    SpatialPlannerConfig,
)
from .SpatialWorldModel import (
    SpatialBeliefMap,
    SpatialTensorObservationEncoder,
)
from ._WorldModel import RSSMState


@dataclass
class SpatialCEMPlannerConfig:
    """Sampling, learned-risk, and hard-geometry settings for local MPC."""

    horizon: int = 12
    num_candidates: int = 128
    iterations: int = 4
    elite_fraction: float = 0.1
    discount: float = 0.99
    minimum_std: float = 0.08
    momentum: float = 0.1
    seed: int = 0
    goal_progress_weight: float = 12.0
    terminal_distance_weight: float = 2.0
    learned_collision_weight: float = 5.0
    geometry_collision_cost: float = 10000.0
    allow_unknown: bool = False
    diagonal: bool = True
    enforce_geodesic_progress: bool = True

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.num_candidates < 2:
            raise ValueError("num_candidates must be at least 2")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1]")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be in [0, 1]")
        if self.minimum_std < 0:
            raise ValueError("minimum_std cannot be negative")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        for name in (
            "goal_progress_weight",
            "terminal_distance_weight",
            "learned_collision_weight",
            "geometry_collision_cost",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass
class SpatialCEMPlan:
    actions: torch.Tensor
    score: float
    predicted_points: List[GridPoint]
    geometry_collisions: int
    predicted_rewards: torch.Tensor
    learned_collision_probability: Optional[torch.Tensor] = None
    occupancy_logits: Optional[torch.Tensor] = None

    @property
    def action(self) -> torch.Tensor:
        return self.actions[0]

    def to_dict(self, include_occupancy: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "action": self.action.detach().cpu().tolist(),
            "actions": self.actions.detach().cpu().tolist(),
            "score": float(self.score),
            "predicted_points": [list(point) for point in self.predicted_points],
            "geometry_collisions": int(self.geometry_collisions),
            "predicted_rewards": self.predicted_rewards.detach().cpu().tolist(),
            "learned_collision_probability": (
                self.learned_collision_probability.detach().cpu().tolist()
                if self.learned_collision_probability is not None
                else None
            ),
        }
        if include_occupancy and self.occupancy_logits is not None:
            probabilities = torch.softmax(self.occupancy_logits.detach().cpu(), dim=1)
            result["future_occupancy"] = {
                "classes": probabilities.argmax(dim=1).tolist(),
                "confidence": probabilities.max(dim=1).values.tolist(),
            }
        return result


@dataclass
class SpatialClosedLoopResult:
    points: List[GridPoint]
    actions: List[List[float]]
    reached_goal: bool
    replans: int
    collisions: int
    total_score: float
    last_plan: Optional[SpatialCEMPlan] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [list(point) for point in self.points],
            "actions": self.actions,
            "reached_goal": self.reached_goal,
            "replans": self.replans,
            "collisions": self.collisions,
            "total_score": self.total_score,
            "last_plan": (
                self.last_plan.to_dict(include_occupancy=True)
                if self.last_plan is not None
                else None
            ),
        }


class SpatialClosedLoopPlanner:
    """Receding-horizon CEM with learned dynamics and a hard occupancy shield."""

    def __init__(
        self,
        runtime: Any,
        encoder: Optional[SpatialTensorObservationEncoder] = None,
        config: Optional[SpatialCEMPlannerConfig] = None,
    ) -> None:
        self.runtime = runtime
        self.encoder = encoder or SpatialTensorObservationEncoder()
        self.config = config or SpatialCEMPlannerConfig()
        model_shape = tuple(int(value) for value in runtime.model.config.observation_shape)
        if model_shape != self.encoder.observation_shape:
            raise ValueError(
                "Spatial closed-loop planner observation mismatch: "
                f"model={model_shape}, encoder={self.encoder.observation_shape}"
            )
        if runtime.model.config.action_type != "continuous":
            raise ValueError("Spatial closed-loop CEM requires continuous actions")
        if runtime.model.config.action_dim < 2:
            raise ValueError("Spatial closed-loop CEM requires action_dim >= 2")

    def plan(
        self,
        state: RSSMState,
        grid: OccupancyGrid,
        current: GridPoint,
        goal: GridPoint,
    ) -> SpatialCEMPlan:
        config = self.config
        action_dim = self.runtime.model.config.action_dim
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        mean = torch.zeros(config.horizon, action_dim)
        std = torch.ones_like(mean)
        elite_count = max(1, int(config.num_candidates * config.elite_fraction))
        expert = self._expert_seed(grid, current, goal, action_dim)
        best: Optional[Tuple[float, torch.Tensor, List[GridPoint], int, Any, int]] = None

        for iteration in range(config.iterations):
            noise = torch.randn(
                config.num_candidates,
                config.horizon,
                action_dim,
                generator=generator,
            )
            candidates = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-1.0, 1.0)
            candidates[0] = expert
            candidates[1].zero_()
            evaluation = self.runtime.score_action_sequences(
                state,
                candidates,
                discount=config.discount,
                continuation_aware=True,
            )
            learned_scores = evaluation.scores.detach().cpu()
            geometry_scores = torch.empty(config.num_candidates)
            paths: List[List[GridPoint]] = []
            collision_counts: List[int] = []
            start_distance = _normalized_distance(grid, current, goal)
            for index, candidate in enumerate(candidates):
                path, collisions = self._simulate_geometry(grid, current, candidate)
                paths.append(path)
                collision_counts.append(collisions)
                final_distance = _normalized_distance(grid, path[-1], goal)
                progress = start_distance - final_distance
                geometry_scores[index] = (
                    config.goal_progress_weight * progress
                    - config.terminal_distance_weight * final_distance
                    - config.geometry_collision_cost * collisions
                )
            scores = learned_scores + geometry_scores
            collision_probability = getattr(
                evaluation.imagination, "collision_probability", None
            )
            if collision_probability is not None:
                learned_risk = collision_probability.detach().cpu().sum(dim=1)
                scores = scores - config.learned_collision_weight * learned_risk

            iteration_index = int(torch.argmax(scores))
            iteration_score = float(scores[iteration_index])
            if best is None or iteration_score > best[0]:
                best = (
                    iteration_score,
                    candidates[iteration_index].detach().clone(),
                    paths[iteration_index],
                    collision_counts[iteration_index],
                    evaluation.imagination,
                    iteration_index,
                )
            elite_indices = torch.topk(scores, elite_count).indices
            elites = candidates[elite_indices]
            elite_mean = elites.mean(dim=0)
            elite_std = elites.std(dim=0, unbiased=False).clamp_min(config.minimum_std)
            mean = config.momentum * mean + (1.0 - config.momentum) * elite_mean
            std = config.momentum * std + (1.0 - config.momentum) * elite_std

        if best is None:
            raise RuntimeError("Spatial CEM did not evaluate candidates")
        score, actions, points, collisions, imagination, candidate_index = best
        learned_collision = getattr(imagination, "collision_probability", None)
        occupancy_logits = getattr(imagination, "occupancy_logits", None)
        return SpatialCEMPlan(
            actions=actions,
            score=score,
            predicted_points=points,
            geometry_collisions=collisions,
            predicted_rewards=imagination.rewards[candidate_index].detach().cpu(),
            learned_collision_probability=(
                learned_collision[candidate_index].detach().cpu()
                if learned_collision is not None
                else None
            ),
            occupancy_logits=(
                occupancy_logits[candidate_index].detach().cpu()
                if occupancy_logits is not None
                else None
            ),
        )

    def run_closed_loop(
        self,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
        max_steps: int = 128,
    ) -> SpatialClosedLoopResult:
        """Execute one safe action per replan in the supplied grid simulator."""

        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not grid.traversable(start) or not grid.traversable(goal):
            raise ValueError("start and goal must be traversable")
        belief = self.encoder.new_belief(grid)
        current = start
        heading = 0.0
        observations: List[np.ndarray] = [
            self.encoder.encode(grid, current, goal, belief=belief, heading=heading)
        ]
        actions: List[List[float]] = []
        points = [current]
        collisions = 0
        total_score = 0.0
        last_plan: Optional[SpatialCEMPlan] = None

        for _ in range(max_steps):
            if current == goal:
                break
            if actions:
                posterior = self.runtime.filter(
                    np.stack(observations, axis=0), actions, deterministic=True
                )
                state = posterior.final_state
            else:
                state = self.runtime.encode_observation(observations[0])
            last_plan = self.plan(state, grid, current, goal)
            action = last_plan.action.detach().cpu()
            following, collided = self._apply_action(grid, current, action)
            expert = self._expert_seed(
                grid, current, goal, self.runtime.model.config.action_dim
            )[0]
            expert_following, expert_collided = self._apply_action(
                grid, current, expert
            )
            should_fallback = collided or following == current
            if self.config.enforce_geodesic_progress and not collided:
                current_steps = self._geodesic_steps(grid, current, goal)
                following_steps = self._geodesic_steps(grid, following, goal)
                should_fallback = should_fallback or following_steps >= current_steps
            if should_fallback:
                following, collided = expert_following, expert_collided
                action = expert
            actions.append(action.tolist())
            collisions += int(collided)
            total_score += last_plan.score
            if following != current:
                heading = math.atan2(
                    following[1] - current[1], following[0] - current[0]
                )
            current = following
            points.append(current)
            observations.append(
                self.encoder.encode(
                    grid, current, goal, belief=belief, heading=heading
                )
            )

        return SpatialClosedLoopResult(
            points=points,
            actions=actions,
            reached_goal=current == goal,
            replans=len(actions),
            collisions=collisions,
            total_score=float(total_score),
            last_plan=last_plan,
        )

    def _expert_seed(
        self,
        grid: OccupancyGrid,
        current: GridPoint,
        goal: GridPoint,
        action_dim: int,
    ) -> torch.Tensor:
        result = torch.zeros(self.config.horizon, action_dim)
        planner = GridPathPlanner(
            SpatialPlannerConfig(
                route_count=1,
                diagonal=self.config.diagonal,
                allow_unknown=self.config.allow_unknown,
            )
        )
        try:
            path = planner.shortest_path(grid, current, goal)
        except ValueError:
            path = [current]
        for index, (first, second) in enumerate(zip(path, path[1:])):
            if index >= self.config.horizon:
                break
            result[index, 0] = float(second[0] - first[0])
            result[index, 1] = float(second[1] - first[1])
        return result

    def _simulate_geometry(
        self,
        grid: OccupancyGrid,
        current: GridPoint,
        actions: torch.Tensor,
    ) -> Tuple[List[GridPoint], int]:
        points = [current]
        collisions = 0
        for action in actions:
            current, collided = self._apply_action(grid, current, action)
            points.append(current)
            collisions += int(collided)
        return points, collisions

    def _apply_action(
        self,
        grid: OccupancyGrid,
        current: GridPoint,
        action: Sequence[float],
    ) -> Tuple[GridPoint, bool]:
        dx, dy = _quantize_action(action, diagonal=self.config.diagonal)
        if dx == 0 and dy == 0:
            return current, False
        following = (current[0] + dx, current[1] + dy)
        if not grid.traversable(following, allow_unknown=self.config.allow_unknown):
            return current, True
        if dx and dy:
            side_a = (current[0] + dx, current[1])
            side_b = (current[0], current[1] + dy)
            if not grid.traversable(
                side_a, allow_unknown=self.config.allow_unknown
            ) or not grid.traversable(side_b, allow_unknown=self.config.allow_unknown):
                return current, True
        return following, False

    def _geodesic_steps(
        self, grid: OccupancyGrid, current: GridPoint, goal: GridPoint
    ) -> float:
        planner = GridPathPlanner(
            SpatialPlannerConfig(
                route_count=1,
                diagonal=self.config.diagonal,
                allow_unknown=self.config.allow_unknown,
            )
        )
        try:
            return float(len(planner.shortest_path(grid, current, goal)) - 1)
        except ValueError:
            return float("inf")


def _quantize_action(action: Sequence[float], diagonal: bool) -> Tuple[int, int]:
    values = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    if values.numel() < 2:
        raise ValueError("Spatial action needs at least two values")
    x, y = float(values[0]), float(values[1])
    if math.hypot(x, y) < 0.25:
        return 0, 0
    if not diagonal:
        return ((1 if x > 0 else -1), 0) if abs(x) >= abs(y) else (
            0,
            1 if y > 0 else -1,
        )
    dx = 0 if abs(x) < 0.25 else (1 if x > 0 else -1)
    dy = 0 if abs(y) < 0.25 else (1 if y > 0 else -1)
    return dx, dy


def _normalized_distance(
    grid: OccupancyGrid, point: GridPoint, goal: GridPoint
) -> float:
    dx = (goal[0] - point[0]) / max(grid.width - 1, 1)
    dy = (goal[1] - point[1]) / max(grid.height - 1, 1)
    return math.hypot(dx, dy)


__all__ = [
    "SpatialCEMPlannerConfig",
    "SpatialCEMPlan",
    "SpatialClosedLoopResult",
    "SpatialClosedLoopPlanner",
]
