"""Native occupancy-grid representation and Top-K route planning."""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass
from itertools import count
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


GridPoint = Tuple[int, int]
GridEdge = Tuple[GridPoint, GridPoint]


@dataclass
class OccupancyGrid:
    """Two-dimensional navigation grid using FREE/BLOCKED/UNKNOWN cells."""

    FREE = 0
    BLOCKED = 1
    UNKNOWN = 2

    cells: np.ndarray
    resolution: float = 1.0
    source_size: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        self.cells = np.asarray(self.cells, dtype=np.uint8)
        if self.cells.ndim != 2 or not self.cells.size:
            raise ValueError("OccupancyGrid.cells must be a non-empty 2D array")
        if not np.isin(self.cells, [self.FREE, self.BLOCKED, self.UNKNOWN]).all():
            raise ValueError("OccupancyGrid cells must be FREE, BLOCKED, or UNKNOWN")
        if self.resolution <= 0:
            raise ValueError("OccupancyGrid.resolution must be positive")
        if self.source_size is not None:
            self.source_size = tuple(int(value) for value in self.source_size)

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    def in_bounds(self, point: GridPoint) -> bool:
        x, y = point
        return 0 <= x < self.width and 0 <= y < self.height

    def cell(self, point: GridPoint) -> int:
        if not self.in_bounds(point):
            raise IndexError(f"Grid point is outside the map: {point}")
        return int(self.cells[point[1], point[0]])

    def traversable(self, point: GridPoint, allow_unknown: bool = False) -> bool:
        if not self.in_bounds(point):
            return False
        value = self.cell(point)
        return value == self.FREE or (allow_unknown and value == self.UNKNOWN)

    def scale_from_source(self, point: Sequence[float]) -> GridPoint:
        if len(point) != 2:
            raise ValueError("A point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if self.source_size is not None:
            source_width, source_height = self.source_size
            x = x * self.width / max(source_width, 1)
            y = y * self.height / max(source_height, 1)
        return (
            min(max(int(round(x)), 0), self.width - 1),
            min(max(int(round(y)), 0), self.height - 1),
        )

    def scale_to_source(self, point: GridPoint) -> Tuple[float, float]:
        x, y = point
        if self.source_size is None:
            return float(x), float(y)
        source_width, source_height = self.source_size
        return (
            x * source_width / max(self.width, 1),
            y * source_height / max(self.height, 1),
        )

    def to_dict(self, include_cells: bool = False) -> Dict[str, Any]:
        values, counts = np.unique(self.cells, return_counts=True)
        cell_counts = {str(int(value)): int(amount) for value, amount in zip(values, counts)}
        result: Dict[str, Any] = {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "source_size": list(self.source_size) if self.source_size else None,
            "cell_counts": {
                "free": cell_counts.get(str(self.FREE), 0),
                "blocked": cell_counts.get(str(self.BLOCKED), 0),
                "unknown": cell_counts.get(str(self.UNKNOWN), 0),
            },
        }
        if include_cells:
            result["cells"] = self.cells.tolist()
        return result


@dataclass
class SpatialPlannerConfig:
    diagonal: bool = True
    allow_unknown: bool = False
    unknown_cost: float = 4.0
    clearance_weight: float = 0.0
    route_count: int = 3
    diversity_weight: float = 2.0
    diversity_radius: int = 2
    max_detour_ratio: float = 2.5

    def __post_init__(self) -> None:
        if self.unknown_cost < 1.0:
            raise ValueError("unknown_cost must be at least 1")
        if self.clearance_weight < 0:
            raise ValueError("clearance_weight cannot be negative")
        if self.route_count <= 0:
            raise ValueError("route_count must be positive")
        if self.diversity_weight < 0:
            raise ValueError("diversity_weight cannot be negative")
        if self.diversity_radius < 0:
            raise ValueError("diversity_radius cannot be negative")
        if self.max_detour_ratio < 1.0:
            raise ValueError("max_detour_ratio must be at least 1")


@dataclass
class RouteCandidate:
    points: List[GridPoint]
    length: float
    cost: float
    turns: int
    minimum_clearance: float
    average_clearance: float
    risk: float
    label: str = "candidate"
    predicted_return: Optional[float] = None
    combined_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["points"] = [[int(x), int(y)] for x, y in self.points]
        return result


class GridPathPlanner:
    """A* shortest path plus spatially diversified alternative routes."""

    def __init__(self, config: Optional[SpatialPlannerConfig] = None) -> None:
        self.config = config or SpatialPlannerConfig()

    def plan(
        self,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
        route_count: Optional[int] = None,
    ) -> List[RouteCandidate]:
        start = _point(start)
        goal = _point(goal)
        if not grid.traversable(start, self.config.allow_unknown):
            raise ValueError(f"Start point is not traversable: {start}")
        if not grid.traversable(goal, self.config.allow_unknown):
            raise ValueError(f"Goal point is not traversable: {goal}")
        requested = int(route_count or self.config.route_count)
        if requested <= 0:
            raise ValueError("route_count must be positive")
        clearance = self.clearance_map(grid)
        paths = (
            self._diverse_paths(grid, start, goal, requested, clearance)
            if self.config.diversity_weight > 0 and requested > 1
            else self._yen_paths(grid, start, goal, requested, clearance)
        )
        labels = ["shortest", "alternative", "alternative"]
        routes = [
            self._route_metrics(
                grid,
                path,
                clearance,
                label=labels[index] if index < len(labels) else f"alternative-{index}",
            )
            for index, path in enumerate(paths)
        ]
        return routes

    def _diverse_paths(
        self,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
        route_count: int,
        clearance: np.ndarray,
    ) -> List[List[GridPoint]]:
        """Find practical alternatives by discouraging reused route corridors.

        Pure K-shortest paths often differ by just one grid cell.  For walking
        directions that is technically correct but not useful.  This search
        keeps the shortest route, then applies a soft influence field around
        accepted paths so later A* searches prefer another corridor while
        respecting a bounded detour ratio.
        """

        first = self._astar(grid, start, goal, clearance)
        if not first:
            raise ValueError(f"No traversable route from {start} to {goal}")
        accepted: List[List[GridPoint]] = [first]
        accepted_keys = {tuple(first)}
        first_cost = max(self._path_cost(grid, first, clearance), 1e-6)
        influence = np.zeros((grid.height, grid.width), dtype=np.float32)
        self._add_route_influence(influence, first)

        while len(accepted) < route_count:
            proposals: List[Tuple[float, float, List[GridPoint]]] = []
            for multiplier in (0.5, 1.0, 2.0, 4.0, 8.0):
                path = self._astar(
                    grid,
                    start,
                    goal,
                    clearance,
                    node_penalties=influence
                    * (self.config.diversity_weight * multiplier),
                )
                key = tuple(path)
                if not path or key in accepted_keys:
                    continue
                cost = self._path_cost(grid, path, clearance)
                if cost > first_cost * self.config.max_detour_ratio:
                    continue
                overlap = max(self._path_overlap(path, value) for value in accepted)
                proposals.append((overlap, cost, path))
            if not proposals:
                break
            _, _, selected = min(proposals, key=lambda item: (item[0], item[1]))
            accepted.append(selected)
            accepted_keys.add(tuple(selected))
            self._add_route_influence(influence, selected)

        if len(accepted) < route_count:
            for path in self._yen_paths(grid, start, goal, route_count * 3, clearance):
                key = tuple(path)
                if key in accepted_keys:
                    continue
                accepted.append(path)
                accepted_keys.add(key)
                if len(accepted) >= route_count:
                    break
        return accepted[:route_count]

    def _add_route_influence(
        self,
        influence: np.ndarray,
        path: Sequence[GridPoint],
    ) -> None:
        radius = self.config.diversity_radius
        for x, y in path[1:-1]:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    distance = abs(dx) + abs(dy)
                    if distance > radius:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= ny < influence.shape[0] and 0 <= nx < influence.shape[1]:
                        influence[ny, nx] += 1.0 / (1.0 + distance)

    @staticmethod
    def _path_overlap(first: Sequence[GridPoint], second: Sequence[GridPoint]) -> float:
        first_cells = set(first[1:-1])
        second_cells = set(second[1:-1])
        denominator = max(1, min(len(first_cells), len(second_cells)))
        return len(first_cells & second_cells) / denominator

    def shortest_path(
        self,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
    ) -> List[GridPoint]:
        return self._astar(grid, _point(start), _point(goal), self.clearance_map(grid))

    def clearance_map(self, grid: OccupancyGrid) -> np.ndarray:
        distances = np.full((grid.height, grid.width), np.inf, dtype=np.float32)
        queue: deque = deque()
        obstacle_mask = grid.cells == grid.BLOCKED
        if not self.config.allow_unknown:
            obstacle_mask |= grid.cells == grid.UNKNOWN
        for y, x in np.argwhere(obstacle_mask):
            distances[y, x] = 0.0
            queue.append((int(x), int(y)))
        if not queue:
            distances.fill(float(max(grid.width, grid.height)))
            return distances
        while queue:
            x, y = queue.popleft()
            next_distance = distances[y, x] + 1.0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    if next_distance < distances[ny, nx]:
                        distances[ny, nx] = next_distance
                        queue.append((nx, ny))
        return distances

    def _yen_paths(
        self,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
        route_count: int,
        clearance: np.ndarray,
    ) -> List[List[GridPoint]]:
        first = self._astar(grid, start, goal, clearance)
        if not first:
            raise ValueError(f"No traversable route from {start} to {goal}")
        accepted: List[List[GridPoint]] = [first]
        accepted_keys = {tuple(first)}
        candidates: List[Tuple[float, int, Tuple[GridPoint, ...]]] = []
        candidate_keys: Set[Tuple[GridPoint, ...]] = set()
        sequence = count()

        while len(accepted) < route_count:
            previous = accepted[-1]
            for index in range(len(previous) - 1):
                spur_node = previous[index]
                root_path = previous[: index + 1]
                banned_edges: Set[GridEdge] = set()
                for path in accepted:
                    if len(path) > index and path[: index + 1] == root_path:
                        banned_edges.add((path[index], path[index + 1]))
                banned_nodes = set(root_path[:-1])
                spur_path = self._astar(
                    grid,
                    spur_node,
                    goal,
                    clearance,
                    banned_nodes=banned_nodes,
                    banned_edges=banned_edges,
                )
                if not spur_path:
                    continue
                total_path = root_path[:-1] + spur_path
                key = tuple(total_path)
                if key in accepted_keys or key in candidate_keys:
                    continue
                candidate_keys.add(key)
                heapq.heappush(
                    candidates,
                    (self._path_cost(grid, total_path, clearance), next(sequence), key),
                )
            if not candidates:
                break
            _, _, next_path = heapq.heappop(candidates)
            candidate_keys.discard(next_path)
            accepted.append(list(next_path))
            accepted_keys.add(next_path)
        return accepted

    def _astar(
        self,
        grid: OccupancyGrid,
        start: GridPoint,
        goal: GridPoint,
        clearance: np.ndarray,
        banned_nodes: Optional[Set[GridPoint]] = None,
        banned_edges: Optional[Set[GridEdge]] = None,
        node_penalties: Optional[np.ndarray] = None,
    ) -> List[GridPoint]:
        banned_nodes = banned_nodes or set()
        banned_edges = banned_edges or set()
        if start in banned_nodes or goal in banned_nodes:
            return []
        frontier: List[Tuple[float, int, GridPoint]] = []
        sequence = count()
        heapq.heappush(frontier, (self._heuristic(start, goal), next(sequence), start))
        came_from: Dict[GridPoint, GridPoint] = {}
        g_score: Dict[GridPoint, float] = {start: 0.0}

        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                return self._reconstruct(came_from, current)
            current_score = g_score[current]
            for neighbor, move_cost in self._neighbors(grid, current):
                if neighbor in banned_nodes or (current, neighbor) in banned_edges:
                    continue
                cell_value = grid.cell(neighbor)
                unknown_multiplier = (
                    self.config.unknown_cost
                    if cell_value == grid.UNKNOWN
                    else 1.0
                )
                clearance_penalty = self.config.clearance_weight / (
                    1.0 + float(clearance[neighbor[1], neighbor[0]])
                )
                tentative = current_score + move_cost * unknown_multiplier + clearance_penalty
                if node_penalties is not None and neighbor not in {start, goal}:
                    tentative += float(node_penalties[neighbor[1], neighbor[0]])
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + self._heuristic(neighbor, goal)
                heapq.heappush(frontier, (priority, next(sequence), neighbor))
        return []

    def _neighbors(
        self, grid: OccupancyGrid, point: GridPoint
    ) -> Iterable[Tuple[GridPoint, float]]:
        x, y = point
        movements = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
        ]
        if self.config.diagonal:
            root_two = math.sqrt(2.0)
            movements.extend(
                [
                    (1, 1, root_two),
                    (1, -1, root_two),
                    (-1, 1, root_two),
                    (-1, -1, root_two),
                ]
            )
        for dx, dy, cost_value in movements:
            neighbor = (x + dx, y + dy)
            if not grid.traversable(neighbor, self.config.allow_unknown):
                continue
            if dx and dy:
                side_a = (x + dx, y)
                side_b = (x, y + dy)
                if not grid.traversable(side_a, self.config.allow_unknown):
                    continue
                if not grid.traversable(side_b, self.config.allow_unknown):
                    continue
            yield neighbor, cost_value

    def _heuristic(self, point: GridPoint, goal: GridPoint) -> float:
        dx = abs(point[0] - goal[0])
        dy = abs(point[1] - goal[1])
        if not self.config.diagonal:
            return float(dx + dy)
        return float(max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy))

    def _path_cost(
        self,
        grid: OccupancyGrid,
        path: Sequence[GridPoint],
        clearance: np.ndarray,
    ) -> float:
        total = 0.0
        for current, following in zip(path, path[1:]):
            diagonal = current[0] != following[0] and current[1] != following[1]
            movement = math.sqrt(2.0) if diagonal else 1.0
            multiplier = (
                self.config.unknown_cost
                if grid.cell(following) == grid.UNKNOWN
                else 1.0
            )
            total += movement * multiplier
            total += self.config.clearance_weight / (
                1.0 + float(clearance[following[1], following[0]])
            )
        return total

    def _route_metrics(
        self,
        grid: OccupancyGrid,
        path: List[GridPoint],
        clearance: np.ndarray,
        label: str,
    ) -> RouteCandidate:
        path_clearance = np.asarray(
            [clearance[y, x] for x, y in path], dtype=np.float32
        )
        directions = [
            (int(math.copysign(1, b[0] - a[0])) if b[0] != a[0] else 0,
             int(math.copysign(1, b[1] - a[1])) if b[1] != a[1] else 0)
            for a, b in zip(path, path[1:])
        ]
        turns = sum(
            first != second for first, second in zip(directions, directions[1:])
        )
        risk = float(np.mean(1.0 / (1.0 + path_clearance))) if path else 1.0
        length = sum(
            math.sqrt(2.0) if a[0] != b[0] and a[1] != b[1] else 1.0
            for a, b in zip(path, path[1:])
        ) * grid.resolution
        return RouteCandidate(
            points=path,
            length=float(length),
            cost=float(self._path_cost(grid, path, clearance)),
            turns=int(turns),
            minimum_clearance=float(path_clearance.min() * grid.resolution),
            average_clearance=float(path_clearance.mean() * grid.resolution),
            risk=risk,
            label=label,
        )

    @staticmethod
    def _reconstruct(
        came_from: Dict[GridPoint, GridPoint], current: GridPoint
    ) -> List[GridPoint]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def _point(value: Sequence[int]) -> GridPoint:
    if len(value) != 2:
        raise ValueError("Grid point must contain x and y")
    return int(value[0]), int(value[1])
