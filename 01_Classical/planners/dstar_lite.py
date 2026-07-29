"""Incremental grid-based D* Lite path planner.

The implementation follows the core D* Lite structure: ``g`` and ``rhs``
values, lexicographic two-component keys, ``km`` updates as the agent moves,
and local vertex repairs after occupancy-grid changes.
"""

from __future__ import annotations

import heapq
import math
import time
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from config.scenarios import CircleObstacle, Scenario
from utils.collision import create_occupancy_grid, grid_to_point, point_to_grid
from utils.metrics import (
    PlanningResult,
    calculate_minimum_clearance,
    calculate_path_length,
)

GridPoint = Tuple[int, int]
Key = Tuple[float, float]


class DStarLitePlanner:
    """Eight-connected D* Lite planner for static and changing grid maps."""

    _MOVEMENTS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.resolution = scenario.grid_resolution
        self.grid = create_occupancy_grid(scenario)
        self.start = point_to_grid(scenario.start, self.resolution)
        self.goal = point_to_grid(scenario.goal, self.resolution)
        self.last_start = self.start
        self.km = 0.0

        self.g: DefaultDict[GridPoint, float] = defaultdict(lambda: float("inf"))
        self.rhs: DefaultDict[GridPoint, float] = defaultdict(lambda: float("inf"))
        self.rhs[self.goal] = 0.0

        self.open_heap: List[Tuple[float, float, int, GridPoint]] = []
        self.open_keys: Dict[GridPoint, Key] = {}
        self._counter = 0
        self._push(self.goal, self._calculate_key(self.goal))

        self.expanded_nodes = 0
        self.updated_vertices = 0
        self.replanning_count = 0

        self._validate_terminal(self.start, "Start")
        self._validate_terminal(self.goal, "Goal")

    @staticmethod
    def _heuristic(a: GridPoint, b: GridPoint) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _validate_terminal(self, node: GridPoint, label: str) -> None:
        if not self._in_grid(node):
            raise ValueError(f"{label} lies outside the occupancy grid.")
        if self.grid[node]:
            raise ValueError(f"{label} lies inside an obstacle or safety margin.")

    def _in_grid(self, node: GridPoint) -> bool:
        return (
            0 <= node[0] < self.grid.shape[0]
            and 0 <= node[1] < self.grid.shape[1]
        )

    def _calculate_key(self, node: GridPoint) -> Key:
        best = min(self.g[node], self.rhs[node])
        return (
            best + self._heuristic(self.start, node) + self.km,
            best,
        )

    @staticmethod
    def _key_less(left: Key, right: Key, tolerance: float = 1e-12) -> bool:
        if left[0] < right[0] - tolerance:
            return True
        if abs(left[0] - right[0]) <= tolerance:
            return left[1] < right[1] - tolerance
        return False

    def _push(self, node: GridPoint, key: Key) -> None:
        self._counter += 1
        self.open_keys[node] = key
        heapq.heappush(self.open_heap, (key[0], key[1], self._counter, node))

    def _remove(self, node: GridPoint) -> None:
        self.open_keys.pop(node, None)

    def _clean_top(self) -> None:
        while self.open_heap:
            k1, k2, _, node = self.open_heap[0]
            current = self.open_keys.get(node)
            if current is None or current != (k1, k2):
                heapq.heappop(self.open_heap)
                continue
            break

    def _top_key(self) -> Key:
        self._clean_top()
        if not self.open_heap:
            return float("inf"), float("inf")
        k1, k2, _, _ = self.open_heap[0]
        return k1, k2

    def _pop(self) -> Tuple[Key, GridPoint]:
        self._clean_top()
        if not self.open_heap:
            raise IndexError("D* Lite priority queue is empty.")
        k1, k2, _, node = heapq.heappop(self.open_heap)
        self.open_keys.pop(node, None)
        return (k1, k2), node

    def _diagonal_move_is_clear(self, current: GridPoint, dx: int, dy: int) -> bool:
        if dx == 0 or dy == 0:
            return True
        adjacent_x = (current[0] + dx, current[1])
        adjacent_y = (current[0], current[1] + dy)
        return (
            self._in_grid(adjacent_x)
            and self._in_grid(adjacent_y)
            and not self.grid[adjacent_x]
            and not self.grid[adjacent_y]
        )

    def _successors(self, node: GridPoint) -> Iterable[Tuple[GridPoint, float]]:
        if not self._in_grid(node) or self.grid[node]:
            return []

        neighbors: List[Tuple[GridPoint, float]] = []
        for dx, dy, movement_cost in self._MOVEMENTS:
            neighbor = (node[0] + dx, node[1] + dy)
            if not self._in_grid(neighbor) or self.grid[neighbor]:
                continue
            if not self._diagonal_move_is_clear(node, dx, dy):
                continue
            neighbors.append((neighbor, movement_cost))
        return neighbors

    def _predecessors(self, node: GridPoint) -> Iterable[Tuple[GridPoint, float]]:
        # The grid graph is undirected, so predecessors and successors coincide.
        return self._successors(node)

    def _update_vertex(self, node: GridPoint) -> None:
        if node != self.goal:
            candidates = [cost + self.g[succ] for succ, cost in self._successors(node)]
            self.rhs[node] = min(candidates, default=float("inf"))

        self._remove(node)
        if not math.isclose(self.g[node], self.rhs[node], rel_tol=0.0, abs_tol=1e-12):
            self._push(node, self._calculate_key(node))
        self.updated_vertices += 1

    def compute_shortest_path(self) -> None:
        """Repair shortest-path values until the current start is consistent."""

        while self._key_less(self._top_key(), self._calculate_key(self.start)) or not math.isclose(
            self.rhs[self.start], self.g[self.start], rel_tol=0.0, abs_tol=1e-12
        ):
            if not self.open_heap:
                break

            old_key, node = self._pop()
            new_key = self._calculate_key(node)

            if self._key_less(old_key, new_key):
                self._push(node, new_key)
            elif self.g[node] > self.rhs[node]:
                self.g[node] = self.rhs[node]
                self.expanded_nodes += 1
                for predecessor, _ in self._predecessors(node):
                    self._update_vertex(predecessor)
            else:
                self.g[node] = float("inf")
                self._update_vertex(node)
                for predecessor, _ in self._predecessors(node):
                    self._update_vertex(predecessor)

    def _extract_grid_path(self, max_steps: Optional[int] = None) -> List[GridPoint]:
        if math.isinf(self.g[self.start]) and math.isinf(self.rhs[self.start]):
            return []

        step_limit = max_steps or self.grid.size * 2
        current = self.start
        path = [current]
        visited: Set[GridPoint] = {current}

        while current != self.goal and len(path) <= step_limit:
            candidates = [
                (cost + self.g[neighbor], self._heuristic(neighbor, self.goal), neighbor)
                for neighbor, cost in self._successors(current)
                if not math.isinf(self.g[neighbor])
            ]
            if not candidates:
                return []

            _, _, next_node = min(candidates)
            if next_node in visited and next_node != self.goal:
                return []
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path if current == self.goal else []

    def current_path(self) -> List[Tuple[float, float]]:
        """Return the currently valid start-to-goal path in world coordinates."""

        return [
            grid_to_point(node, self.resolution)
            for node in self._extract_grid_path()
        ]

    def move_start(self, new_start_point: Tuple[float, float]) -> None:
        """Move the agent start and update ``km`` as specified by D* Lite."""

        new_start = point_to_grid(new_start_point, self.resolution)
        self._validate_terminal(new_start, "Updated start")
        self.km += self._heuristic(self.last_start, new_start)
        self.start = new_start
        self.last_start = new_start

    def reveal_obstacles(self, obstacles: Iterable[CircleObstacle]) -> int:
        """Add newly observed obstacles and incrementally repair affected vertices.

        Returns the number of occupancy-grid cells whose state changed.
        """

        old_grid = self.grid.copy()
        combined = self.scenario.obstacles + tuple(obstacles)
        changed_scenario = Scenario(
            name=self.scenario.name,
            width=self.scenario.width,
            height=self.scenario.height,
            start=grid_to_point(self.start, self.resolution),
            goal=self.scenario.goal,
            obstacles=combined,
            safety_margin=self.scenario.safety_margin,
            grid_resolution=self.scenario.grid_resolution,
            random_seed=self.scenario.random_seed,
        )
        self.scenario = changed_scenario
        self.grid = create_occupancy_grid(changed_scenario)

        changed_indices = np.argwhere(old_grid != self.grid)
        affected: Set[GridPoint] = set()
        for x_index, y_index in changed_indices:
            changed = (int(x_index), int(y_index))
            affected.add(changed)
            for dx, dy, _ in self._MOVEMENTS:
                neighbor = (changed[0] + dx, changed[1] + dy)
                if self._in_grid(neighbor):
                    affected.add(neighbor)

        for node in affected:
            self._update_vertex(node)

        if len(changed_indices) > 0:
            self.replanning_count += 1
        return len(changed_indices)

    def replan(self) -> List[Tuple[float, float]]:
        """Repair the search after a start or map change and return a new path."""

        self.compute_shortest_path()
        return self.current_path()

    def plan(self) -> PlanningResult:
        """Compute an initial path in the common result format."""

        start_time = time.perf_counter()
        self.compute_shortest_path()
        path = self.current_path()
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PlanningResult(
            algorithm="D* Lite",
            success=bool(path),
            path=path,
            planning_time_ms=elapsed_ms,
            path_length=calculate_path_length(path),
            waypoint_count=len(path),
            minimum_clearance=calculate_minimum_clearance(path, self.scenario),
            details={
                "expanded_nodes": self.expanded_nodes,
                "updated_vertices": self.updated_vertices,
                "replanning_count": self.replanning_count,
                "grid_resolution": self.resolution,
                "connectivity": 8,
                "heuristic": "euclidean",
                "incremental_search": True,
                "uses_g_rhs": True,
                "uses_km": True,
            },
        )
