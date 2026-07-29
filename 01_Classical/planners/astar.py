"""Grid-based A* path planner."""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Set, Tuple

from config.scenarios import Scenario
from utils.collision import create_occupancy_grid, grid_to_point, point_to_grid
from utils.metrics import (
    PlanningResult,
    calculate_minimum_clearance,
    calculate_path_length,
)

GridPoint = Tuple[int, int]


class AStarPlanner:
    """Eight-connected A* planner with Euclidean movement costs."""

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
        self.grid = create_occupancy_grid(scenario)

    @staticmethod
    def _heuristic(node: GridPoint, goal: GridPoint) -> float:
        return math.hypot(goal[0] - node[0], goal[1] - node[1])

    def _in_grid(self, node: GridPoint) -> bool:
        return (
            0 <= node[0] < self.grid.shape[0]
            and 0 <= node[1] < self.grid.shape[1]
        )

    def _diagonal_move_is_clear(self, current: GridPoint, dx: int, dy: int) -> bool:
        """Prevent a diagonal step from cutting through blocked corners."""

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

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[GridPoint, GridPoint],
        current: GridPoint,
    ) -> List[GridPoint]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plan(self) -> PlanningResult:
        """Plan a path and return it in the common result format."""

        start_time = time.perf_counter()
        resolution = self.scenario.grid_resolution
        start = point_to_grid(self.scenario.start, resolution)
        goal = point_to_grid(self.scenario.goal, resolution)

        if not self._in_grid(start) or not self._in_grid(goal):
            raise ValueError("Start or goal lies outside the occupancy grid.")
        if self.grid[start]:
            raise ValueError("Start lies inside an obstacle or safety margin.")
        if self.grid[goal]:
            raise ValueError("Goal lies inside an obstacle or safety margin.")

        open_heap: List[Tuple[float, float, GridPoint]] = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))

        came_from: Dict[GridPoint, GridPoint] = {}
        g_score: Dict[GridPoint, float] = {start: 0.0}
        closed: Set[GridPoint] = set()
        expanded_nodes = 0
        final_grid_path: Optional[List[GridPoint]] = None

        while open_heap:
            _, queued_g, current = heapq.heappop(open_heap)

            if current in closed:
                continue
            if queued_g > g_score.get(current, float("inf")):
                continue

            if current == goal:
                final_grid_path = self._reconstruct_path(came_from, current)
                break

            closed.add(current)
            expanded_nodes += 1

            for dx, dy, movement_cost in self._MOVEMENTS:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._in_grid(neighbor) or self.grid[neighbor]:
                    continue
                if not self._diagonal_move_is_clear(current, dx, dy):
                    continue

                tentative_g = g_score[current] + movement_cost
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self._heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        path = (
            [grid_to_point(node, resolution) for node in final_grid_path]
            if final_grid_path
            else []
        )

        return PlanningResult(
            algorithm="A*",
            success=bool(path),
            path=path,
            planning_time_ms=elapsed_ms,
            path_length=calculate_path_length(path),
            waypoint_count=len(path),
            minimum_clearance=calculate_minimum_clearance(path, self.scenario),
            details={
                "expanded_nodes": expanded_nodes,
                "grid_resolution": resolution,
                "connectivity": 8,
                "heuristic": "euclidean",
                "diagonal_corner_cutting": False,
            },
        )
