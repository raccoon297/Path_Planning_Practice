"""Shared result type and path-quality metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from config.scenarios import Point, Scenario


@dataclass
class PlanningResult:
    """Common planner output used by visualization and comparison code."""

    algorithm: str
    success: bool
    path: List[Point]
    planning_time_ms: float
    path_length: float
    waypoint_count: int
    minimum_clearance: float
    details: Dict[str, Any]
    visualization_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""

        data = asdict(self)
        data.pop("visualization_data", None)
        data["path"] = [[float(x), float(y)] for x, y in self.path]
        return data


def calculate_path_length(path: List[Point]) -> float:
    """Calculate total Euclidean length of a polyline path."""

    return sum(
        math.hypot(x2 - x1, y2 - y1)
        for (x1, y1), (x2, y2) in zip(path, path[1:])
    )


def calculate_minimum_clearance(path: List[Point], scenario: Scenario) -> float:
    """Calculate the minimum boundary-to-obstacle clearance along waypoints.

    A positive value indicates free space. A negative value indicates that a
    waypoint lies inside an obstacle or its configured safety margin.
    """

    if not path:
        return float("nan")

    minimum = float("inf")
    for x, y in path:
        boundary_clearance = min(x, y, scenario.width - x, scenario.height - y)
        minimum = min(minimum, boundary_clearance)

        for obstacle in scenario.obstacles:
            center_distance = math.hypot(x - obstacle.x, y - obstacle.y)
            obstacle_clearance = (
                center_distance - obstacle.radius - scenario.safety_margin
            )
            minimum = min(minimum, obstacle_clearance)

    return minimum
