"""Human-readable path quality metrics separate from optimizer fitness."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config.scenario import Scenario
from .collision import (
    count_path_collisions,
    minimum_obstacle_clearance,
    path_is_within_bounds,
)
from .path_utils import ensure_path_array, path_length, smoothness_cost


@dataclass(frozen=True)
class PathMetrics:
    success: bool
    collision_free: bool
    safety_margin_satisfied: bool
    within_bounds: bool
    path_length: float
    waypoint_count: int
    minimum_clearance: float
    collision_count: int
    safety_violation_count: int
    smoothness: float
    start_error: float
    goal_error: float

    def as_dict(self) -> dict[str, bool | float | int]:
        return asdict(self)


def compute_path_metrics(path: np.ndarray, scenario: Scenario) -> PathMetrics:
    """Compute common reporting metrics for an optimizer result."""

    array = ensure_path_array(path)
    collision_count = count_path_collisions(array, scenario.obstacles, margin=0.0)
    safety_violation_count = count_path_collisions(
        array, scenario.obstacles, margin=scenario.safety_margin
    )
    minimum_clearance = minimum_obstacle_clearance(array, scenario.obstacles)
    within_bounds = path_is_within_bounds(array, scenario.width, scenario.height)
    collision_free = collision_count == 0
    safety_margin_satisfied = safety_violation_count == 0
    start_error = float(np.linalg.norm(array[0] - scenario.start_array))
    goal_error = float(np.linalg.norm(array[-1] - scenario.goal_array))

    endpoint_tolerance = 1e-9
    success = (
        within_bounds
        and collision_free
        and safety_margin_satisfied
        and start_error <= endpoint_tolerance
        and goal_error <= endpoint_tolerance
    )

    return PathMetrics(
        success=success,
        collision_free=collision_free,
        safety_margin_satisfied=safety_margin_satisfied,
        within_bounds=within_bounds,
        path_length=path_length(array),
        waypoint_count=max(0, len(array) - 2),
        minimum_clearance=minimum_clearance,
        collision_count=collision_count,
        safety_violation_count=safety_violation_count,
        smoothness=smoothness_cost(array),
        start_error=start_error,
        goal_error=goal_error,
    )
