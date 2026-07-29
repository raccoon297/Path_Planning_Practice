"""Shared path objective used by GA, GWO, PSO, and final ACO evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config.scenario import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    ObjectiveWeights,
    Scenario,
)
from .collision import boundary_violation_amount, segment_obstacle_clearances
from .path_utils import path_length, smoothness_cost, vector_to_path


@dataclass(frozen=True)
class ObjectiveResult:
    """Weighted objective breakdown for one candidate path."""

    total: float
    length: float
    collision: float
    clearance: float
    smoothness: float
    boundary: float
    collision_count: int
    safety_violation_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_path(
    path: np.ndarray,
    scenario: Scenario,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
) -> ObjectiveResult:
    """Evaluate one complete path under the shared objective.

    ``collision`` penalizes intersections with physical obstacles. ``clearance``
    separately penalizes entering the requested safety-margin region.
    """

    length_value = path_length(path)
    smoothness_value = smoothness_cost(path)
    boundary_value = boundary_violation_amount(path, scenario.width, scenario.height)

    clearances = segment_obstacle_clearances(path, scenario.obstacles)
    if clearances.size == 0:
        collision_count = 0
        safety_violation_count = 0
        clearance_shortfall = 0.0
    else:
        collision_count = int(np.count_nonzero(clearances <= 0.0))
        shortfalls = np.clip(scenario.safety_margin - clearances, 0.0, None)
        safety_violation_count = int(np.count_nonzero(shortfalls > 0.0))
        # A fixed violation term makes the safety margin an actual preference,
        # while the squared shortfall still distinguishes shallow and deep
        # intrusions. Without the fixed term, an optimizer can settle a tiny
        # distance inside the margin because its penalty approaches zero.
        clearance_shortfall = float(
            safety_violation_count + np.square(shortfalls).sum()
        )

    weighted_length = weights.length * length_value
    weighted_collision = weights.collision * collision_count
    weighted_clearance = weights.clearance * clearance_shortfall
    weighted_smoothness = weights.smoothness * smoothness_value
    weighted_boundary = weights.boundary * boundary_value

    total = (
        weighted_length
        + weighted_collision
        + weighted_clearance
        + weighted_smoothness
        + weighted_boundary
    )

    return ObjectiveResult(
        total=float(total),
        length=float(weighted_length),
        collision=float(weighted_collision),
        clearance=float(weighted_clearance),
        smoothness=float(weighted_smoothness),
        boundary=float(weighted_boundary),
        collision_count=collision_count,
        safety_violation_count=safety_violation_count,
    )


def evaluate_waypoint_vector(
    vector: np.ndarray,
    scenario: Scenario,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
) -> ObjectiveResult:
    """Decode a GA/GWO/PSO vector and evaluate the resulting path."""

    path = vector_to_path(
        vector,
        start=scenario.start_array,
        goal=scenario.goal_array,
        num_waypoints=scenario.num_waypoints,
    )
    return evaluate_path(path, scenario, weights)
