"""Common path-planning scenario definitions.

The metaheuristic optimizers share this module so that every algorithm solves
exactly the same geometric problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class CircleObstacle:
    """Circular obstacle in continuous 2-D space."""

    center: Point
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Obstacle radius must be positive.")

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float)


@dataclass(frozen=True)
class ObjectiveWeights:
    """Provisional weights for the shared path objective.

    These values are intentionally centralized. They will be tuned after the
    first PSO baseline reveals the numerical scale of each objective term.
    """

    length: float = 1.0
    collision: float = 10_000.0
    clearance: float = 100.0
    smoothness: float = 5.0
    boundary: float = 10_000.0


@dataclass(frozen=True)
class Scenario:
    """Continuous 2-D path-planning problem shared by all algorithms."""

    width: float
    height: float
    start: Point
    goal: Point
    obstacles: Tuple[CircleObstacle, ...]
    safety_margin: float = 5.0
    num_waypoints: int = 5

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Map width and height must be positive.")
        if self.safety_margin < 0:
            raise ValueError("Safety margin cannot be negative.")
        if self.num_waypoints < 0:
            raise ValueError("Number of waypoints cannot be negative.")

        for name, point in (("start", self.start), ("goal", self.goal)):
            x, y = point
            if not (0.0 <= x <= self.width and 0.0 <= y <= self.height):
                raise ValueError(f"{name} point lies outside the map.")

        for obstacle in self.obstacles:
            x, y = obstacle.center
            if not (0.0 <= x <= self.width and 0.0 <= y <= self.height):
                raise ValueError("Obstacle center lies outside the map.")

    @property
    def start_array(self) -> np.ndarray:
        return np.asarray(self.start, dtype=float)

    @property
    def goal_array(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=float)

    @property
    def dimension(self) -> int:
        """Dimension of GA/GWO/PSO waypoint vectors."""

        return self.num_waypoints * 2

    def waypoint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds for a flattened waypoint vector."""

        lower = np.tile(np.array([0.0, 0.0]), self.num_waypoints)
        upper = np.tile(np.array([self.width, self.height]), self.num_waypoints)
        return lower, upper


DEFAULT_SCENARIO = Scenario(
    width=100.0,
    height=100.0,
    start=(5.0, 5.0),
    goal=(95.0, 95.0),
   obstacles=(
    CircleObstacle(center=(20.0, 20.0), radius=7.0),
    CircleObstacle(center=(38.0, 42.0), radius=9.0),
    CircleObstacle(center=(60.0, 24.0), radius=9.0),
    CircleObstacle(center=(28.0, 68.0), radius=8.0),
    CircleObstacle(center=(74.0, 55.0), radius=8.0),
    CircleObstacle(center=(58.0, 80.0), radius=8.0),
),
    safety_margin=3.0,
    num_waypoints=5,
)

DEFAULT_OBJECTIVE_WEIGHTS = ObjectiveWeights()
