"""Shared path-planning scenarios.

All planners receive the same :class:`Scenario` object so that their results
can be compared under identical map, start, goal, obstacle, and safety-margin
conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class CircleObstacle:
    """Circular obstacle in continuous 2-D coordinates."""

    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class Scenario:
    """Common environment definition used by every planner."""

    name: str
    width: int
    height: int
    start: Point
    goal: Point
    obstacles: Tuple[CircleObstacle, ...]
    safety_margin: float = 0.5
    grid_resolution: float = 1.0
    random_seed: int = 42


@dataclass(frozen=True)
class DynamicScenario:
    """Scenario used later for D* Lite replanning experiments."""

    name: str
    width: int
    height: int
    start: Point
    goal: Point
    static_obstacles: Tuple[CircleObstacle, ...]
    hidden_obstacles: Tuple[CircleObstacle, ...]
    sensor_range: float
    safety_margin: float = 0.5
    grid_resolution: float = 1.0
    random_seed: int = 42


def get_static_scenario() -> Scenario:
    """Return the common map for the four-algorithm static comparison."""

    return Scenario(
        name="static_comparison",
        width=50,
        height=50,
        start=(5.0, 5.0),
        goal=(45.0, 45.0),
        obstacles=(
            CircleObstacle(15.0, 15.0, 6.0),
            CircleObstacle(30.0, 10.0, 7.0),
            CircleObstacle(25.0, 35.0, 8.0),
        ),
        safety_margin=0.5,
        grid_resolution=1.0,
        random_seed=42,
    )


def get_dynamic_scenario() -> DynamicScenario:
    """Return the planned D* Lite dynamic-replanning scenario.

    The exact hidden-obstacle position will be tuned after the static planners
    are implemented and their paths are inspected together.
    """

    static = get_static_scenario()
    return DynamicScenario(
        name="dynamic_replanning",
        width=static.width,
        height=static.height,
        start=static.start,
        goal=static.goal,
        static_obstacles=static.obstacles,
        hidden_obstacles=(CircleObstacle(36.0, 34.0, 7.0),),
        sensor_range=8.0,
        safety_margin=static.safety_margin,
        grid_resolution=static.grid_resolution,
        random_seed=static.random_seed,
    )
