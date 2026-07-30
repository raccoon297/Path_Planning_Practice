"""Shared scenario definitions for centralized multi-agent path planning."""

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
        if self.radius <= 0.0:
            raise ValueError("Obstacle radius must be positive.")

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=float)


@dataclass(frozen=True)
class AgentTask:
    """Fixed start and goal assigned to one agent."""

    name: str
    start: Point
    goal: Point

    @property
    def start_array(self) -> np.ndarray:
        return np.asarray(self.start, dtype=float)

    @property
    def goal_array(self) -> np.ndarray:
        return np.asarray(self.goal, dtype=float)


@dataclass(frozen=True)
class MultiAgentObjectiveWeights:
    """Weights for one centralized joint-plan objective."""

    length: float = 1.0
    obstacle_collision: float = 10_000.0
    obstacle_clearance: float = 100.0
    inter_agent_collision: float = 15_000.0
    inter_agent_clearance: float = 500.0
    smoothness: float = 5.0
    backtracking: float = 100.0
    waypoint_spacing: float = 200.0
    boundary: float = 10_000.0
    start_delay: float = 1.0
    makespan: float = 2.0


@dataclass(frozen=True)
class MultiAgentScenario:
    """Centralized cooperative path-planning problem for multiple agents."""

    width: float
    height: float
    tasks: Tuple[AgentTask, ...]
    obstacles: Tuple[CircleObstacle, ...]
    obstacle_safety_margin: float = 3.0
    boundary_safety_margin: float = 3.0
    num_waypoints: int = 5
    speed: float = 5.0
    agent_radius: float = 1.0
    minimum_agent_separation: float = 6.0
    max_start_delay: float = 15.0
    time_step: float = 0.1

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Map width and height must be positive.")
        if len(self.tasks) < 2:
            raise ValueError("A multi-agent scenario requires at least two tasks.")
        if self.obstacle_safety_margin < 0.0:
            raise ValueError("Obstacle safety margin cannot be negative.")
        if self.boundary_safety_margin < 0.0:
            raise ValueError("Boundary safety margin cannot be negative.")
        if 2.0 * self.boundary_safety_margin >= min(self.width, self.height):
            raise ValueError("Boundary safety margin leaves no usable map interior.")
        if self.num_waypoints < 1:
            raise ValueError("num_waypoints must be positive.")
        if self.speed <= 0.0:
            raise ValueError("speed must be positive.")
        if self.agent_radius <= 0.0:
            raise ValueError("agent_radius must be positive.")
        if self.minimum_agent_separation < 2.0 * self.agent_radius:
            raise ValueError(
                "minimum_agent_separation must be at least the physical collision distance."
            )
        if self.max_start_delay < 0.0:
            raise ValueError("max_start_delay cannot be negative.")
        if self.time_step <= 0.0:
            raise ValueError("time_step must be positive.")

        for task in self.tasks:
            for label, point in (("start", task.start), ("goal", task.goal)):
                x, y = point
                if not (0.0 <= x <= self.width and 0.0 <= y <= self.height):
                    raise ValueError(f"{task.name} {label} lies outside the map.")
                wall_clearance = min(x, self.width - x, y, self.height - y)
                if wall_clearance < self.boundary_safety_margin:
                    raise ValueError(
                        f"{task.name} {label} violates the boundary safety margin."
                    )

        for obstacle in self.obstacles:
            x, y = obstacle.center
            if not (0.0 <= x <= self.width and 0.0 <= y <= self.height):
                raise ValueError("Obstacle center lies outside the map.")

    @property
    def num_agents(self) -> int:
        return len(self.tasks)

    @property
    def physical_agent_collision_distance(self) -> float:
        return 2.0 * self.agent_radius

    @property
    def agent_block_dimension(self) -> int:
        """Waypoint coordinates plus one start-delay variable for one agent."""

        return 2 * self.num_waypoints + 1

    @property
    def dimension(self) -> int:
        """Dimension of one complete joint-plan candidate vector."""

        return self.num_agents * self.agent_block_dimension

    def candidate_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lower and upper bounds for a flattened joint-plan vector."""

        lower_blocks: list[np.ndarray] = []
        upper_blocks: list[np.ndarray] = []
        margin = self.boundary_safety_margin
        waypoint_lower = np.tile(np.array([margin, margin]), self.num_waypoints)
        waypoint_upper = np.tile(
            np.array([self.width - margin, self.height - margin]),
            self.num_waypoints,
        )
        for _ in self.tasks:
            lower_blocks.append(np.concatenate([waypoint_lower, [0.0]]))
            upper_blocks.append(
                np.concatenate([waypoint_upper, [self.max_start_delay]])
            )
        return np.concatenate(lower_blocks), np.concatenate(upper_blocks)


DEFAULT_SCENARIO = MultiAgentScenario(
    width=100.0,
    height=100.0,
    tasks=(
        AgentTask(name="Agent 1", start=(5.0, 5.0), goal=(95.0, 95.0)),
        AgentTask(name="Agent 2", start=(10.0, 5.0), goal=(90.0, 95.0)),
        AgentTask(name="Agent 3", start=(5.0, 10.0), goal=(95.0, 90.0)),
    ),
    obstacles=(
        CircleObstacle(center=(38.0, 42.0), radius=9.0),
        CircleObstacle(center=(60.0, 24.0), radius=9.0),
        CircleObstacle(center=(28.0, 68.0), radius=8.0),
        CircleObstacle(center=(74.0, 55.0), radius=8.0),
    ),
    obstacle_safety_margin=3.0,
    boundary_safety_margin=3.0,
    num_waypoints=5,
    speed=5.0,
    agent_radius=1.0,
    minimum_agent_separation=3.0,
    max_start_delay=15.0,
    time_step=0.1,
)

DEFAULT_OBJECTIVE_WEIGHTS = MultiAgentObjectiveWeights()
