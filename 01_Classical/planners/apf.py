"""Artificial Potential Field path planner with deterministic local-minimum recovery."""

from __future__ import annotations

import math
import time
from typing import List, Tuple

import numpy as np

from config.scenarios import CircleObstacle, Point, Scenario
from utils.collision import is_within_bounds, point_in_collision, segment_in_collision
from utils.metrics import PlanningResult, calculate_minimum_clearance, calculate_path_length


class APFPlanner:
    """Continuous APF planner evaluated through the common project interface."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 35.0,
        influence_distance: float = 4.0,
        step_size: float = 0.25,
        goal_tolerance: float = 0.5,
        max_iterations: int = 2500,
        stagnation_window: int = 30,
        progress_tolerance: float = 0.08,
    ) -> None:
        self.scenario = scenario
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance
        self.step_size = step_size
        self.goal_tolerance = goal_tolerance
        self.max_iterations = max_iterations
        self.stagnation_window = stagnation_window
        self.progress_tolerance = progress_tolerance

    def _attractive_force(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        delta = target - current
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            return np.zeros(2, dtype=float)
        return self.attractive_gain * delta / distance

    def _repulsive_force(self, current: np.ndarray) -> np.ndarray:
        force = np.zeros(2, dtype=float)
        for obstacle in self.scenario.obstacles:
            center = np.array([obstacle.x, obstacle.y], dtype=float)
            delta = current - center
            center_distance = float(np.linalg.norm(delta))
            inflated_radius = obstacle.radius + self.scenario.safety_margin
            boundary_distance = center_distance - inflated_radius

            if boundary_distance >= self.influence_distance:
                continue

            direction = (
                delta / center_distance
                if center_distance > 1e-12
                else np.array([1.0, 0.0], dtype=float)
            )
            safe_distance = max(boundary_distance, 0.05)
            magnitude = self.repulsive_gain * (
                (1.0 / safe_distance) - (1.0 / self.influence_distance)
            ) / (safe_distance**2)
            force += magnitude * direction
        return force

    def _nearest_obstacle(self, current: np.ndarray) -> CircleObstacle:
        return min(
            self.scenario.obstacles,
            key=lambda obstacle: math.hypot(
                current[0] - obstacle.x,
                current[1] - obstacle.y,
            )
            - obstacle.radius,
        )

    def _make_escape_waypoint(
        self,
        current: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        """Create a deterministic temporary target around the nearest obstacle."""

        obstacle = self._nearest_obstacle(current)
        center = np.array([obstacle.x, obstacle.y], dtype=float)
        radial_angle = math.atan2(current[1] - center[1], current[0] - center[0])
        goal_angle = math.atan2(goal[1] - center[1], goal[0] - center[0])

        clockwise = (goal_angle - radial_angle) % (2.0 * math.pi)
        counterclockwise = (radial_angle - goal_angle) % (2.0 * math.pi)
        side = 1.0 if clockwise <= counterclockwise else -1.0

        waypoint_radius = (
            obstacle.radius
            + self.scenario.safety_margin
            + self.influence_distance
            + 1.0
        )
        candidate_angles = (
            radial_angle + side * math.radians(70.0),
            radial_angle - side * math.radians(70.0),
            radial_angle + side * math.radians(110.0),
            radial_angle - side * math.radians(110.0),
        )

        candidates: List[np.ndarray] = []
        for angle in candidate_angles:
            candidate = center + waypoint_radius * np.array(
                [math.cos(angle), math.sin(angle)], dtype=float
            )
            point = (float(candidate[0]), float(candidate[1]))
            if not is_within_bounds(point, self.scenario.width, self.scenario.height):
                continue
            if point_in_collision(
                point, self.scenario.obstacles, self.scenario.safety_margin
            ):
                continue
            candidates.append(candidate)

        if not candidates:
            return current + np.array([0.0, self.step_size], dtype=float)

        return min(
            candidates,
            key=lambda candidate: float(np.linalg.norm(candidate - current))
            + float(np.linalg.norm(goal - candidate)),
        )

    def _safe_step(
        self,
        current: np.ndarray,
        force: np.ndarray,
    ) -> Tuple[np.ndarray | None, bool]:
        norm = float(np.linalg.norm(force))
        if norm <= 1e-12:
            return None, False

        desired_heading = math.atan2(force[1], force[0])
        offsets = (
            0.0,
            math.radians(10.0),
            -math.radians(10.0),
            math.radians(20.0),
            -math.radians(20.0),
            math.radians(35.0),
            -math.radians(35.0),
            math.radians(60.0),
            -math.radians(60.0),
            math.radians(90.0),
            -math.radians(90.0),
        )
        current_point = (float(current[0]), float(current[1]))

        for index, offset in enumerate(offsets):
            heading = desired_heading + offset
            candidate = current + self.step_size * np.array(
                [math.cos(heading), math.sin(heading)], dtype=float
            )
            candidate_point = (float(candidate[0]), float(candidate[1]))
            if not is_within_bounds(
                candidate_point, self.scenario.width, self.scenario.height
            ):
                continue
            if point_in_collision(
                candidate_point,
                self.scenario.obstacles,
                self.scenario.safety_margin,
            ):
                continue
            if segment_in_collision(
                current_point,
                candidate_point,
                self.scenario.obstacles,
                self.scenario.safety_margin,
            ):
                continue
            return candidate, index != 0

        return None, False

    def plan(self) -> PlanningResult:
        start_time = time.perf_counter()
        current = np.array(self.scenario.start, dtype=float)
        goal = np.array(self.scenario.goal, dtype=float)

        for name, point in (("Start", self.scenario.start), ("Goal", self.scenario.goal)):
            if point_in_collision(
                point, self.scenario.obstacles, self.scenario.safety_margin
            ):
                raise ValueError(f"{name} lies inside an obstacle or safety margin.")

        path: List[Point] = [self.scenario.start]
        recent_distances: List[float] = []
        temporary_target: np.ndarray | None = None
        local_minimum_recoveries = 0
        collision_avoidance_adjustments = 0
        failure_reason = "maximum_iterations"

        for iteration in range(1, self.max_iterations + 1):
            goal_distance = float(np.linalg.norm(goal - current))
            if goal_distance <= self.goal_tolerance:
                path.append(self.scenario.goal)
                failure_reason = ""
                break

            if temporary_target is not None and float(
                np.linalg.norm(temporary_target - current)
            ) <= 0.75:
                temporary_target = None
                recent_distances.clear()

            target = temporary_target if temporary_target is not None else goal
            resultant = self._attractive_force(current, target) + self._repulsive_force(
                current
            )

            recent_distances.append(goal_distance)
            if len(recent_distances) > self.stagnation_window:
                recent_distances.pop(0)

            stagnating = (
                temporary_target is None
                and len(recent_distances) == self.stagnation_window
                and recent_distances[0] - min(recent_distances)
                < self.progress_tolerance
            )
            if stagnating:
                temporary_target = self._make_escape_waypoint(current, goal)
                target = temporary_target
                resultant = self._attractive_force(current, target) + self._repulsive_force(
                    current
                )
                local_minimum_recoveries += 1
                recent_distances.clear()

            next_position, adjusted = self._safe_step(current, resultant)
            if adjusted:
                collision_avoidance_adjustments += 1
            if next_position is None:
                failure_reason = "no_collision_free_step"
                break

            current = next_position
            path.append((float(current[0]), float(current[1])))
        else:
            iteration = self.max_iterations

        success = math.hypot(
            path[-1][0] - self.scenario.goal[0],
            path[-1][1] - self.scenario.goal[1],
        ) <= self.goal_tolerance
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PlanningResult(
            algorithm="APF",
            success=success,
            path=path,
            planning_time_ms=elapsed_ms,
            path_length=calculate_path_length(path),
            waypoint_count=len(path),
            minimum_clearance=calculate_minimum_clearance(path, self.scenario),
            details={
                "iterations": iteration,
                "attractive_gain": self.attractive_gain,
                "repulsive_gain": self.repulsive_gain,
                "influence_distance": self.influence_distance,
                "step_size": self.step_size,
                "goal_tolerance": self.goal_tolerance,
                "local_minimum_recoveries": local_minimum_recoveries,
                "collision_avoidance_adjustments": collision_avoidance_adjustments,
                "failure_reason": failure_reason if not success else "",
                "recovery_method": "temporary_tangential_waypoint",
            },
        )
