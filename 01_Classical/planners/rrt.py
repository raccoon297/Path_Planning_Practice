"""Rapidly-exploring Random Tree path planner."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from config.scenarios import Point, Scenario
from utils.collision import is_within_bounds, point_in_collision, segment_in_collision
from utils.metrics import PlanningResult, calculate_minimum_clearance, calculate_path_length


@dataclass
class _Node:
    point: Point
    parent_index: Optional[int]


class RRTPlanner:
    """Continuous-space RRT planner with deterministic random sampling."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        step_size: float = 1.25,
        goal_sample_rate: float = 0.15,
        goal_tolerance: float = 1.25,
        max_iterations: int = 5000,
    ) -> None:
        self.scenario = scenario
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.max_iterations = max_iterations

    def _sample(self, rng: np.random.Generator) -> Point:
        if rng.random() < self.goal_sample_rate:
            return self.scenario.goal
        return (
            float(rng.uniform(0.0, self.scenario.width)),
            float(rng.uniform(0.0, self.scenario.height)),
        )

    @staticmethod
    def _nearest_node_index(nodes: List[_Node], sample: Point) -> int:
        sx, sy = sample
        return min(
            range(len(nodes)),
            key=lambda index: math.hypot(
                nodes[index].point[0] - sx,
                nodes[index].point[1] - sy,
            ),
        )

    def _steer(self, source: Point, target: Point) -> Point:
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        distance = math.hypot(dx, dy)
        if distance <= self.step_size:
            return target
        scale = self.step_size / distance
        return source[0] + dx * scale, source[1] + dy * scale

    @staticmethod
    def _reconstruct_path(nodes: List[_Node], goal_index: int) -> List[Point]:
        path: List[Point] = []
        current_index: Optional[int] = goal_index
        while current_index is not None:
            node = nodes[current_index]
            path.append(node.point)
            current_index = node.parent_index
        path.reverse()
        return path

    def plan(self) -> PlanningResult:
        """Build an RRT and return a path in the common result format."""

        start_time = time.perf_counter()
        scenario = self.scenario
        rng = np.random.default_rng(scenario.random_seed)

        if not is_within_bounds(scenario.start, scenario.width, scenario.height):
            raise ValueError("Start lies outside the map boundary.")
        if not is_within_bounds(scenario.goal, scenario.width, scenario.height):
            raise ValueError("Goal lies outside the map boundary.")
        if point_in_collision(scenario.start, scenario.obstacles, scenario.safety_margin):
            raise ValueError("Start lies inside an obstacle or safety margin.")
        if point_in_collision(scenario.goal, scenario.obstacles, scenario.safety_margin):
            raise ValueError("Goal lies inside an obstacle or safety margin.")

        nodes = [_Node(scenario.start, None)]
        tree_edges: List[tuple[Point, Point]] = []
        rejected_samples = 0
        goal_index: Optional[int] = None
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            sample = self._sample(rng)
            nearest_index = self._nearest_node_index(nodes, sample)
            nearest_point = nodes[nearest_index].point
            new_point = self._steer(nearest_point, sample)

            if not is_within_bounds(new_point, scenario.width, scenario.height):
                rejected_samples += 1
                continue
            if point_in_collision(new_point, scenario.obstacles, scenario.safety_margin):
                rejected_samples += 1
                continue
            if segment_in_collision(
                nearest_point,
                new_point,
                scenario.obstacles,
                scenario.safety_margin,
            ):
                rejected_samples += 1
                continue

            nodes.append(_Node(new_point, nearest_index))
            tree_edges.append((nearest_point, new_point))
            new_index = len(nodes) - 1

            distance_to_goal = math.hypot(
                new_point[0] - scenario.goal[0],
                new_point[1] - scenario.goal[1],
            )
            if distance_to_goal <= self.goal_tolerance and not segment_in_collision(
                new_point,
                scenario.goal,
                scenario.obstacles,
                scenario.safety_margin,
            ):
                nodes.append(_Node(scenario.goal, new_index))
                tree_edges.append((new_point, scenario.goal))
                goal_index = len(nodes) - 1
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        path = self._reconstruct_path(nodes, goal_index) if goal_index is not None else []

        return PlanningResult(
            algorithm="RRT",
            success=bool(path),
            path=path,
            planning_time_ms=elapsed_ms,
            path_length=calculate_path_length(path),
            waypoint_count=len(path),
            minimum_clearance=calculate_minimum_clearance(path, scenario),
            details={
                "iterations": iterations,
                "tree_nodes": len(nodes),
                "rejected_samples": rejected_samples,
                "step_size": self.step_size,
                "goal_sample_rate": self.goal_sample_rate,
                "goal_tolerance": self.goal_tolerance,
                "max_iterations": self.max_iterations,
                "random_seed": scenario.random_seed,
            },
            visualization_data={
                "tree_edges": tree_edges,
                "goal_reached_iteration": iterations if goal_index is not None else None,
            },
        )
