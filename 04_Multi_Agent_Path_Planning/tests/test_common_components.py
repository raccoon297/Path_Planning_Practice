"""Regression tests for shared multi-agent planning components."""

from __future__ import annotations

import unittest

import numpy as np

from config.scenario import AgentTask, DEFAULT_SCENARIO, MultiAgentScenario
from utils.metrics import compute_joint_plan_metrics
from utils.path_utils import (
    JointPlan,
    backtracking_cost,
    decode_joint_vector,
    encode_joint_plan,
    normalize_start_delays,
    waypoint_spacing_cost,
)


def _straight_waypoints(start: np.ndarray, goal: np.ndarray, count: int) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, count + 2)[1:-1]
    return start + fractions[:, None] * (goal - start)


def _simple_crossing_scenario() -> MultiAgentScenario:
    return MultiAgentScenario(
        width=20.0,
        height=20.0,
        tasks=(
            AgentTask("Agent 1", start=(2.0, 10.0), goal=(18.0, 10.0)),
            AgentTask("Agent 2", start=(10.0, 2.0), goal=(10.0, 18.0)),
        ),
        obstacles=(),
        obstacle_safety_margin=0.0,
        boundary_safety_margin=1.0,
        num_waypoints=1,
        speed=2.0,
        agent_radius=0.5,
        minimum_agent_separation=2.0,
        max_start_delay=5.0,
        time_step=0.05,
    )


class CommonComponentTests(unittest.TestCase):
    def test_default_scenario_matches_final_layout(self) -> None:
        scenario = DEFAULT_SCENARIO
        self.assertEqual(len(scenario.obstacles), 4)
        self.assertEqual(scenario.minimum_agent_separation, 3.0)
        self.assertEqual(scenario.tasks[0].start, (5.0, 5.0))
        self.assertEqual(scenario.tasks[1].start, (10.0, 5.0))
        self.assertEqual(scenario.tasks[2].start, (5.0, 10.0))

    def test_candidate_bounds_respect_boundary_margin(self) -> None:
        scenario = DEFAULT_SCENARIO
        lower, upper = scenario.candidate_bounds()
        block = scenario.agent_block_dimension
        waypoint_values = 2 * scenario.num_waypoints
        for agent_index in range(scenario.num_agents):
            start = agent_index * block
            np.testing.assert_allclose(
                lower[start : start + waypoint_values],
                scenario.boundary_safety_margin,
            )
            expected_upper = np.tile(
                np.array(
                    [
                        scenario.width - scenario.boundary_safety_margin,
                        scenario.height - scenario.boundary_safety_margin,
                    ]
                ),
                scenario.num_waypoints,
            )
            np.testing.assert_allclose(
                upper[start : start + waypoint_values], expected_upper
            )

    def test_delay_normalization_removes_common_offset(self) -> None:
        normalized = normalize_start_delays(np.array([4.0, 7.0, 9.0]), 15.0)
        np.testing.assert_allclose(normalized, np.array([0.0, 3.0, 5.0]))

    def test_joint_vector_dimension_and_round_trip(self) -> None:
        scenario = DEFAULT_SCENARIO
        waypoint_groups = [
            _straight_waypoints(
                task.start_array, task.goal_array, scenario.num_waypoints
            )
            for task in scenario.tasks
        ]
        delays = np.array([0.0, 1.0, 2.0])
        vector = encode_joint_plan(waypoint_groups, delays, scenario)
        self.assertEqual(vector.size, scenario.dimension)
        decoded = decode_joint_vector(vector, scenario)
        np.testing.assert_allclose(decoded.start_delays, delays)
        for index, path in enumerate(decoded.paths):
            np.testing.assert_allclose(path[1:-1], waypoint_groups[index])

    def test_direct_crossing_plan_is_rejected(self) -> None:
        scenario = _simple_crossing_scenario()
        plan = JointPlan(
            paths=(
                np.array([[2.0, 10.0], [10.0, 10.0], [18.0, 10.0]]),
                np.array([[10.0, 2.0], [10.0, 10.0], [10.0, 18.0]]),
            ),
            start_delays=np.zeros(2),
        )
        metrics = compute_joint_plan_metrics(plan, scenario)
        self.assertFalse(metrics.success)
        self.assertGreater(metrics.inter_agent_collision_episodes, 0)

    def test_plan_inside_wall_margin_is_rejected(self) -> None:
        scenario = DEFAULT_SCENARIO
        paths = []
        for task in scenario.tasks:
            waypoints = _straight_waypoints(
                task.start_array, task.goal_array, scenario.num_waypoints
            )
            paths.append(np.vstack([task.start_array, waypoints, task.goal_array]))
        paths[0][1, 0] = 1.0
        plan = JointPlan(paths=tuple(paths), start_delays=np.zeros(3))
        metrics = compute_joint_plan_metrics(plan, scenario)
        self.assertFalse(metrics.boundary_margin_satisfied)
        self.assertGreater(metrics.boundary_safety_violation_count, 0)

    def test_path_regularizers_reward_ordered_waypoints(self) -> None:
        ordered = np.array(
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]]
        )
        reversed_path = np.array(
            [[0.0, 0.0], [4.0, 0.0], [2.0, 0.0], [6.0, 0.0]]
        )
        clustered = np.array(
            [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [6.0, 0.0]]
        )
        self.assertAlmostEqual(backtracking_cost(ordered), 0.0)
        self.assertGreater(backtracking_cost(reversed_path), 0.0)
        self.assertAlmostEqual(waypoint_spacing_cost(ordered), 0.0)
        self.assertGreater(waypoint_spacing_cost(clustered), 0.0)


if __name__ == "__main__":
    unittest.main()
