"""Fast structural tests for the Multi-Agent GWO."""

from __future__ import annotations

import unittest

import numpy as np

from config.scenario import DEFAULT_SCENARIO
from optimizers.gwo import (
    MultiAgentGWOConfig,
    _reflect_bounds,
    _update_leader_archive,
    run_multi_agent_gwo,
)


class MultiAgentGWOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MultiAgentGWOConfig(num_wolves=10, max_iterations=3)
        cls.result = run_multi_agent_gwo(DEFAULT_SCENARIO, cls.config, seed=7)

    def test_result_shapes(self) -> None:
        scenario = DEFAULT_SCENARIO
        self.assertEqual(self.result.best_vector.shape, (scenario.dimension,))
        self.assertEqual(
            self.result.fitness_history.shape,
            (self.config.max_iterations + 1,),
        )
        self.assertEqual(
            self.result.population_history.shape,
            (
                self.config.max_iterations + 1,
                self.config.num_wolves,
                scenario.dimension,
            ),
        )
        self.assertEqual(
            self.result.leader_vector_history.shape,
            (self.config.max_iterations + 1, 3, scenario.dimension),
        )

    def test_best_so_far_history_is_nonincreasing(self) -> None:
        differences = np.diff(self.result.fitness_history)
        self.assertTrue(np.all(differences <= 1e-9))

    def test_delay_normalization(self) -> None:
        self.assertAlmostEqual(float(self.result.plan.start_delays.min()), 0.0)
        self.assertTrue(
            np.all(self.result.plan.start_delays <= DEFAULT_SCENARIO.max_start_delay)
        )

    def test_result_is_finite(self) -> None:
        self.assertTrue(np.isfinite(self.result.best_fitness))
        self.assertTrue(np.all(np.isfinite(self.result.best_vector)))

    def test_reflection_returns_values_to_interior(self) -> None:
        lower = np.array([3.0, 3.0])
        upper = np.array([97.0, 97.0])
        values = np.array([[-7.0, 105.0], [201.0, -101.0]])
        repaired = _reflect_bounds(values, lower, upper)
        self.assertTrue(np.all(repaired >= lower))
        self.assertTrue(np.all(repaired <= upper))
        self.assertFalse(np.any(np.isclose(repaired, lower)))
        self.assertFalse(np.any(np.isclose(repaired, upper)))

    def test_leader_archive_preserves_best_so_far_wolves(self) -> None:
        leaders = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        leader_fitness = np.array([1.0, 2.0, 3.0])
        positions = np.array([[0.5, 0.5], [4.0, 4.0], [5.0, 5.0]])
        fitness = np.array([0.5, 4.0, 5.0])
        updated_positions, updated_fitness = _update_leader_archive(
            leaders, leader_fitness, positions, fitness
        )
        np.testing.assert_allclose(updated_fitness, np.array([0.5, 1.0, 2.0]))
        np.testing.assert_allclose(updated_positions[0], positions[0])
        np.testing.assert_allclose(updated_positions[1], leaders[0])
        np.testing.assert_allclose(updated_positions[2], leaders[1])

    def test_all_gwo_waypoints_respect_boundary_margin(self) -> None:
        scenario = DEFAULT_SCENARIO
        block = scenario.agent_block_dimension
        waypoint_values = 2 * scenario.num_waypoints
        waypoint_columns = []
        for agent_index in range(scenario.num_agents):
            start = agent_index * block
            waypoint_columns.extend(range(start, start + waypoint_values))
        waypoint_values_history = self.result.population_history[:, :, waypoint_columns]
        self.assertGreaterEqual(
            float(waypoint_values_history.min()), scenario.boundary_safety_margin
        )
        self.assertLessEqual(
            float(waypoint_values_history.max()),
            max(scenario.width, scenario.height) - scenario.boundary_safety_margin,
        )


if __name__ == "__main__":
    unittest.main()
