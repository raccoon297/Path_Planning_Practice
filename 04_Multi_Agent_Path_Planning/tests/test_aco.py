"""Fast structural tests for the Multi-Agent ACO."""

from __future__ import annotations

import unittest

import numpy as np

from config.scenario import DEFAULT_SCENARIO
from optimizers.aco import MultiAgentACOConfig, build_grid_graph, run_multi_agent_aco


class MultiAgentACOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MultiAgentACOConfig(
            num_ants=8,
            max_iterations=2,
            grid_resolution=5.0,
            recorded_colony_plans=3,
        )
        cls.result = run_multi_agent_aco(DEFAULT_SCENARIO, cls.config, seed=7)

    def test_grid_contains_all_tasks(self) -> None:
        graph = build_grid_graph(DEFAULT_SCENARIO, self.config)
        self.assertEqual(graph.rows, 21)
        self.assertEqual(graph.cols, 21)
        self.assertEqual(len(graph.start_indices), DEFAULT_SCENARIO.num_agents)
        self.assertEqual(len(graph.goal_indices), DEFAULT_SCENARIO.num_agents)

    def test_result_shapes(self) -> None:
        scenario = DEFAULT_SCENARIO
        self.assertEqual(
            self.result.fitness_history.shape,
            (self.config.max_iterations + 1,),
        )
        self.assertEqual(
            self.result.pheromone_history.shape,
            (
                self.config.max_iterations + 1,
                scenario.num_agents,
                self.result.graph_rows,
                self.result.graph_cols,
            ),
        )
        self.assertEqual(len(self.result.plan.paths), scenario.num_agents)

    def test_best_so_far_history_is_nonincreasing(self) -> None:
        finite = self.result.fitness_history[np.isfinite(self.result.fitness_history)]
        self.assertGreater(len(finite), 0)
        self.assertTrue(np.all(np.diff(finite) <= 1e-9))

    def test_delay_normalization_and_finite_result(self) -> None:
        self.assertAlmostEqual(float(self.result.plan.start_delays.min()), 0.0)
        self.assertTrue(np.isfinite(self.result.best_fitness))
        self.assertTrue(
            np.all(self.result.plan.start_delays <= DEFAULT_SCENARIO.max_start_delay)
        )


if __name__ == "__main__":
    unittest.main()
