"""Fast structural tests for the Multi-Agent PSO."""

from __future__ import annotations

import unittest

import numpy as np

from config.scenario import DEFAULT_SCENARIO
from optimizers.pso import MultiAgentPSOConfig, run_multi_agent_pso


class MultiAgentPSOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = MultiAgentPSOConfig(num_particles=10, max_iterations=3)
        cls.result = run_multi_agent_pso(DEFAULT_SCENARIO, cls.config, seed=7)

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
                self.config.num_particles,
                scenario.dimension,
            ),
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


if __name__ == "__main__":
    unittest.main()
