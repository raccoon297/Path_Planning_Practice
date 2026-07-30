"""Tests for result metadata and comparison validation."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from config.scenario import DEFAULT_SCENARIO
from run_comparison import validate_result
from utils.reporting import scenario_as_dict


class ComparisonValidationTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "scenario": scenario_as_dict(DEFAULT_SCENARIO),
            "paths": [
                [list(task.start), list(task.goal)] for task in DEFAULT_SCENARIO.tasks
            ],
            "start_delays": [0.0] * DEFAULT_SCENARIO.num_agents,
            "metrics": {
                "success": True,
                "total_path_length": 1.0,
                "makespan": 1.0,
                "sum_start_delay": 0.0,
                "minimum_obstacle_clearance": 3.0,
                "minimum_boundary_clearance": 3.0,
                "minimum_inter_agent_distance": 3.0,
                "obstacle_collision_count": 0,
                "obstacle_safety_violation_count": 0,
                "boundary_safety_violation_count": 0,
                "inter_agent_collision_episodes": 0,
                "inter_agent_separation_violation_episodes": 0,
                "total_smoothness": 0.0,
                "total_backtracking": 0.0,
                "waypoint_spacing_imbalance": 0.0,
            },
        }

    def test_current_payload_is_accepted(self) -> None:
        validate_result(self._payload(), Path("dummy.json"))

    def test_stale_start_point_is_rejected(self) -> None:
        payload = copy.deepcopy(self._payload())
        payload["paths"][0][0] = [99.0, 99.0]
        with self.assertRaisesRegex(ValueError, "stale start point"):
            validate_result(payload, Path("dummy.json"))

    def test_missing_scenario_metadata_is_rejected(self) -> None:
        payload = self._payload()
        del payload["scenario"]
        with self.assertRaisesRegex(ValueError, "no scenario metadata"):
            validate_result(payload, Path("dummy.json"))


if __name__ == "__main__":
    unittest.main()
