from __future__ import annotations

import numpy as np
import pytest

from core.environment import PathPlanning3DEnv
from evaluate import build_parser, calculate_trajectory_metrics


def test_trajectory_metrics_use_start_to_goal_distance() -> None:
    env = PathPlanning3DEnv("dqn", obstacles=[])
    env.reset(
        seed=0,
        options={"start": [10.0, 10.0, 10.0], "goal": [14.0, 14.0, 10.0]},
    )
    trajectory = np.array(
        [
            [10.0, 10.0, 10.0],
            [14.0, 10.0, 10.0],
            [14.0, 14.0, 10.0],
        ]
    )
    controls = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    metrics = calculate_trajectory_metrics(env, trajectory, controls)

    assert metrics["path_length"] == pytest.approx(8.0)
    assert metrics["direct_displacement"] == pytest.approx(np.sqrt(32.0))
    assert metrics["path_efficiency"] == pytest.approx(np.sqrt(32.0) / 8.0)
    assert metrics["direction_changes"] == 1
    assert metrics["total_turning_angle_degrees"] == pytest.approx(90.0)
    assert metrics["trajectory_roughness"] == pytest.approx(32.0)
    assert metrics["control_variation"] == pytest.approx(np.sqrt(2.0))


def test_trajectory_metrics_reject_invalid_shape() -> None:
    env = PathPlanning3DEnv("dqn", obstacles=[])
    with pytest.raises(ValueError):
        calculate_trajectory_metrics(env, np.zeros((3, 2)), np.zeros((2, 3)))


def test_evaluate_defaults_to_all_algorithms_and_gifs() -> None:
    args = build_parser().parse_args([])
    assert args.algorithm == "all"
    assert not args.no_gif
