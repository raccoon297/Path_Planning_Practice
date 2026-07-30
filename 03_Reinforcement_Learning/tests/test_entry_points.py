from __future__ import annotations

import math

from config.scenario import SCENARIO_CONFIG
from train import build_parser


def test_train_defaults_to_all_algorithms() -> None:
    args = build_parser().parse_args([])
    assert args.algorithm == "all"


def test_ppo_default_budget_and_early_stopping() -> None:
    settings = SCENARIO_CONFIG["ppo"]
    assert settings["total_timesteps"] == 500_000
    assert settings["rollout_steps"] == 2_048
    assert math.ceil(settings["total_timesteps"] / settings["rollout_steps"]) == 245

    early_stopping = settings["early_stopping"]
    assert early_stopping["enabled"]
    assert early_stopping["minimum_timesteps"] == 200_000
    assert early_stopping["patience_evaluations"] == 5
