"""Configuration for the fixed 3D urban path-planning scenario.

The values in this module are project-specific design choices. They are kept in
plain Python so the repository does not require an additional YAML parser and
matches the configuration style used by the other Path Planning Practice
projects.
"""

from __future__ import annotations

SCENARIO_CONFIG = {
    "project": {
        "seed": 42,
    },
    "environment": {
        "workspace_size": 100.0,
        "ground_level": 0.0,
        "max_building_height": 60.0,
        "start": [6.0, 6.0, 8.0],
        "goal": [94.0, 94.0, 8.0],
        "agent_radius": 1.0,
        "safety_margin": 2.0,
        "goal_radius": 3.0,
        "max_steps": 250,
        "sensor_range": 18.0,
        "ray_count": 26,
        "reward": {
            "goal": 250.0,
            "collision": -250.0,
            "progress_scale": 2.0,
            "step": -0.1,
            "clearance_scale": 0.5,
            "control_change_scale": 0.02,
        },
        "dqn": {
            "step_size": 2.0,
        },
        "ppo": {
            "dt": 1.0,
            "max_speed": 2.0,
            "max_acceleration": 0.5,
        },
    },
    "training": {
        "evaluation_episodes": 100,
    },
    "dqn": {
        "episodes": 1_500,
        "hidden_sizes": [256, 256],
        "learning_rate": 0.0005,
        "gamma": 0.99,
        "batch_size": 128,
        "replay_capacity": 100_000,
        "learning_starts": 2_000,
        "train_frequency": 1,
        "target_update_interval": 1_000,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 100_000,
        "gradient_clip_norm": 10.0,
        "print_every": 25,
        "checkpoint_every": 100,
        "evaluation_interval": 25,
    },
    "ppo": {
        "total_timesteps": 500_000,
        "hidden_sizes": [256, 256],
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "rollout_steps": 2_048,
        "minibatch_size": 256,
        "update_epochs": 10,
        "clip_ratio": 0.2,
        "value_clip_ratio": 0.2,
        "entropy_coefficient": 0.001,
        "value_coefficient": 0.5,
        "gradient_clip_norm": 0.5,
        "log_std_initial": -0.5,
        "log_std_minimum": -5.0,
        "log_std_maximum": 1.0,
        "target_kl": 0.03,
        "reward_scale": 0.01,
        "anneal_learning_rate": True,
        "print_every_updates": 5,
        "evaluation_interval_updates": 5,
        "checkpoint_every_updates": 25,
        "early_stopping": {
            "enabled": True,
            "minimum_timesteps": 200_000,
            "patience_evaluations": 5,
            "minimum_return": 475.0,
            "maximum_episode_steps": 120,
        },
    },

}
