from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.ppo import PPOAgent, PPOConfig, RolloutBuffer


def make_config(**overrides) -> PPOConfig:
    values = {
        "state_dim": 36,
        "action_dim": 3,
        "hidden_sizes": (32, 32),
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_clip_ratio": 0.2,
        "update_epochs": 2,
        "minibatch_size": 8,
        "entropy_coefficient": 1e-3,
        "value_coefficient": 0.5,
        "gradient_clip_norm": 0.5,
        "log_std_initial": -0.5,
        "log_std_minimum": -5.0,
        "log_std_maximum": 1.0,
        "target_kl": None,
        "reward_scale": 0.01,
    }
    values.update(overrides)
    return PPOConfig(**values)


def fill_rollout(agent: PPOAgent, size: int = 32) -> RolloutBuffer:
    rng = np.random.default_rng(7)
    rollout = RolloutBuffer(size, state_dim=36, action_dim=3)
    for index in range(size):
        state = rng.normal(size=36).astype(np.float32)
        action, raw_action, log_probability, value = agent.act(state)
        next_state = rng.normal(size=36).astype(np.float32)
        next_value = agent.estimate_value(next_state)
        episode_done = (index + 1) % 8 == 0
        terminated = episode_done and index % 16 == 15
        rollout.add(
            state=state,
            raw_action=raw_action,
            log_probability=log_probability,
            reward=float(rng.normal(scale=0.1)),
            value=value,
            next_value=0.0 if terminated else next_value,
            terminated=terminated,
            episode_done=episode_done,
        )
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
    return rollout


def test_squashed_gaussian_actions_are_bounded_and_finite() -> None:
    agent = PPOAgent(make_config(), seed=3)
    state = np.zeros(36, dtype=np.float32)

    for _ in range(64):
        action, raw_action, log_probability, value = agent.act(state)
        assert action.shape == (3,)
        assert raw_action.shape == (3,)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)
        assert np.isfinite(log_probability)
        assert np.isfinite(value)

    deterministic = agent.predict(state, deterministic=True)
    assert np.all(deterministic >= -1.0)
    assert np.all(deterministic <= 1.0)


def test_timeout_gae_bootstraps_but_does_not_cross_episode_boundary() -> None:
    rollout = RolloutBuffer(capacity=2, state_dim=1, action_dim=1)
    rollout.add(
        state=np.array([0.0], dtype=np.float32),
        raw_action=np.array([0.0], dtype=np.float32),
        log_probability=0.0,
        reward=1.0,
        value=0.5,
        next_value=2.0,
        terminated=False,
        episode_done=True,
    )
    rollout.add(
        state=np.array([0.0], dtype=np.float32),
        raw_action=np.array([0.0], dtype=np.float32),
        log_probability=0.0,
        reward=10.0,
        value=0.0,
        next_value=0.0,
        terminated=True,
        episode_done=True,
    )

    advantages, returns = rollout.compute_advantages(gamma=0.9, gae_lambda=0.95)
    assert advantages[0] == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)
    assert returns[0] == pytest.approx(1.0 + 0.9 * 2.0)
    assert advantages[1] == pytest.approx(10.0)


def test_ppo_update_changes_network_and_returns_finite_metrics() -> None:
    agent = PPOAgent(make_config(), seed=4)
    rollout = fill_rollout(agent)
    before = [parameter.detach().clone() for parameter in agent.network.parameters()]

    metrics = agent.update(rollout)
    after = list(agent.network.parameters())

    assert any(not torch.equal(old, new) for old, new in zip(before, after))
    for key in (
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "gradient_norm",
    ):
        assert np.isfinite(metrics[key])
    assert 1 <= metrics["epochs_completed"] <= agent.config.update_epochs


def test_ppo_checkpoint_roundtrip(tmp_path) -> None:
    agent = PPOAgent(make_config(), seed=5)
    checkpoint = tmp_path / "ppo.pt"
    agent.save(checkpoint, global_step=456)

    loaded, global_step = PPOAgent.load(checkpoint, seed=5)
    assert global_step == 456
    for expected, actual in zip(
        agent.network.parameters(),
        loaded.network.parameters(),
    ):
        assert torch.equal(expected, actual)

    state = np.zeros(36, dtype=np.float32)
    assert np.allclose(
        agent.predict(state, deterministic=True),
        loaded.predict(state, deterministic=True),
    )
