from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.dqn import DQNAgent, DQNConfig, ReplayBuffer


def make_config() -> DQNConfig:
    return DQNConfig(
        state_dim=36,
        action_dim=6,
        hidden_sizes=(32, 32),
        learning_rate=1e-3,
        batch_size=8,
        replay_capacity=64,
        learning_starts=0,
        target_update_interval=5,
        epsilon_decay_steps=100,
    )


def test_replay_buffer_shapes() -> None:
    buffer = ReplayBuffer(capacity=16, state_dim=36)
    state = np.zeros(36, dtype=np.float32)
    for action in range(8):
        buffer.add(state + action, action % 6, float(action), state + action + 1, False)

    batch = buffer.sample(8, np.random.default_rng(0), torch.device("cpu"))
    states, actions, rewards, next_states, terminated = batch
    assert states.shape == (8, 36)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 36)
    assert terminated.shape == (8,)


def test_dqn_update_changes_online_network() -> None:
    agent = DQNAgent(make_config(), seed=3)
    rng = np.random.default_rng(4)
    for _ in range(32):
        state = rng.normal(size=36).astype(np.float32)
        next_state = rng.normal(size=36).astype(np.float32)
        agent.store_transition(
            state,
            int(rng.integers(6)),
            float(rng.normal()),
            next_state,
            bool(rng.integers(2)),
        )

    before = [parameter.detach().clone() for parameter in agent.online_network.parameters()]
    loss = agent.update()
    after = list(agent.online_network.parameters())

    assert np.isfinite(loss)
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_epsilon_schedule_and_target_update() -> None:
    agent = DQNAgent(make_config(), seed=1)
    assert agent.epsilon(0) == 1.0
    assert agent.epsilon(100) == pytest.approx(0.05)
    assert agent.epsilon(1000) == pytest.approx(0.05)
    assert not agent.maybe_update_target(4)
    assert agent.maybe_update_target(5)


def test_checkpoint_roundtrip(tmp_path) -> None:
    agent = DQNAgent(make_config(), seed=2)
    checkpoint = tmp_path / "dqn.pt"
    agent.save(checkpoint, global_step=123)

    loaded, global_step = DQNAgent.load(checkpoint, seed=2)
    assert global_step == 123
    for expected, actual in zip(
        agent.online_network.parameters(),
        loaded.online_network.parameters(),
    ):
        assert torch.equal(expected, actual)
