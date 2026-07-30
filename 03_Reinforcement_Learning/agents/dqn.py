"""Deep Q-Network agent for the six-direction 3D path-planning task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

Array = np.ndarray


@dataclass(frozen=True)
class DQNConfig:
    state_dim: int
    action_dim: int = 6
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 5e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 2_000
    train_frequency: int = 1
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    gradient_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive.")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")
        if self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size.")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
        if self.epsilon_decay_steps <= 0:
            raise ValueError("epsilon_decay_steps must be positive.")


class QNetwork(nn.Module):
    """MLP that estimates one action value for each discrete action."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_sizes:
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ReplayBuffer:
    """Fixed-size replay buffer backed by NumPy arrays."""

    def __init__(self, capacity: int, state_dim: int) -> None:
        if capacity <= 0 or state_dim <= 0:
            raise ValueError("capacity and state_dim must be positive.")
        self.capacity = int(capacity)
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: Array,
        action: int,
        reward: float,
        next_state: Array,
        terminated: bool,
    ) -> None:
        index = self.position
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.terminated[index] = float(terminated)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough transitions to sample the requested batch.")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.terminated[indices], device=device),
        )


class DQNAgent:
    """DQN with experience replay and a periodically synchronized target network."""

    def __init__(
        self,
        config: DQNConfig,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        self.online_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_sizes,
        ).to(self.device)
        self.target_network = QNetwork(
            config.state_dim,
            config.action_dim,
            config.hidden_sizes,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=config.learning_rate,
        )
        self.replay_buffer = ReplayBuffer(config.replay_capacity, config.state_dim)
        self.hard_update_target()
        self.target_network.eval()
        self.update_count = 0

    def epsilon(self, global_step: int) -> float:
        fraction = min(max(global_step, 0) / self.config.epsilon_decay_steps, 1.0)
        return float(
            self.config.epsilon_start
            + fraction * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def select_action(
        self,
        state: Array,
        epsilon: float,
        deterministic: bool = False,
    ) -> int:
        if not deterministic and self.rng.random() < epsilon:
            return int(self.rng.integers(self.config.action_dim))

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action_values = self.online_network(state_tensor)
        return int(torch.argmax(action_values, dim=1).item())

    def store_transition(
        self,
        state: Array,
        action: int,
        reward: float,
        next_state: Array,
        terminated: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, terminated)

    def can_update(self, global_step: int) -> bool:
        return (
            global_step >= self.config.learning_starts
            and global_step % self.config.train_frequency == 0
            and len(self.replay_buffer) >= self.config.batch_size
        )

    def update(self) -> float:
        states, actions, rewards, next_states, terminated = self.replay_buffer.sample(
            self.config.batch_size,
            self.rng,
            self.device,
        )

        action_values = self.online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_action_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - terminated) * next_action_values

        loss = F.smooth_l1_loss(action_values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_network.parameters(),
            max_norm=self.config.gradient_clip_norm,
        )
        self.optimizer.step()
        self.update_count += 1
        return float(loss.item())

    def hard_update_target(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def maybe_update_target(self, global_step: int) -> bool:
        if global_step > 0 and global_step % self.config.target_update_interval == 0:
            self.hard_update_target()
            return True
        return False

    def save(self, path: str | Path, global_step: int = 0) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": int(global_step),
                "update_count": int(self.update_count),
            },
            output,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        seed: int = 0,
        load_optimizer: bool = False,
    ) -> tuple["DQNAgent", int]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config_data = dict(checkpoint["config"])
        config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
        agent = cls(DQNConfig(**config_data), device=device, seed=seed)
        agent.online_network.load_state_dict(checkpoint["online_network"])
        agent.target_network.load_state_dict(checkpoint["target_network"])
        if load_optimizer and "optimizer" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.update_count = int(checkpoint.get("update_count", 0))
        return agent, int(checkpoint.get("global_step", 0))


__all__ = ["DQNAgent", "DQNConfig", "QNetwork", "ReplayBuffer"]
