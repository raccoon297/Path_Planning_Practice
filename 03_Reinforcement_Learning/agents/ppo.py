"""Proximal Policy Optimization for continuous 3D path planning.

The policy uses a tanh-squashed Gaussian distribution so every action lies in
``[-1, 1]^3`` without post-sampling clipping. Generalized Advantage Estimation
(GAE), the clipped PPO objective, clipped value loss, minibatch updates, and
gradient clipping are implemented directly in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

Array = np.ndarray
_EPS = 1e-6
_LOG_TWO = float(np.log(2.0))


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters and network dimensions for PPO."""

    state_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 256
    entropy_coefficient: float = 1e-3
    value_coefficient: float = 0.5
    gradient_clip_norm: float = 0.5
    log_std_initial: float = -0.5
    log_std_minimum: float = -5.0
    log_std_maximum: float = 1.0
    target_kl: float | None = 0.03
    reward_scale: float = 0.01

    def __post_init__(self) -> None:
        if self.state_dim <= 0 or self.action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive.")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1].")
        if self.clip_ratio <= 0 or self.value_clip_ratio <= 0:
            raise ValueError("PPO clipping ratios must be positive.")
        if self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("update_epochs and minibatch_size must be positive.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if self.log_std_minimum >= self.log_std_maximum:
            raise ValueError("log_std_minimum must be smaller than log_std_maximum.")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive.")


def _orthogonal_initialize(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)


def _build_mlp(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    output_dim: int,
    output_gain: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_sizes:
        layer = nn.Linear(previous, hidden)
        _orthogonal_initialize(layer, gain=np.sqrt(2.0))
        layers.extend((layer, nn.Tanh()))
        previous = hidden
    output = nn.Linear(previous, output_dim)
    _orthogonal_initialize(output, gain=output_gain)
    layers.append(output)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs with a global Gaussian log standard deviation."""

    def __init__(self, config: PPOConfig) -> None:
        super().__init__()
        self.config = config
        self.actor = _build_mlp(
            config.state_dim,
            config.hidden_sizes,
            config.action_dim,
            output_gain=0.01,
        )
        self.critic = _build_mlp(
            config.state_dim,
            config.hidden_sizes,
            1,
            output_gain=1.0,
        )
        self.log_std = nn.Parameter(
            torch.full((config.action_dim,), float(config.log_std_initial))
        )

    def distribution(self, states: torch.Tensor) -> Normal:
        means = self.actor(states)
        log_std = torch.clamp(
            self.log_std,
            self.config.log_std_minimum,
            self.config.log_std_maximum,
        )
        standard_deviation = torch.exp(log_std).expand_as(means)
        return Normal(means, standard_deviation)

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)

    @staticmethod
    def _squashed_log_probability(
        distribution: Normal,
        raw_actions: torch.Tensor,
    ) -> torch.Tensor:
        # Stable form of log(1 - tanh(x)^2):
        # 2 * (log(2) - x - softplus(-2x)).
        log_jacobian = 2.0 * (
            _LOG_TWO - raw_actions - F.softplus(-2.0 * raw_actions)
        )
        return (distribution.log_prob(raw_actions) - log_jacobian).sum(dim=-1)

    def sample_action(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probabilities = self._squashed_log_probability(distribution, raw_actions)
        values = self.value(states)
        return actions, raw_actions, log_probabilities, values

    def evaluate_raw_actions(
        self,
        states: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        log_probabilities = self._squashed_log_probability(distribution, raw_actions)
        # The exact entropy of a tanh-transformed Gaussian has no simple closed
        # form. Base Gaussian entropy is used as the exploration diagnostic and
        # entropy bonus, as is common in compact PPO implementations.
        entropy = distribution.entropy().sum(dim=-1)
        values = self.value(states)
        return log_probabilities, entropy, values


class RolloutBuffer:
    """Fixed-capacity on-policy rollout storage with timeout-aware GAE."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.raw_actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probabilities = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.episode_done = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: Array,
        raw_action: Array,
        log_probability: float,
        reward: float,
        value: float,
        next_value: float,
        terminated: bool,
        episode_done: bool,
    ) -> None:
        if self.size >= self.capacity:
            raise RuntimeError("RolloutBuffer is full.")
        state_array = np.asarray(state, dtype=np.float32)
        raw_action_array = np.asarray(raw_action, dtype=np.float32)
        if state_array.shape != (self.state_dim,):
            raise ValueError(f"state must have shape ({self.state_dim},).")
        if raw_action_array.shape != (self.action_dim,):
            raise ValueError(f"raw_action must have shape ({self.action_dim},).")

        index = self.size
        self.states[index] = state_array
        self.raw_actions[index] = raw_action_array
        self.log_probabilities[index] = float(log_probability)
        self.rewards[index] = float(reward)
        self.values[index] = float(value)
        self.next_values[index] = float(next_value)
        self.terminated[index] = float(terminated)
        self.episode_done[index] = float(episode_done)
        self.size += 1

    def compute_advantages(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[Array, Array]:
        if self.size == 0:
            raise ValueError("Cannot compute advantages from an empty rollout.")

        rewards = self.rewards[: self.size]
        values = self.values[: self.size]
        next_values = self.next_values[: self.size]
        terminated = self.terminated[: self.size]
        episode_done = self.episode_done[: self.size]

        deltas = rewards + gamma * next_values * (1.0 - terminated) - values
        advantages = np.zeros(self.size, dtype=np.float32)
        gae = 0.0
        for index in range(self.size - 1, -1, -1):
            continuation = 1.0 - episode_done[index]
            gae = float(deltas[index]) + gamma * gae_lambda * continuation * gae
            advantages[index] = gae
        returns = advantages + values
        return advantages, returns.astype(np.float32)

    def minibatches(
        self,
        advantages: Array,
        returns: Array,
        minibatch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> Iterator[dict[str, torch.Tensor]]:
        if len(advantages) != self.size or len(returns) != self.size:
            raise ValueError("advantages and returns must match rollout size.")
        indices = rng.permutation(self.size)
        for start in range(0, self.size, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]
            yield {
                "states": torch.as_tensor(
                    self.states[batch_indices], dtype=torch.float32, device=device
                ),
                "raw_actions": torch.as_tensor(
                    self.raw_actions[batch_indices], dtype=torch.float32, device=device
                ),
                "old_log_probabilities": torch.as_tensor(
                    self.log_probabilities[batch_indices],
                    dtype=torch.float32,
                    device=device,
                ),
                "old_values": torch.as_tensor(
                    self.values[batch_indices], dtype=torch.float32, device=device
                ),
                "advantages": torch.as_tensor(
                    advantages[batch_indices], dtype=torch.float32, device=device
                ),
                "returns": torch.as_tensor(
                    returns[batch_indices], dtype=torch.float32, device=device
                ),
            }


class PPOAgent:
    """PPO actor-critic agent with checkpoint support."""

    def __init__(
        self,
        config: PPOConfig,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.network = ActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self.rng = np.random.default_rng(seed)

    def act(
        self,
        state: Array,
        deterministic: bool = False,
    ) -> tuple[Array, Array, float, float]:
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            action, raw_action, log_probability, value = self.network.sample_action(
                state_tensor,
                deterministic=deterministic,
            )
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            raw_action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_probability.item()),
            float(value.item()),
        )

    def predict(self, state: Array, deterministic: bool = True) -> Array:
        action, _, _, _ = self.act(state, deterministic=deterministic)
        return action

    def estimate_value(self, state: Array) -> float:
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            value = self.network.value(state_tensor)
        return float(value.item())

    def set_learning_rate(self, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = float(learning_rate)

    def update(self, rollout: RolloutBuffer) -> dict[str, float | int]:
        advantages, returns = rollout.compute_advantages(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        advantage_mean = float(np.mean(advantages))
        advantage_std = float(np.std(advantages))
        advantages = (advantages - advantage_mean) / max(advantage_std, _EPS)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approximate_kls: list[float] = []
        clip_fractions: list[float] = []
        gradient_norms: list[float] = []
        epochs_completed = 0

        for epoch in range(self.config.update_epochs):
            epoch_kls: list[float] = []
            for batch in rollout.minibatches(
                advantages,
                returns,
                self.config.minibatch_size,
                self.rng,
                self.device,
            ):
                new_log_probabilities, entropy, new_values = (
                    self.network.evaluate_raw_actions(
                        batch["states"],
                        batch["raw_actions"],
                    )
                )
                log_ratio = new_log_probabilities - batch["old_log_probabilities"]
                ratio = torch.exp(log_ratio)

                surrogate_unclipped = ratio * batch["advantages"]
                surrogate_clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * batch["advantages"]
                policy_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()

                clipped_values = batch["old_values"] + torch.clamp(
                    new_values - batch["old_values"],
                    -self.config.value_clip_ratio,
                    self.config.value_clip_ratio,
                )
                value_loss_unclipped = (new_values - batch["returns"]) ** 2
                value_loss_clipped = (clipped_values - batch["returns"]) ** 2
                value_loss = 0.5 * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()
                entropy_mean = entropy.mean()

                total_loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.gradient_clip_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                        .float()
                        .mean()
                    )

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                kl_value = float(approximate_kl.item())
                approximate_kls.append(kl_value)
                epoch_kls.append(kl_value)
                clip_fractions.append(float(clip_fraction.item()))
                gradient_norms.append(float(gradient_norm.item()))

            epochs_completed = epoch + 1
            if (
                self.config.target_kl is not None
                and epoch_kls
                and float(np.mean(epoch_kls)) > 1.5 * self.config.target_kl
            ):
                break

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approximate_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "gradient_norm": float(np.mean(gradient_norms)),
            "advantage_mean": advantage_mean,
            "advantage_std": advantage_std,
            "epochs_completed": int(epochs_completed),
        }

    def save(self, path: str | Path, global_step: int) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": int(global_step),
            },
            output,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> tuple["PPOAgent", int]:
        checkpoint = torch.load(path, map_location=torch.device(device), weights_only=False)
        config_data = dict(checkpoint["config"])
        config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
        config = PPOConfig(**config_data)
        agent = cls(config, device=device, seed=seed)
        agent.network.load_state_dict(checkpoint["network_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return agent, int(checkpoint.get("global_step", 0))


__all__ = [
    "ActorCritic",
    "PPOAgent",
    "PPOConfig",
    "RolloutBuffer",
]
