"""Training entry point for the reinforcement-learning path-planning agents."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import math
import random
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch

from agents.dqn import DQNAgent, DQNConfig
from agents.ppo import PPOAgent, PPOConfig, RolloutBuffer
from core.environment import PathPlanning3DEnv, load_config
from core.visualization import plot_ppo_training_history, plot_training_history


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device



def build_dqn_agent(
    env: PathPlanning3DEnv,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> DQNAgent:
    settings = config["dqn"]
    agent_config = DQNConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_sizes=tuple(int(value) for value in settings["hidden_sizes"]),
        learning_rate=float(settings["learning_rate"]),
        gamma=float(settings["gamma"]),
        batch_size=int(settings["batch_size"]),
        replay_capacity=int(settings["replay_capacity"]),
        learning_starts=int(settings["learning_starts"]),
        train_frequency=int(settings["train_frequency"]),
        target_update_interval=int(settings["target_update_interval"]),
        epsilon_start=float(settings["epsilon_start"]),
        epsilon_end=float(settings["epsilon_end"]),
        epsilon_decay_steps=int(settings["epsilon_decay_steps"]),
        gradient_clip_norm=float(settings["gradient_clip_norm"]),
    )
    return DQNAgent(agent_config, device=device, seed=seed)



def build_ppo_agent(
    env: PathPlanning3DEnv,
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> PPOAgent:
    settings = config["ppo"]
    target_kl_value = settings.get("target_kl")
    agent_config = PPOConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_sizes=tuple(int(value) for value in settings["hidden_sizes"]),
        learning_rate=float(settings["learning_rate"]),
        gamma=float(settings["gamma"]),
        gae_lambda=float(settings["gae_lambda"]),
        clip_ratio=float(settings["clip_ratio"]),
        value_clip_ratio=float(settings["value_clip_ratio"]),
        update_epochs=int(settings["update_epochs"]),
        minibatch_size=int(settings["minibatch_size"]),
        entropy_coefficient=float(settings["entropy_coefficient"]),
        value_coefficient=float(settings["value_coefficient"]),
        gradient_clip_norm=float(settings["gradient_clip_norm"]),
        log_std_initial=float(settings["log_std_initial"]),
        log_std_minimum=float(settings["log_std_minimum"]),
        log_std_maximum=float(settings["log_std_maximum"]),
        target_kl=None if target_kl_value is None else float(target_kl_value),
        reward_scale=float(settings["reward_scale"]),
    )
    return PPOAgent(agent_config, device=device, seed=seed)



def train_dqn(args: argparse.Namespace) -> None:
    config = load_config()
    seed = int(config["project"]["seed"] if args.seed is None else args.seed)
    set_global_seed(seed)
    device = resolve_device(args.device)

    env = PathPlanning3DEnv("dqn")
    agent = build_dqn_agent(env, config, device, seed)
    dqn_settings = config["dqn"]
    episodes = int(args.episodes or dqn_settings["episodes"])
    print_every = int(args.print_every or dqn_settings["print_every"])
    checkpoint_every = int(dqn_settings["checkpoint_every"])
    evaluation_interval = int(dqn_settings.get("evaluation_interval", print_every))
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive.")
        env.max_steps = int(args.max_steps)
    step_limit = env.max_steps

    results_dir = resolve_results_dir(args.results_dir, "dqn")
    model_dir = resolve_model_dir(args.model_dir, "dqn")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    global_step = 0
    best_evaluation_score: tuple[int, float] | None = None

    print(
        f"DQN training started: episodes={episodes}, device={device}, "
        f"workspace={env.workspace_size:g}^3"
    )

    for episode in range(1, episodes + 1):
        observation, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        losses: list[float] = []
        outcome = "timeout"
        info: dict[str, Any] = {}

        for _ in range(step_limit):
            epsilon = agent.epsilon(global_step)
            action = agent.select_action(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(
                observation,
                action,
                reward,
                next_observation,
                terminated=terminated,
            )
            observation = next_observation
            episode_reward += reward
            global_step += 1

            if agent.can_update(global_step):
                losses.append(agent.update())
            agent.maybe_update_target(global_step)

            if terminated or truncated:
                outcome = str(info["outcome"])
                break

        training_length = int(env.step_count)
        evaluation_return = float("nan")
        evaluation_length = float("nan")
        evaluation_outcome = ""
        should_evaluate = episode == 1 or episode == episodes or episode % evaluation_interval == 0
        if should_evaluate:
            evaluation = evaluate_greedy_dqn(agent, env, seed=seed)
            evaluation_return = float(evaluation["return"])
            evaluation_length = int(evaluation["length"])
            evaluation_outcome = str(evaluation["outcome"])
            evaluation_score = (
                int(evaluation_outcome == "success"),
                evaluation_return,
            )
            if best_evaluation_score is None or evaluation_score > best_evaluation_score:
                best_evaluation_score = evaluation_score
                agent.save(model_dir / "best_model.pt", global_step)

        mean_loss = float(mean(losses)) if losses else float("nan")
        history.append(
            {
                "episode": episode,
                "global_step": global_step,
                "reward": episode_reward,
                "length": training_length,
                "epsilon": agent.epsilon(global_step),
                "mean_loss": mean_loss,
                "outcome": outcome,
                "distance_to_goal": float(info["distance_to_goal"]),
                "minimum_clearance": float(info["minimum_clearance"]),
                "evaluation_return": evaluation_return,
                "evaluation_length": evaluation_length,
                "evaluation_outcome": evaluation_outcome,
            }
        )

        if episode % print_every == 0 or episode == 1 or episode == episodes:
            recent = history[-min(print_every, len(history)) :]
            recent_reward = mean(float(row["reward"]) for row in recent)
            recent_success = mean(row["outcome"] == "success" for row in recent)
            message = (
                f"Episode {episode:4d} | step {global_step:7d} | "
                f"reward {recent_reward:8.2f} | success {recent_success:5.1%} | "
                f"epsilon {agent.epsilon(global_step):.3f}"
            )
            if evaluation_outcome:
                message += (
                    f" | eval {evaluation_outcome} "
                    f"({evaluation_return:.2f}, {int(evaluation_length)} steps)"
                )
            print(message)

        if episode % checkpoint_every == 0:
            agent.save(checkpoint_dir / f"dqn_episode_{episode:04d}.pt", global_step)

    agent.save(model_dir / "model.pt", global_step)
    write_training_log(results_dir / "training_log.csv", history)
    plot_training_history(
        [float(row["reward"]) for row in history],
        [float(row["mean_loss"]) for row in history],
        save_path=results_dir / "training_curve.png",
        show=False,
        evaluation_returns=[float(row["evaluation_return"]) for row in history],
    )
    print(f"Model artifacts saved to: {model_dir}")
    print(f"Training plots/logs saved to: {results_dir}")



def train_ppo(args: argparse.Namespace) -> None:
    config = load_config()
    seed = int(config["project"]["seed"] if args.seed is None else args.seed)
    set_global_seed(seed)
    device = resolve_device(args.device)

    env = PathPlanning3DEnv("ppo")
    evaluation_env = PathPlanning3DEnv("ppo")
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive.")
        env.max_steps = int(args.max_steps)
        evaluation_env.max_steps = int(args.max_steps)

    agent = build_ppo_agent(env, config, device, seed)
    settings = config["ppo"]
    total_timesteps = int(args.total_timesteps or settings["total_timesteps"])
    rollout_steps = int(args.rollout_steps or settings["rollout_steps"])
    print_every = int(args.print_every or settings["print_every_updates"])
    evaluation_interval = int(settings["evaluation_interval_updates"])
    checkpoint_every = int(settings["checkpoint_every_updates"])
    anneal_learning_rate = bool(settings.get("anneal_learning_rate", True))

    early_stopping = settings.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping.get("enabled", False))
    early_stopping_minimum_timesteps = int(
        early_stopping.get("minimum_timesteps", 0)
    )
    early_stopping_patience = int(
        early_stopping.get("patience_evaluations", 5)
    )
    early_stopping_minimum_return = float(
        early_stopping.get("minimum_return", 475.0)
    )
    early_stopping_maximum_steps = int(
        early_stopping.get("maximum_episode_steps", env.max_steps)
    )

    if total_timesteps <= 0 or rollout_steps <= 0:
        raise ValueError("PPO total_timesteps and rollout_steps must be positive.")

    results_dir = resolve_results_dir(args.results_dir, "ppo")
    model_dir = resolve_model_dir(args.model_dir, "ppo")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    observation, _ = env.reset(seed=seed)
    global_step = 0
    episode_index = 0
    episode_return = 0.0
    episode_length = 0
    completed_returns: deque[float] = deque(maxlen=20)
    completed_lengths: deque[int] = deque(maxlen=20)
    completed_outcomes: deque[str] = deque(maxlen=20)
    history: list[dict[str, Any]] = []
    best_evaluation_score: tuple[int, float] | None = None
    consecutive_qualified_evaluations = 0
    early_stop_triggered = False
    number_of_updates = math.ceil(total_timesteps / rollout_steps)
    base_learning_rate = agent.config.learning_rate

    print(
        f"PPO training started: timesteps={total_timesteps}, "
        f"rollout_steps={rollout_steps}, updates={number_of_updates}, "
        f"device={device}, workspace={env.workspace_size:g}^3"
    )

    for update in range(1, number_of_updates + 1):
        steps_this_update = min(rollout_steps, total_timesteps - global_step)
        rollout = RolloutBuffer(
            capacity=steps_this_update,
            state_dim=agent.config.state_dim,
            action_dim=agent.config.action_dim,
        )
        update_episode_returns: list[float] = []
        update_episode_lengths: list[int] = []
        update_episode_outcomes: list[str] = []

        if anneal_learning_rate:
            fraction_remaining = max(0.0, 1.0 - global_step / total_timesteps)
            current_learning_rate = max(1e-8, base_learning_rate * fraction_remaining)
            agent.set_learning_rate(current_learning_rate)
        else:
            current_learning_rate = base_learning_rate

        for _ in range(steps_this_update):
            action, raw_action, log_probability, value = agent.act(
                observation,
                deterministic=False,
            )
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_done = bool(terminated or truncated)
            next_value = 0.0 if terminated else agent.estimate_value(next_observation)

            rollout.add(
                state=observation,
                raw_action=raw_action,
                log_probability=log_probability,
                reward=reward * agent.config.reward_scale,
                value=value,
                next_value=next_value,
                terminated=terminated,
                episode_done=episode_done,
            )

            global_step += 1
            episode_return += reward
            episode_length += 1

            if episode_done:
                outcome = str(info["outcome"])
                update_episode_returns.append(float(episode_return))
                update_episode_lengths.append(int(episode_length))
                update_episode_outcomes.append(outcome)
                completed_returns.append(float(episode_return))
                completed_lengths.append(int(episode_length))
                completed_outcomes.append(outcome)
                episode_index += 1
                observation, _ = env.reset(seed=seed + episode_index)
                episode_return = 0.0
                episode_length = 0
            else:
                observation = next_observation

        update_metrics = agent.update(rollout)
        evaluation_return = float("nan")
        evaluation_length = float("nan")
        evaluation_outcome = ""
        should_evaluate = update == 1 or update == number_of_updates or update % evaluation_interval == 0
        if should_evaluate:
            evaluation = evaluate_deterministic_ppo(agent, evaluation_env, seed=seed)
            evaluation_return = float(evaluation["return"])
            evaluation_length = int(evaluation["length"])
            evaluation_outcome = str(evaluation["outcome"])
            evaluation_score = (
                int(evaluation_outcome == "success"),
                evaluation_return,
            )
            if best_evaluation_score is None or evaluation_score > best_evaluation_score:
                best_evaluation_score = evaluation_score
                agent.save(model_dir / "best_model.pt", global_step)

            if early_stopping_enabled:
                qualified_evaluation = (
                    global_step >= early_stopping_minimum_timesteps
                    and evaluation_outcome == "success"
                    and evaluation_return >= early_stopping_minimum_return
                    and evaluation_length <= early_stopping_maximum_steps
                )
                if qualified_evaluation:
                    consecutive_qualified_evaluations += 1
                else:
                    consecutive_qualified_evaluations = 0

                print(
                    "  Early stopping progress: "
                    f"{consecutive_qualified_evaluations}/"
                    f"{early_stopping_patience}"
                )

                if consecutive_qualified_evaluations >= early_stopping_patience:
                    early_stop_triggered = True

        recent_return = float(mean(completed_returns)) if completed_returns else float("nan")
        recent_length = float(mean(completed_lengths)) if completed_lengths else float("nan")
        recent_success = (
            float(mean(outcome == "success" for outcome in completed_outcomes))
            if completed_outcomes
            else float("nan")
        )
        rollout_return = float(mean(update_episode_returns)) if update_episode_returns else float("nan")
        rollout_success = (
            float(mean(outcome == "success" for outcome in update_episode_outcomes))
            if update_episode_outcomes
            else float("nan")
        )

        history.append(
            {
                "update": update,
                "global_step": global_step,
                "episodes_completed": episode_index,
                "rollout_mean_return": rollout_return,
                "recent_mean_return_20": recent_return,
                "rollout_success_rate": rollout_success,
                "recent_success_rate_20": recent_success,
                "recent_mean_length_20": recent_length,
                "policy_loss": update_metrics["policy_loss"],
                "value_loss": update_metrics["value_loss"],
                "entropy": update_metrics["entropy"],
                "approximate_kl": update_metrics["approximate_kl"],
                "clip_fraction": update_metrics["clip_fraction"],
                "gradient_norm": update_metrics["gradient_norm"],
                "epochs_completed": update_metrics["epochs_completed"],
                "learning_rate": current_learning_rate,
                "evaluation_return": evaluation_return,
                "evaluation_length": evaluation_length,
                "evaluation_outcome": evaluation_outcome,
            }
        )

        if update % print_every == 0 or update == 1 or update == number_of_updates:
            message = (
                f"Update {update:4d}/{number_of_updates} | step {global_step:7d} | "
                f"episodes {episode_index:5d} | return20 {recent_return:8.2f} | "
                f"success20 {recent_success:5.1%} | "
                f"pi_loss {float(update_metrics['policy_loss']):7.4f} | "
                f"v_loss {float(update_metrics['value_loss']):7.4f} | "
                f"KL {float(update_metrics['approximate_kl']):.5f}"
            )
            if evaluation_outcome:
                message += (
                    f" | eval {evaluation_outcome} "
                    f"({evaluation_return:.2f}, {int(evaluation_length)} steps)"
                )
            print(message)

        if update % checkpoint_every == 0:
            agent.save(checkpoint_dir / f"ppo_update_{update:04d}.pt", global_step)

        if early_stop_triggered:
            print(
                "PPO early stopping triggered: "
                f"{consecutive_qualified_evaluations} consecutive "
                "qualified evaluations."
            )
            print(f"Stopped at update {update}, global step {global_step}.")
            break

    agent.save(model_dir / "model.pt", global_step)
    write_training_log(results_dir / "training_log.csv", history)
    plot_ppo_training_history(history, save_path=results_dir / "training_curve.png", show=False)
    print(f"Model artifacts saved to: {model_dir}")
    print(f"Training plots/logs saved to: {results_dir}")



def evaluate_greedy_dqn(
    agent: DQNAgent,
    env: PathPlanning3DEnv,
    seed: int,
) -> dict[str, Any]:
    """Run one exploration-free benchmark episode during training."""
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(env.max_steps):
        action = agent.select_action(observation, epsilon=0.0, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return {
        "return": float(total_reward),
        "length": int(info["step_count"]),
        "outcome": str(info["outcome"]),
    }



def evaluate_deterministic_ppo(
    agent: PPOAgent,
    env: PathPlanning3DEnv,
    seed: int,
) -> dict[str, Any]:
    """Run one mean-action PPO benchmark episode during training."""
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(env.max_steps):
        action = agent.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return {
        "return": float(total_reward),
        "length": int(info["step_count"]),
        "outcome": str(info["outcome"]),
    }



def write_training_log(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        raise ValueError("Training history is empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)



def resolve_results_dir(results_dir: Path | None, algorithm: str) -> Path:
    if results_dir is not None:
        return results_dir
    return Path(__file__).resolve().parent / "results" / algorithm



def resolve_model_dir(model_dir: Path | None, algorithm: str) -> Path:
    if model_dir is not None:
        return model_dir
    return Path(__file__).resolve().parent / "models" / algorithm



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=("dqn", "ppo", "all"),
        default="all",
        help="Train one agent or both agents sequentially (default: all).",
    )
    parser.add_argument("--episodes", type=int, default=None, help="DQN episodes.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="PPO environment timesteps.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="PPO rollout size per update.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--print-every", type=int, default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory for logs/plots; for --algorithm all, this is the results root.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory for weights/checkpoints; for --algorithm all, this is the models root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Backward-compatible alias for --results-dir.",
    )
    return parser



def _args_for_algorithm(args: argparse.Namespace, algorithm: str) -> argparse.Namespace:
    """Return an independent argument namespace for one algorithm."""
    algorithm_args = copy.copy(args)
    algorithm_args.algorithm = algorithm
    if args.results_dir is not None and args.algorithm == "all":
        algorithm_args.results_dir = args.results_dir / algorithm
    if args.model_dir is not None and args.algorithm == "all":
        algorithm_args.model_dir = args.model_dir / algorithm
    return algorithm_args



def _release_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def main() -> None:
    args = build_parser().parse_args()
    if args.results_dir is None and args.output_dir is not None:
        args.results_dir = args.output_dir

    if args.algorithm == "dqn":
        train_dqn(args)
        return
    if args.algorithm == "ppo":
        train_ppo(args)
        return

    print("Sequential training started: DQN -> PPO")
    train_dqn(_args_for_algorithm(args, "dqn"))
    _release_device_memory()
    train_ppo(_args_for_algorithm(args, "ppo"))
    print("Sequential training completed for DQN and PPO.")


if __name__ == "__main__":
    main()
