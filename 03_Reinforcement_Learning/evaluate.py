"""Evaluate DQN and PPO models and export individual and comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from core.environment import PathPlanning3DEnv, load_config
from core.visualization import (
    animate_trajectory,
    animate_trajectory_comparison,
    plot_trajectory,
    plot_trajectory_comparison,
)

Array = np.ndarray
_EPS = 1e-9
_COMPARISON_METRICS = (
    "success",
    "episode_return",
    "episode_steps",
    "final_distance_to_goal",
    "path_length",
    "direct_displacement",
    "path_efficiency",
    "minimum_clearance",
    "total_turning_angle_degrees",
    "mean_turning_angle_degrees",
    "maximum_turning_angle_degrees",
    "trajectory_roughness",
    "mean_step_roughness",
    "control_variation",
    "mean_control_change",
)


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



def calculate_trajectory_metrics(
    env: PathPlanning3DEnv,
    trajectory: Sequence[Sequence[float]],
    controls: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    """Calculate geometry and smoothness metrics for one trajectory."""
    path = np.asarray(trajectory, dtype=np.float64)
    control_array = np.asarray(controls, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 1:
        raise ValueError("trajectory must have shape (N, 3), N >= 1.")
    if control_array.size == 0:
        control_array = np.empty((0, 3), dtype=np.float64)
    if control_array.ndim != 2 or control_array.shape[1] != 3:
        raise ValueError("controls must have shape (M, 3).")

    displacements = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(displacements, axis=1)
    path_length = float(np.sum(segment_lengths))
    direct_distance = float(np.linalg.norm(env.goal - path[0]))
    path_efficiency = direct_distance / path_length if path_length > _EPS else 0.0
    average_step_length = float(np.mean(segment_lengths)) if len(segment_lengths) else 0.0
    maximum_step_length = float(np.max(segment_lengths)) if len(segment_lengths) else 0.0

    turning_angles = np.empty(0, dtype=np.float64)
    valid_displacements = displacements[segment_lengths > _EPS]
    if len(valid_displacements) >= 2:
        first = valid_displacements[:-1]
        second = valid_displacements[1:]
        denominators = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        cosines = np.sum(first * second, axis=1) / np.maximum(denominators, _EPS)
        turning_angles = np.arccos(np.clip(cosines, -1.0, 1.0))

    total_turning_angle = float(np.sum(turning_angles))
    mean_turning_angle = float(np.mean(turning_angles)) if len(turning_angles) else 0.0
    maximum_turning_angle = float(np.max(turning_angles)) if len(turning_angles) else 0.0

    roughness = 0.0
    mean_step_roughness = 0.0
    if len(displacements) >= 2:
        second_difference = np.diff(displacements, axis=0)
        squared_changes = np.linalg.norm(second_difference, axis=1) ** 2
        roughness = float(np.sum(squared_changes))
        mean_step_roughness = float(np.mean(squared_changes))

    direction_changes = 0
    control_variation = 0.0
    mean_control_change = 0.0
    if len(control_array) >= 2:
        control_differences = np.diff(control_array, axis=0)
        change_magnitudes = np.linalg.norm(control_differences, axis=1)
        direction_changes = int(np.count_nonzero(change_magnitudes > _EPS))
        control_variation = float(np.sum(change_magnitudes))
        mean_control_change = float(np.mean(change_magnitudes))

    clearances = [env.minimum_clearance(point) for point in path]
    minimum_clearance = float(min(clearances)) if clearances else float("nan")

    return {
        "path_length": path_length,
        "direct_displacement": direct_distance,
        "path_efficiency": path_efficiency,
        "average_step_length": average_step_length,
        "maximum_step_length": maximum_step_length,
        "minimum_clearance": minimum_clearance,
        "direction_changes": direction_changes,
        "total_turning_angle_degrees": float(np.degrees(total_turning_angle)),
        "mean_turning_angle_degrees": float(np.degrees(mean_turning_angle)),
        "maximum_turning_angle_degrees": float(np.degrees(maximum_turning_angle)),
        "trajectory_roughness": roughness,
        "mean_step_roughness": mean_step_roughness,
        "control_variation": control_variation,
        "mean_control_change": mean_control_change,
    }



def run_dqn_episode(
    env: PathPlanning3DEnv,
    agent: DQNAgent,
    seed: int,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    actions: list[int] = []

    for _ in range(env.max_steps):
        action = agent.select_action(observation, epsilon=0.0, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        total_reward += reward
        if terminated or truncated:
            break

    return build_episode_result(
        algorithm="DQN",
        seed=seed,
        env=env,
        total_reward=total_reward,
        info=info,
        actions=np.asarray(actions, dtype=np.int64),
    )



def run_ppo_episode(
    env: PathPlanning3DEnv,
    agent: PPOAgent,
    seed: int,
) -> dict[str, Any]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    actions: list[Array] = []

    for _ in range(env.max_steps):
        action = agent.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(np.asarray(action, dtype=np.float64))
        total_reward += reward
        if terminated or truncated:
            break

    action_array = np.asarray(actions, dtype=np.float64) if actions else np.empty((0, 3), dtype=np.float64)
    return build_episode_result(
        algorithm="PPO",
        seed=seed,
        env=env,
        total_reward=total_reward,
        info=info,
        actions=action_array,
    )



def build_episode_result(
    algorithm: str,
    seed: int,
    env: PathPlanning3DEnv,
    total_reward: float,
    info: dict[str, Any],
    actions: Array,
) -> dict[str, Any]:
    trajectory = np.asarray(env.trajectory, dtype=np.float64)
    controls = np.asarray(env.controls, dtype=np.float64)
    metrics: dict[str, Any] = {
        "algorithm": algorithm,
        "seed": int(seed),
        "outcome": str(info["outcome"]),
        "success": bool(info["outcome"] == "success"),
        "episode_return": float(total_reward),
        "episode_steps": int(info["step_count"]),
        "final_position": [float(value) for value in info["position"]],
        "final_distance_to_goal": float(info["distance_to_goal"]),
        "final_speed": float(info["speed"]),
    }
    metrics.update(calculate_trajectory_metrics(env, trajectory, controls))
    return {
        "metrics": metrics,
        "trajectory": trajectory,
        "controls": controls,
        "actions": actions,
    }



def write_trajectory(path: Path, trajectory: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "x", "y", "z"))
        for index, point in enumerate(trajectory):
            writer.writerow((index, *[float(value) for value in point]))



def write_discrete_actions(path: Path, actions: Array, controls: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "action", "control_x", "control_y", "control_z"))
        for index, (action, control) in enumerate(zip(actions, controls), start=1):
            writer.writerow((index, int(action), *[float(value) for value in control]))



def write_continuous_actions(path: Path, actions: Array, controls: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            (
                "step",
                "action_x",
                "action_y",
                "action_z",
                "executed_control_x",
                "executed_control_y",
                "executed_control_z",
            )
        )
        for index, (action, control) in enumerate(zip(actions, controls), start=1):
            writer.writerow((index, *[float(value) for value in action], *[float(value) for value in control]))



def write_comparison_table(path: Path, results: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "DQN", "PPO"))
        for metric in _COMPARISON_METRICS:
            writer.writerow((metric, results["dqn"]["metrics"].get(metric), results["ppo"]["metrics"].get(metric)))



def build_comparison_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    dqn = results["dqn"]["metrics"]
    ppo = results["ppo"]["metrics"]

    def relative_reduction(metric: str) -> float | None:
        baseline = float(dqn[metric])
        if abs(baseline) <= _EPS:
            return None
        return float((baseline - float(ppo[metric])) / baseline)

    return {
        "selection_rule": "Highest deterministic evaluation return among successful checkpoints.",
        "dqn": {metric: dqn.get(metric) for metric in _COMPARISON_METRICS},
        "ppo": {metric: ppo.get(metric) for metric in _COMPARISON_METRICS},
        "ppo_relative_to_dqn": {
            "path_length_reduction": relative_reduction("path_length"),
            "episode_step_reduction": relative_reduction("episode_steps"),
            "turning_angle_reduction": relative_reduction("total_turning_angle_degrees"),
            "trajectory_roughness_reduction": relative_reduction("trajectory_roughness"),
            "path_efficiency_difference": float(ppo["path_efficiency"] - dqn["path_efficiency"]),
            "minimum_clearance_difference": float(ppo["minimum_clearance"] - dqn["minimum_clearance"]),
        },
        "notes": {
            "direction_changes": (
                "Excluded from the common comparison because discrete DQN action "
                "changes and continuous PPO control changes have different meanings."
            ),
            "safety_margin": (
                "Minimum clearance is measured, while the configured safety margin "
                "is a soft reward penalty rather than a hard constraint."
            ),
        },
    }



def resolve_model_path(
    project_dir: Path,
    algorithm: str,
    shared_model: Path | None,
    algorithm_model: Path | None,
) -> Path:
    if algorithm_model is not None:
        return algorithm_model
    if shared_model is not None:
        return shared_model
    return project_dir / "models" / algorithm / "best_model.pt"



def evaluate_algorithm(
    algorithm: str,
    model_path: Path,
    output_dir: Path,
    seed: int,
    device: torch.device,
    save_gif: bool,
    show: bool,
) -> tuple[PathPlanning3DEnv, dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run train.py first or select a model."
        )

    env = PathPlanning3DEnv(algorithm)
    if algorithm == "dqn":
        agent, global_step = DQNAgent.load(model_path, device=device, seed=seed)
        result = run_dqn_episode(env, agent, seed=seed)
        label = "DQN"
    elif algorithm == "ppo":
        agent, global_step = PPOAgent.load(model_path, device=device, seed=seed)
        result = run_ppo_episode(env, agent, seed=seed)
        label = "PPO"
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    metrics["model_path"] = str(model_path)
    metrics["checkpoint_global_step"] = int(global_step)

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    write_trajectory(output_dir / "trajectory.csv", result["trajectory"])
    if algorithm == "dqn":
        write_discrete_actions(output_dir / "actions.csv", result["actions"], result["controls"])
    else:
        write_continuous_actions(output_dir / "actions.csv", result["actions"], result["controls"])

    plot_trajectory(env, result["trajectory"], label=label, save_path=output_dir / "trajectory.png", show=show)
    if save_gif:
        animate_trajectory(env, result["trajectory"], label=label, save_path=output_dir / "navigation.gif", show=show)

    print(f"{label} evaluation completed")
    print(f"  outcome             : {metrics['outcome']}")
    print(f"  return              : {metrics['episode_return']:.2f}")
    print(f"  steps               : {metrics['episode_steps']}")
    print(f"  final goal distance : {metrics['final_distance_to_goal']:.2f}")
    print(f"  path length         : {metrics['path_length']:.2f}")
    print(f"  path efficiency     : {metrics['path_efficiency']:.3f}")
    print(f"  minimum clearance   : {metrics['minimum_clearance']:.2f}")
    print(f"  total turning angle : {metrics['total_turning_angle_degrees']:.2f} deg")
    print(f"  roughness           : {metrics['trajectory_roughness']:.2f}")
    print(f"Artifacts saved to: {output_dir}")
    return env, result



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=("dqn", "ppo", "all"),
        default="all",
        help="Evaluate one agent or both agents (default: all).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Model path for single-algorithm evaluation only.",
    )
    parser.add_argument("--dqn-model", type=Path, default=None)
    parser.add_argument("--ppo-model", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for one algorithm, or the result root when --algorithm all is used.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF generation. GIFs are generated by default.",
    )
    parser.add_argument("--show", action="store_true")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    if args.algorithm == "all" and args.model is not None:
        raise ValueError(
            "--model is ambiguous with --algorithm all. Use --dqn-model and --ppo-model instead."
        )

    config = load_config()
    seed = int(config["project"]["seed"] if args.seed is None else args.seed)
    set_global_seed(seed)
    device = resolve_device(args.device)
    project_dir = Path(__file__).resolve().parent
    save_gif = not args.no_gif

    algorithms = ("dqn", "ppo") if args.algorithm == "all" else (args.algorithm,)
    result_root = args.output_dir or project_dir / "results"
    results: dict[str, dict[str, Any]] = {}
    environments: dict[str, PathPlanning3DEnv] = {}

    for algorithm in algorithms:
        specific_model = args.dqn_model if algorithm == "dqn" else args.ppo_model
        model_path = resolve_model_path(
            project_dir,
            algorithm,
            args.model if args.algorithm != "all" else None,
            specific_model,
        )
        output_dir = result_root / algorithm if args.algorithm == "all" else args.output_dir or project_dir / "results" / algorithm
        env, result = evaluate_algorithm(
            algorithm=algorithm,
            model_path=model_path,
            output_dir=output_dir,
            seed=seed,
            device=device,
            save_gif=save_gif,
            show=args.show,
        )
        environments[algorithm] = env
        results[algorithm] = result

    if args.algorithm == "all":
        comparison_dir = result_root / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        trajectories = {"DQN": results["dqn"]["trajectory"], "PPO": results["ppo"]["trajectory"]}
        plot_trajectory_comparison(
            environments["dqn"],
            trajectories,
            save_path=comparison_dir / "trajectory_comparison.png",
            show=args.show,
        )
        if save_gif:
            animate_trajectory_comparison(
                environments["dqn"],
                trajectories,
                save_path=comparison_dir / "navigation_comparison.gif",
                show=args.show,
            )
        write_comparison_table(comparison_dir / "comparison_table.csv", results)
        with (comparison_dir / "comparison_summary.json").open("w", encoding="utf-8") as file:
            json.dump(build_comparison_summary(results), file, indent=2, ensure_ascii=False)
        print(f"Comparison artifacts saved to: {comparison_dir}")


if __name__ == "__main__":
    main()
