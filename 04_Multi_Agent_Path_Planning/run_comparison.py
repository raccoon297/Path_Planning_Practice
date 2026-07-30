"""Validate, compare, and visualize the four multi-agent planning results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config.scenario import DEFAULT_SCENARIO
from utils.path_utils import JointPlan
from utils.reporting import scenario_as_dict
from utils.visualization import plot_scenario

ALGORITHM_ORDER = ("ACO", "GA", "GWO", "PSO")
RESULT_FILES = {
    "ACO": Path("results/aco/aco_result.json"),
    "GA": Path("results/ga/ga_result.json"),
    "GWO": Path("results/gwo/gwo_result.json"),
    "PSO": Path("results/pso/pso_result.json"),
}
RUN_COMMANDS = {
    "ACO": "run_aco.py",
    "GA": "run_ga.py",
    "GWO": "run_gwo.py",
    "PSO": "run_pso.py",
}
REQUIRED_METRICS = {
    "success",
    "total_path_length",
    "makespan",
    "sum_start_delay",
    "minimum_obstacle_clearance",
    "minimum_boundary_clearance",
    "minimum_inter_agent_distance",
    "obstacle_collision_count",
    "obstacle_safety_violation_count",
    "boundary_safety_violation_count",
    "inter_agent_collision_episodes",
    "inter_agent_separation_violation_episodes",
    "total_smoothness",
    "total_backtracking",
    "waypoint_spacing_imbalance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Rerun all algorithms with --skip-gifs before comparing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed used with --rerun.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/comparison"),
        help="Directory for comparison artifacts.",
    )
    return parser.parse_args()


def _rerun_algorithms(seed: int) -> None:
    for algorithm in ALGORITHM_ORDER:
        command = [
            sys.executable,
            RUN_COMMANDS[algorithm],
            "--seed",
            str(seed),
            "--skip-gifs",
        ]
        print(f"[comparison] running {algorithm}: {' '.join(command)}")
        subprocess.run(command, check=True)


def load_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}. Run the algorithm first or use --rerun."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_scenario_metadata(payload: dict[str, Any], path: Path) -> None:
    metadata = payload.get("scenario")
    if metadata is None:
        raise ValueError(
            f"{path} has no scenario metadata and may be stale. Rerun with --rerun."
        )
    expected = scenario_as_dict(DEFAULT_SCENARIO)
    if metadata != expected:
        raise ValueError(
            f"{path} was generated from a different scenario. Rerun with --rerun."
        )


def _validate_paths(payload: dict[str, Any], path: Path) -> None:
    paths = payload.get("paths")
    delays = payload.get("start_delays")
    if not isinstance(paths, list) or len(paths) != DEFAULT_SCENARIO.num_agents:
        raise ValueError(f"{path} does not contain one path per agent.")
    if not isinstance(delays, list) or len(delays) != DEFAULT_SCENARIO.num_agents:
        raise ValueError(f"{path} does not contain one delay per agent.")

    for index, (points, task) in enumerate(zip(paths, DEFAULT_SCENARIO.tasks)):
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
            raise ValueError(f"{path}: invalid path shape for agent {index + 1}.")
        if not np.allclose(array[0], task.start_array, atol=1e-9):
            raise ValueError(f"{path}: stale start point for agent {index + 1}.")
        if not np.allclose(array[-1], task.goal_array, atol=1e-9):
            raise ValueError(f"{path}: stale goal point for agent {index + 1}.")


def validate_result(payload: dict[str, Any], path: Path) -> None:
    _validate_scenario_metadata(payload, path)
    _validate_paths(payload, path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} has no metrics object.")
    missing = REQUIRED_METRICS.difference(metrics)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing current metrics: {names}.")


def payload_to_plan(payload: dict[str, Any]) -> JointPlan:
    return JointPlan(
        paths=tuple(np.asarray(path, dtype=float) for path in payload["paths"]),
        start_delays=np.asarray(payload["start_delays"], dtype=float),
    )


def collect_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for algorithm in ALGORITHM_ORDER:
        path = RESULT_FILES[algorithm]
        payload = load_result(path)
        validate_result(payload, path)
        results[algorithm] = payload
    return results


def comparison_rows(
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHM_ORDER:
        payload = results[algorithm]
        metrics = payload["metrics"]
        rows.append(
            {
                "algorithm": algorithm,
                "success": bool(metrics["success"]),
                "seed": int(payload["seed"]),
                "evaluations": int(payload["evaluations"]),
                "runtime_seconds": float(payload["runtime_seconds"]),
                "total_path_length": float(metrics["total_path_length"]),
                "makespan_seconds": float(metrics["makespan"]),
                "sum_start_delay_seconds": float(metrics["sum_start_delay"]),
                "minimum_obstacle_clearance": float(
                    metrics["minimum_obstacle_clearance"]
                ),
                "minimum_boundary_clearance": float(
                    metrics["minimum_boundary_clearance"]
                ),
                "minimum_inter_agent_distance": float(
                    metrics["minimum_inter_agent_distance"]
                ),
                "total_smoothness": float(metrics["total_smoothness"]),
                "total_backtracking": float(metrics["total_backtracking"]),
                "waypoint_spacing_imbalance": float(
                    metrics["waypoint_spacing_imbalance"]
                ),
                "start_delays": [float(value) for value in payload["start_delays"]],
            }
        )
    return rows


def save_comparison_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [key for key in rows[0] if key != "start_delays"] + [
        f"agent_{index + 1}_delay_seconds"
        for index in range(DEFAULT_SCENARIO.num_agents)
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: value for key, value in row.items() if key != "start_delays"}
            for index, value in enumerate(row["start_delays"]):
                flat[f"agent_{index + 1}_delay_seconds"] = value
            writer.writerow(flat)


def save_joint_plan_comparison(
    results: dict[str, dict[str, Any]], output_path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 13), constrained_layout=True)
    for axis, algorithm in zip(axes.ravel(), ALGORITHM_ORDER):
        plot_scenario(axis, DEFAULT_SCENARIO)
        plan = payload_to_plan(results[algorithm])
        for index, path in enumerate(plan.paths):
            axis.plot(
                path[:, 0],
                path[:, 1],
                marker="o",
                markersize=3.5,
                linewidth=2.0,
                label=(
                    f"Agent {index + 1} | delay="
                    f"{plan.start_delays[index]:.1f}s"
                ),
            )
        metrics = results[algorithm]["metrics"]
        axis.set_title(
            f"{algorithm} | length={metrics['total_path_length']:.1f} | "
            f"makespan={metrics['makespan']:.1f}s"
        )
        axis.legend(loc="upper left", fontsize=8)
    figure.suptitle("Multi-Agent Path Planning: Joint Plan Comparison", fontsize=18)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _bar_panel(
    axis: plt.Axes,
    algorithms: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    *,
    threshold: float | None = None,
) -> None:
    bars = axis.bar(algorithms, values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    if threshold is not None:
        axis.axhline(threshold, linestyle="--", linewidth=1.5, label="Required minimum")
        axis.legend(fontsize=8)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_metric_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    algorithms = [row["algorithm"] for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        ("total_path_length", "Total path length", "distance", None),
        ("makespan_seconds", "Makespan", "seconds", None),
        ("runtime_seconds", "Planning time", "seconds", None),
        (
            "minimum_inter_agent_distance",
            "Minimum inter-agent distance",
            "distance",
            DEFAULT_SCENARIO.minimum_agent_separation,
        ),
        (
            "minimum_obstacle_clearance",
            "Minimum obstacle clearance",
            "distance",
            DEFAULT_SCENARIO.obstacle_safety_margin,
        ),
        ("sum_start_delay_seconds", "Sum of start delays", "seconds", None),
    ]
    for axis, (key, title, ylabel, threshold) in zip(axes.ravel(), panels):
        _bar_panel(
            axis,
            algorithms,
            [float(row[key]) for row in rows],
            title,
            ylabel,
            threshold=threshold,
        )
    figure.suptitle("Single-Seed Comparison (seed=42)", fontsize=18)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_delay_comparison(rows: list[dict[str, Any]], output_path: Path) -> None:
    algorithms = [row["algorithm"] for row in rows]
    values = np.asarray([row["start_delays"] for row in rows], dtype=float)
    positions = np.arange(len(algorithms), dtype=float)
    width = 0.22
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for agent_index in range(DEFAULT_SCENARIO.num_agents):
        axis.bar(
            positions + (agent_index - 1) * width,
            values[:, agent_index],
            width=width,
            label=f"Agent {agent_index + 1}",
        )
    axis.set_xticks(positions, algorithms)
    axis.set_ylabel("Start delay (s)")
    axis.set_title("Optimized Start Delays")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("=== Multi-Agent Algorithm Comparison ===")
    header = (
        f"{'Algorithm':<10}{'Success':<10}{'Length':>12}{'Makespan':>12}"
        f"{'Runtime':>12}{'Min distance':>15}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['algorithm']:<10}{str(row['success']):<10}"
            f"{row['total_path_length']:>12.3f}"
            f"{row['makespan_seconds']:>12.3f}"
            f"{row['runtime_seconds']:>12.3f}"
            f"{row['minimum_inter_agent_distance']:>15.3f}"
        )


def main() -> None:
    args = parse_args()
    if args.rerun:
        _rerun_algorithms(args.seed)

    results = collect_results()
    rows = comparison_rows(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_comparison_table(rows, args.output_dir / "comparison_table.csv")
    save_joint_plan_comparison(
        results, args.output_dir / "comparison_joint_plans.png"
    )
    save_metric_comparison(rows, args.output_dir / "comparison_metrics.png")
    save_delay_comparison(rows, args.output_dir / "comparison_start_delays.png")

    summary = {
        "comparison_scope": "single representative run; not a statistical benchmark",
        "scenario": scenario_as_dict(DEFAULT_SCENARIO),
        "algorithms": rows,
    }
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print_summary(rows)
    print(f"generated: {args.output_dir}")


if __name__ == "__main__":
    main()
