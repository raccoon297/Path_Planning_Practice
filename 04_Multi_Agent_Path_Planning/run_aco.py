"""Run reservation-aware Multi-Agent ACO and generate ACO artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.scenario import DEFAULT_OBJECTIVE_WEIGHTS, DEFAULT_SCENARIO
from optimizers.aco import MultiAgentACOConfig, run_multi_agent_aco
from utils.reporting import objective_weights_as_dict, scenario_as_dict
from utils.visualization import (
    save_aco_search_gif,
    save_fitness_convergence_figure,
    save_joint_motion_gif,
    save_joint_plan_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aco"),
        help="Directory for generated ACO artifacts.",
    )
    parser.add_argument("--ants", type=int, default=40, help="Number of joint ants.")
    parser.add_argument(
        "--iterations", type=int, default=40, help="Number of ACO colonies."
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=5.0,
        help="Spacing of the 8-connected search grid.",
    )
    parser.add_argument("--skip-gifs", action="store_true", help="Skip GIF generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = MultiAgentACOConfig(
        num_ants=args.ants,
        max_iterations=args.iterations,
        grid_resolution=args.grid_resolution,
    )
    result = run_multi_agent_aco(DEFAULT_SCENARIO, config, seed=args.seed)

    save_joint_plan_figure(
        result.plan,
        DEFAULT_SCENARIO,
        args.output_dir / "aco_joint_plan.png",
        title="Multi-Agent ACO: Optimized Joint Plan",
    )
    save_fitness_convergence_figure(
        result.fitness_history,
        args.output_dir / "aco_convergence.png",
        title="Multi-Agent ACO Best-so-far Objective",
    )
    if not args.skip_gifs:
        save_joint_motion_gif(
            result.plan,
            DEFAULT_SCENARIO,
            args.output_dir / "aco_joint_motion.gif",
            title="Multi-Agent ACO: Synchronized Motion",
        )
        save_aco_search_gif(
            result,
            DEFAULT_SCENARIO,
            args.output_dir / "aco_search_evolution.gif",
        )

    report = {
        "algorithm": result.algorithm,
        "seed": result.seed,
        "scenario": scenario_as_dict(DEFAULT_SCENARIO),
        "objective_weights": objective_weights_as_dict(DEFAULT_OBJECTIVE_WEIGHTS),
        "ants": config.num_ants,
        "iterations": result.iterations,
        "evaluations": result.evaluations,
        "runtime_seconds": result.runtime,
        "grid_rows": result.graph_rows,
        "grid_cols": result.graph_cols,
        "grid_resolution": result.grid_resolution,
        "delay_step": result.delay_step,
        "successful_candidates": result.successful_candidates,
        "best_fitness": result.best_fitness,
        "start_delays": result.plan.start_delays.tolist(),
        "metrics": result.metrics.as_dict(),
        "objective": result.objective.as_dict(),
        "paths": [path.tolist() for path in result.plan.paths],
        "raw_paths": [path.tolist() for path in result.raw_paths],
    }
    (args.output_dir / "aco_result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== Multi-Agent ACO ===")
    print(f"success: {result.success}")
    print(f"seed: {result.seed}")
    print(f"ants: {config.num_ants}")
    print(f"iterations: {result.iterations}")
    print(f"evaluations: {result.evaluations}")
    print(f"successful candidates: {result.successful_candidates}")
    print(f"grid: {result.graph_rows} x {result.graph_cols}")
    print(f"planning time: {result.runtime:.3f} s")
    print(f"best fitness: {result.best_fitness:.3f}")
    print(f"start delays: {result.plan.start_delays.round(3).tolist()}")
    print(f"total path length: {result.metrics.total_path_length:.3f}")
    print(f"makespan: {result.metrics.makespan:.3f} s")
    print(
        "minimum inter-agent distance: "
        f"{result.metrics.minimum_inter_agent_distance:.3f}"
    )
    print(
        "minimum boundary clearance: "
        f"{result.metrics.minimum_boundary_clearance:.3f}"
    )
    print(
        "boundary margin violations: "
        f"{result.metrics.boundary_safety_violation_count}"
    )
    print(f"generated: {args.output_dir}")


if __name__ == "__main__":
    main()
