"""Run centralized Multi-Agent PSO and generate PSO artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.scenario import DEFAULT_OBJECTIVE_WEIGHTS, DEFAULT_SCENARIO
from optimizers.pso import MultiAgentPSOConfig, run_multi_agent_pso
from utils.reporting import objective_weights_as_dict, scenario_as_dict
from utils.visualization import (
    save_fitness_convergence_figure,
    save_joint_motion_gif,
    save_joint_plan_figure,
    save_pso_search_gif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pso"),
        help="Directory for generated PSO artifacts.",
    )
    parser.add_argument(
        "--particles", type=int, default=80, help="Number of PSO particles."
    )
    parser.add_argument(
        "--iterations", type=int, default=150, help="Number of PSO iterations."
    )
    parser.add_argument(
        "--skip-gifs", action="store_true", help="Skip GIF generation."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = MultiAgentPSOConfig(
        num_particles=args.particles,
        max_iterations=args.iterations,
    )
    result = run_multi_agent_pso(
        DEFAULT_SCENARIO,
        config,
        seed=args.seed,
    )

    save_joint_plan_figure(
        result.plan,
        DEFAULT_SCENARIO,
        args.output_dir / "pso_joint_plan.png",
        title="Multi-Agent PSO: Optimized Joint Plan",
    )
    save_fitness_convergence_figure(
        result.fitness_history,
        args.output_dir / "pso_convergence.png",
    )
    if not args.skip_gifs:
        save_joint_motion_gif(
            result.plan,
            DEFAULT_SCENARIO,
            args.output_dir / "pso_joint_motion.gif",
            title="Multi-Agent PSO: Synchronized Motion",
        )
        save_pso_search_gif(
            result,
            DEFAULT_SCENARIO,
            args.output_dir / "pso_search_evolution.gif",
        )

    report = {
        "algorithm": result.algorithm,
        "seed": result.seed,
        "scenario": scenario_as_dict(DEFAULT_SCENARIO),
        "objective_weights": objective_weights_as_dict(DEFAULT_OBJECTIVE_WEIGHTS),
        "particles": config.num_particles,
        "iterations": result.iterations,
        "evaluations": result.evaluations,
        "runtime_seconds": result.runtime,
        "best_fitness": result.best_fitness,
        "start_delays": result.plan.start_delays.tolist(),
        "metrics": result.metrics.as_dict(),
        "objective": result.objective.as_dict(),
        "paths": [path.tolist() for path in result.plan.paths],
    }
    (args.output_dir / "pso_result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== Multi-Agent PSO ===")
    print(f"success: {result.success}")
    print(f"seed: {result.seed}")
    print(f"particles: {config.num_particles}")
    print(f"iterations: {result.iterations}")
    print(f"evaluations: {result.evaluations}")
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
