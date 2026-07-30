"""Run centralized Multi-Agent GA and generate GA artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.scenario import DEFAULT_OBJECTIVE_WEIGHTS, DEFAULT_SCENARIO
from optimizers.ga import MultiAgentGAConfig, run_multi_agent_ga
from utils.reporting import objective_weights_as_dict, scenario_as_dict
from utils.visualization import (
    save_fitness_convergence_figure,
    save_ga_search_gif,
    save_joint_motion_gif,
    save_joint_plan_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ga"),
        help="Directory for generated GA artifacts.",
    )
    parser.add_argument(
        "--population", type=int, default=80, help="GA population size."
    )
    parser.add_argument(
        "--generations", type=int, default=150, help="Number of GA generations."
    )
    parser.add_argument("--skip-gifs", action="store_true", help="Skip GIF generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = MultiAgentGAConfig(
        population_size=args.population,
        max_generations=args.generations,
    )
    result = run_multi_agent_ga(DEFAULT_SCENARIO, config, seed=args.seed)

    save_joint_plan_figure(
        result.plan,
        DEFAULT_SCENARIO,
        args.output_dir / "ga_joint_plan.png",
        title="Multi-Agent GA: Optimized Joint Plan",
    )
    save_fitness_convergence_figure(
        result.fitness_history,
        args.output_dir / "ga_convergence.png",
        title="Multi-Agent GA Best-so-far Objective",
    )
    if not args.skip_gifs:
        save_joint_motion_gif(
            result.plan,
            DEFAULT_SCENARIO,
            args.output_dir / "ga_joint_motion.gif",
            title="Multi-Agent GA: Synchronized Motion",
        )
        save_ga_search_gif(
            result,
            DEFAULT_SCENARIO,
            args.output_dir / "ga_search_evolution.gif",
        )

    report = {
        "algorithm": result.algorithm,
        "seed": result.seed,
        "scenario": scenario_as_dict(DEFAULT_SCENARIO),
        "objective_weights": objective_weights_as_dict(DEFAULT_OBJECTIVE_WEIGHTS),
        "population_size": config.population_size,
        "generations": result.iterations,
        "evaluations": result.evaluations,
        "runtime_seconds": result.runtime,
        "best_fitness": result.best_fitness,
        "start_delays": result.plan.start_delays.tolist(),
        "metrics": result.metrics.as_dict(),
        "objective": result.objective.as_dict(),
        "paths": [path.tolist() for path in result.plan.paths],
    }
    (args.output_dir / "ga_result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== Multi-Agent GA ===")
    print(f"success: {result.success}")
    print(f"seed: {result.seed}")
    print(f"population: {config.population_size}")
    print(f"generations: {result.iterations}")
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
