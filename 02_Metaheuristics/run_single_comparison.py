"""Run ACO, GA, GWO, and PSO once under the same path-planning scenario."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile


_ALGORITHM_ORDER = ("ACO", "GA", "GWO", "PSO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Representative random seed.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/single"),
        help="Directory for comparison outputs.",
    )
    parser.add_argument("--skip-gifs", action="store_true", help="Skip GIF generation.")
    parser.add_argument("--show", action="store_true", help="Open comparison figures.")
    return parser.parse_args()


def _launch_worker(algorithm: str, seed: int, result_file: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "utils.optimizer_worker",
            algorithm,
            str(seed),
            str(result_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        tasks = []
        for algorithm in _ALGORITHM_ORDER:
            result_file = temporary_path / f"{algorithm.lower()}.pkl"
            print(f"Launching {algorithm}...", flush=True)
            tasks.append(
                (algorithm, result_file, _launch_worker(algorithm, args.seed, result_file))
            )

        for algorithm, _, process in tasks:
            _, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f"{algorithm} worker failed:\n{stderr}")
            print(f"Finished {algorithm}.", flush=True)

        from config.scenario import DEFAULT_SCENARIO
        from utils.reporting import SINGLE_RESULT_COLUMNS, result_to_row, write_rows_csv
        from utils.visualization import (
            save_aco_evolution_gif,
            save_convergence_comparison_figure,
            save_ga_evolution_gif,
            save_gwo_evolution_gif,
            save_path_comparison_figure,
            save_pso_evolution_gif,
        )

        by_name = {}
        for algorithm, result_file, _ in tasks:
            with result_file.open("rb") as file:
                by_name[algorithm] = pickle.load(file)

        results = [by_name[name] for name in _ALGORITHM_ORDER]
        metrics_file = args.output_dir / "metrics.csv"
        path_file = args.output_dir / "path_comparison.png"
        convergence_file = args.output_dir / "convergence_comparison.png"

        write_rows_csv(
            [result_to_row(by_name[name]) for name in _ALGORITHM_ORDER],
            metrics_file,
            fieldnames=SINGLE_RESULT_COLUMNS,
        )
        save_path_comparison_figure(results, DEFAULT_SCENARIO, path_file, show=args.show)
        save_convergence_comparison_figure(results, convergence_file, show=args.show)

        if not args.skip_gifs:
            save_aco_evolution_gif(
                by_name["ACO"],
                DEFAULT_SCENARIO,
                args.output_dir / "aco_evolution.gif",
            )
            save_ga_evolution_gif(
                by_name["GA"],
                DEFAULT_SCENARIO,
                args.output_dir / "ga_evolution.gif",
            )
            save_gwo_evolution_gif(
                by_name["GWO"],
                DEFAULT_SCENARIO,
                args.output_dir / "gwo_evolution.gif",
            )
            save_pso_evolution_gif(
                by_name["PSO"],
                DEFAULT_SCENARIO,
                args.output_dir / "pso_evolution.gif",
            )

    print("\n=== Single comparison ===")
    for name in _ALGORITHM_ORDER:
        result = by_name[name]
        print(
            f"{name}: success={result.success}, fitness={result.best_fitness:.4f}, "
            f"length={result.metrics.path_length:.4f}, "
            f"clearance={result.metrics.minimum_clearance:.4f}, "
            f"cpu_time={result.runtime:.4f}s"
        )

    print("\nGenerated files:")
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()
