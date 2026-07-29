"""Internal subprocess worker for running exactly one optimizer trial."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from config.scenario import DEFAULT_SCENARIO
from optimizers import ACOConfig, GAConfig, GWOConfig, PSOConfig
from optimizers import run_aco, run_ga, run_gwo, run_pso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("ACO", "GA", "GWO", "PSO"))
    parser.add_argument("seed", type=int)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.algorithm == "ACO":
        result = run_aco(DEFAULT_SCENARIO, ACOConfig(), seed=args.seed)
    elif args.algorithm == "GA":
        result = run_ga(DEFAULT_SCENARIO, GAConfig(), seed=args.seed)
    elif args.algorithm == "GWO":
        result = run_gwo(DEFAULT_SCENARIO, GWOConfig(), seed=args.seed)
    else:
        result = run_pso(DEFAULT_SCENARIO, PSOConfig(), seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
