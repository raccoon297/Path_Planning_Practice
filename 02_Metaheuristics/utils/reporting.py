"""CSV reporting helpers for representative single-run results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SINGLE_RESULT_COLUMNS = (
    "algorithm",
    "seed",
    "success",
    "best_fitness",
    "path_length",
    "cpu_time_seconds",
    "waypoint_count",
    "minimum_clearance",
    "smoothness",
    "collision_count",
    "safety_violation_count",
    "iterations",
    "evaluations",
)


def result_to_row(result: Any) -> dict[str, Any]:
    """Convert any optimizer result into the shared tabular schema."""

    return {
        "algorithm": result.algorithm,
        "seed": result.seed,
        "success": bool(result.success),
        "best_fitness": float(result.best_fitness),
        "path_length": float(result.metrics.path_length),
        "cpu_time_seconds": float(result.runtime),
        "waypoint_count": int(result.metrics.waypoint_count),
        "minimum_clearance": float(result.metrics.minimum_clearance),
        "smoothness": float(result.metrics.smoothness),
        "collision_count": int(result.metrics.collision_count),
        "safety_violation_count": int(result.metrics.safety_violation_count),
        "iterations": int(result.iterations),
        "evaluations": int(result.evaluations),
    }


def write_rows_csv(
    rows: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write dictionaries to UTF-8 CSV with stable column ordering."""

    materialized = [dict(row) for row in rows]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        ordered: list[str] = []
        for row in materialized:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered

    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
