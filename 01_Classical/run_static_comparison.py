"""Run the shared static path-planning comparison.

Stage 4 compares A*, APF, RRT, and a genuine incremental D* Lite
implementation under the same map and evaluation functions.
"""

from __future__ import annotations

import json
from pathlib import Path

from config.scenarios import get_static_scenario
from planners.apf import APFPlanner
from planners.astar import AStarPlanner
from planners.dstar_lite import DStarLitePlanner
from planners.rrt import RRTPlanner
from utils.visualization import (
    save_comparison_figure,
    save_path_animation,
    save_result_figure,
    save_rrt_growth_animation,
)


def main() -> None:
    scenario = get_static_scenario()
    output_dir = Path("results/static")
    output_dir.mkdir(parents=True, exist_ok=True)

    planners = [
        AStarPlanner(scenario),
        APFPlanner(scenario),
        RRTPlanner(scenario),
        DStarLitePlanner(scenario),
    ]
    results = []

    for planner in planners:
        result = planner.plan()
        results.append(result)

        filename = result.algorithm.lower().replace("*", "star").replace(" ", "_")
        save_result_figure(
            result,
            scenario,
            output_dir / f"{filename}_result.png",
        )
        if result.algorithm == "RRT":
            save_rrt_growth_animation(
                result,
                scenario,
                output_dir / f"{filename}_animation.gif",
            )
        else:
            save_path_animation(
                result,
                scenario,
                output_dir / f"{filename}_animation.gif",
            )

        print(
            f"{result.algorithm}: success={result.success}, "
            f"length={result.path_length:.2f}, "
            f"time={result.planning_time_ms:.2f} ms, "
            f"waypoints={result.waypoint_count}, "
            f"minimum_clearance={result.minimum_clearance:.2f}"
        )

    save_comparison_figure(
        results,
        scenario,
        output_dir / "path_comparison.png",
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "scenario": scenario.name,
                "results": [result.to_dict() for result in results],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
