"""Demonstrate D* Lite incremental replanning after obstacle discovery."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from config.scenarios import Scenario, get_dynamic_scenario
from planners.dstar_lite import DStarLitePlanner
from utils.metrics import calculate_minimum_clearance, calculate_path_length
from utils.visualization import (
    save_dynamic_replanning_animation,
    save_dynamic_replanning_figure,
)


def main() -> None:
    dynamic = get_dynamic_scenario()
    initial_scenario = Scenario(
        name=dynamic.name,
        width=dynamic.width,
        height=dynamic.height,
        start=dynamic.start,
        goal=dynamic.goal,
        obstacles=dynamic.static_obstacles,
        safety_margin=dynamic.safety_margin,
        grid_resolution=dynamic.grid_resolution,
        random_seed=dynamic.random_seed,
    )
    final_scenario = Scenario(
        name=dynamic.name,
        width=dynamic.width,
        height=dynamic.height,
        start=dynamic.start,
        goal=dynamic.goal,
        obstacles=dynamic.static_obstacles + dynamic.hidden_obstacles,
        safety_margin=dynamic.safety_margin,
        grid_resolution=dynamic.grid_resolution,
        random_seed=dynamic.random_seed,
    )

    planner = DStarLitePlanner(initial_scenario)

    initial_start = time.perf_counter()
    planner.compute_shortest_path()
    initial_path = planner.current_path()
    initial_time_ms = (time.perf_counter() - initial_start) * 1000.0
    if not initial_path:
        raise RuntimeError("D* Lite could not find the initial path.")

    travelled = [initial_path[0]]
    detection_point = None
    replanned_path = []
    changed_cells = 0
    replan_time_ms = 0.0

    hidden = dynamic.hidden_obstacles[0]
    for point in initial_path[1:]:
        travelled.append(point)
        distance_to_hidden_boundary = (
            math.hypot(point[0] - hidden.x, point[1] - hidden.y) - hidden.radius
        )
        if distance_to_hidden_boundary <= dynamic.sensor_range:
            detection_point = point
            planner.move_start(point)
            changed_cells = planner.reveal_obstacles(dynamic.hidden_obstacles)

            replan_start = time.perf_counter()
            replanned_path = planner.replan()
            replan_time_ms = (time.perf_counter() - replan_start) * 1000.0
            break

    if detection_point is None:
        raise RuntimeError("The hidden obstacle was not detected on the initial route.")
    if not replanned_path:
        raise RuntimeError("D* Lite could not find a path after obstacle discovery.")

    # Avoid duplicating the detection point when joining travelled and replanned paths.
    final_travelled_path = travelled + replanned_path[1:]

    output_dir = Path("results/dynamic")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dynamic_replanning_figure(
        initial_path=initial_path,
        replanned_path=replanned_path,
        travelled_path=final_travelled_path,
        detection_point=detection_point,
        static_scenario=initial_scenario,
        hidden_obstacles=dynamic.hidden_obstacles,
        output_path=output_dir / "dstar_lite_replanning.png",
    )
    save_dynamic_replanning_animation(
        initial_path=initial_path,
        replanned_path=replanned_path,
        travelled_before_detection=travelled,
        detection_point=detection_point,
        static_scenario=initial_scenario,
        hidden_obstacles=dynamic.hidden_obstacles,
        sensor_range=dynamic.sensor_range,
        output_path=output_dir / "dstar_lite_replanning.gif",
    )

    metrics = {
        "scenario": dynamic.name,
        "algorithm": "D* Lite",
        "success": True,
        "initial_planning_time_ms": initial_time_ms,
        "replanning_time_ms": replan_time_ms,
        "replanning_count": planner.replanning_count,
        "changed_grid_cells": changed_cells,
        "updated_vertices_total": planner.updated_vertices,
        "expanded_nodes_total": planner.expanded_nodes,
        "initial_path_length": calculate_path_length(initial_path),
        "final_travelled_distance": calculate_path_length(final_travelled_path),
        "final_minimum_clearance": calculate_minimum_clearance(
            final_travelled_path,
            final_scenario,
        ),
        "detection_point": list(detection_point),
        "sensor_range": dynamic.sensor_range,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print(
        "D* Lite dynamic replanning: "
        f"initial={initial_time_ms:.2f} ms, "
        f"replan={replan_time_ms:.2f} ms, "
        f"changed_cells={changed_cells}, "
        f"final_distance={metrics['final_travelled_distance']:.2f}"
    )


if __name__ == "__main__":
    main()
