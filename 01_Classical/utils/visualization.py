"""Common plotting and animation functions for path-planning results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from config.scenarios import Scenario
from utils.metrics import PlanningResult


def _draw_environment(ax: plt.Axes, scenario: Scenario) -> None:
    ax.set_xlim(0, scenario.width)
    ax.set_ylim(0, scenario.height)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    for obstacle in scenario.obstacles:
        obstacle_patch = plt.Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            alpha=0.55,
            label="Obstacle",
        )
        safety_patch = plt.Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius + scenario.safety_margin,
            fill=False,
            linestyle="--",
            alpha=0.45,
            label="Safety margin",
        )
        ax.add_patch(obstacle_patch)
        ax.add_patch(safety_patch)

    ax.scatter(*scenario.start, marker="o", s=80, label="Start", zorder=5)
    ax.scatter(*scenario.goal, marker="*", s=140, label="Goal", zorder=5)


def _unique_legend(ax: plt.Axes, *, loc: str = "upper left") -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc=loc)


def save_result_figure(
    result: PlanningResult,
    scenario: Scenario,
    output_path: str | Path,
) -> None:
    """Save one planner result using the common visual style."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_environment(ax, scenario)

    if result.path:
        x_values = [point[0] for point in result.path]
        y_values = [point[1] for point in result.path]
        ax.plot(x_values, y_values, linewidth=2.0, label=result.algorithm, zorder=4)

    status = "Success" if result.success else "Failed"
    ax.set_title(
        f"{result.algorithm} - {status}\n"
        f"Length: {result.path_length:.2f} | "
        f"Time: {result.planning_time_ms:.2f} ms | "
        f"Waypoints: {result.waypoint_count}"
    )

    _unique_legend(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_path_animation(
    result: PlanningResult,
    scenario: Scenario,
    output_path: str | Path,
    *,
    fps: int = 12,
    max_frames: int = 140,
) -> None:
    """Save a GIF that animates an agent along a computed path.

    The planner output is not modified. Long paths are sampled only for GIF
    rendering so that repository file sizes remain manageable.
    """

    if not result.path:
        return

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    stride = max(1, (len(result.path) + max_frames - 1) // max_frames)
    frames = list(result.path[::stride])
    if frames[-1] != result.path[-1]:
        frames.append(result.path[-1])

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_environment(ax, scenario)

    full_x = [point[0] for point in result.path]
    full_y = [point[1] for point in result.path]
    ax.plot(full_x, full_y, linestyle="--", linewidth=1.2, alpha=0.35, label="Planned path")

    trail_line, = ax.plot([], [], linewidth=2.3, label="Travelled route", zorder=4)
    agent, = ax.plot([], [], marker="o", markersize=8, linestyle="None", label="Agent", zorder=6)
    step_text = ax.text(0.02, 0.02, "", transform=ax.transAxes)

    status = "Success" if result.success else "Failed"
    ax.set_title(f"{result.algorithm} Path Animation - {status}")
    _unique_legend(ax)

    def init():
        trail_line.set_data([], [])
        agent.set_data([], [])
        step_text.set_text("")
        return trail_line, agent, step_text

    def update(frame_index: int):
        visited = frames[: frame_index + 1]
        x_values = [point[0] for point in visited]
        y_values = [point[1] for point in visited]
        trail_line.set_data(x_values, y_values)
        agent.set_data([x_values[-1]], [y_values[-1]])
        step_text.set_text(f"Frame {frame_index + 1}/{len(frames)}")
        return trail_line, agent, step_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        blit=True,
        interval=1000 / fps,
        repeat=False,
    )
    fig.tight_layout()
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)




def save_rrt_growth_animation(
    result: PlanningResult,
    scenario: Scenario,
    output_path: str | Path,
    *,
    fps: int = 12,
    max_tree_frames: int = 80,
    max_flight_frames: int = 55,
    hold_frames: int = 10,
) -> None:
    """Animate RRT tree growth, goal connection, and final route flight."""

    if not result.path:
        return

    tree_edges = list(result.visualization_data.get("tree_edges", []))
    if not tree_edges:
        save_path_animation(result, scenario, output_path, fps=fps)
        return

    from matplotlib.collections import LineCollection

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tree_stride = max(1, (len(tree_edges) + max_tree_frames - 1) // max_tree_frames)
    edge_counts = list(range(tree_stride, len(tree_edges) + 1, tree_stride))
    if edge_counts[-1] != len(tree_edges):
        edge_counts.append(len(tree_edges))

    path_stride = max(1, (len(result.path) + max_flight_frames - 1) // max_flight_frames)
    flight_points = list(result.path[::path_stride])
    if flight_points[-1] != result.path[-1]:
        flight_points.append(result.path[-1])

    tree_frames = len(edge_counts)
    connection_frames = 6
    flight_frames = len(flight_points)
    total_frames = tree_frames + connection_frames + flight_frames + hold_frames

    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_environment(ax, scenario)

    segments = [[parent, child] for parent, child in tree_edges]
    tree_collection = LineCollection([], linewidths=0.7, alpha=0.45, zorder=2)
    ax.add_collection(tree_collection)

    route_x = [point[0] for point in result.path]
    route_y = [point[1] for point in result.path]
    selected_route, = ax.plot(
        route_x, route_y, linewidth=3.0, alpha=0.0, label="Selected route", zorder=5
    )
    travelled_route, = ax.plot([], [], linewidth=3.4, label="Drone route", zorder=6)
    drone, = ax.plot([], [], marker="o", markersize=9, linestyle="None", label="Drone", zorder=7)
    connection_marker, = ax.plot(
        [], [], marker="X", markersize=10, linestyle="None", label="Goal connected", zorder=8
    )
    state_text = ax.text(0.02, 0.02, "", transform=ax.transAxes)

    ax.set_title("RRT Tree Growth and Route Execution")
    _unique_legend(ax)

    def init():
        tree_collection.set_segments([])
        selected_route.set_alpha(0.0)
        travelled_route.set_data([], [])
        drone.set_data([], [])
        connection_marker.set_data([], [])
        state_text.set_text("")
        return tree_collection, selected_route, travelled_route, drone, connection_marker, state_text

    def update(frame_index: int):
        if frame_index < tree_frames:
            visible_count = edge_counts[frame_index]
            tree_collection.set_segments(segments[:visible_count])
            tree_collection.set_alpha(0.45)
            state_text.set_text(f"Growing search tree: {visible_count}/{len(tree_edges)} branches")

        elif frame_index < tree_frames + connection_frames:
            tree_collection.set_segments(segments)
            tree_collection.set_alpha(0.25)
            progress = (frame_index - tree_frames + 1) / connection_frames
            selected_route.set_alpha(progress)
            connection_marker.set_data([scenario.goal[0]], [scenario.goal[1]])
            state_text.set_text("Goal reached - extracting route")

        else:
            tree_collection.set_segments(segments)
            tree_collection.set_alpha(0.18)
            selected_route.set_alpha(0.85)
            connection_marker.set_data([scenario.goal[0]], [scenario.goal[1]])
            flight_index = min(frame_index - tree_frames - connection_frames, flight_frames - 1)
            visited = flight_points[: flight_index + 1]
            travelled_route.set_data([p[0] for p in visited], [p[1] for p in visited])
            drone.set_data([visited[-1][0]], [visited[-1][1]])
            state_text.set_text(
                "Drone following the selected RRT route"
                if flight_index < flight_frames - 1 else "Goal reached"
            )

        return tree_collection, selected_route, travelled_route, drone, connection_marker, state_text

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=True,
        interval=1000 / fps,
        repeat=False,
    )
    fig.tight_layout()
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)

def save_comparison_figure(
    results: Iterable[PlanningResult],
    scenario: Scenario,
    output_path: str | Path,
) -> None:
    """Save planner results as equally styled side-by-side panels."""

    result_list = list(results)
    if not result_list:
        raise ValueError("At least one result is required for comparison.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    column_count = min(2, len(result_list))
    row_count = (len(result_list) + column_count - 1) // column_count
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7 * column_count, 7 * row_count),
        squeeze=False,
    )

    flat_axes = axes.ravel()
    for ax, result in zip(flat_axes, result_list):
        _draw_environment(ax, scenario)
        if result.path:
            x_values = [point[0] for point in result.path]
            y_values = [point[1] for point in result.path]
            ax.plot(
                x_values,
                y_values,
                linewidth=2.0,
                label=result.algorithm,
                zorder=4,
            )

        status = "Success" if result.success else "Failed"
        ax.set_title(
            f"{result.algorithm} - {status}\n"
            f"Length: {result.path_length:.2f} | "
            f"Time: {result.planning_time_ms:.2f} ms"
        )
        _unique_legend(ax)

    for ax in flat_axes[len(result_list) :]:
        ax.axis("off")

    fig.suptitle("Static Environment Path Comparison", fontsize=16)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_dynamic_replanning_figure(
    initial_path,
    replanned_path,
    travelled_path,
    detection_point,
    static_scenario: Scenario,
    hidden_obstacles,
    output_path: str | Path,
) -> None:
    """Save a D* Lite before/after replanning visualization."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_environment(ax, static_scenario)

    for obstacle in hidden_obstacles:
        hidden_patch = plt.Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            alpha=0.65,
            label="Discovered obstacle",
            zorder=3,
        )
        hidden_margin = plt.Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius + static_scenario.safety_margin,
            fill=False,
            linestyle="--",
            alpha=0.55,
            label="Discovered safety margin",
            zorder=3,
        )
        ax.add_patch(hidden_patch)
        ax.add_patch(hidden_margin)

    if initial_path:
        ax.plot(
            [point[0] for point in initial_path],
            [point[1] for point in initial_path],
            linestyle="--",
            linewidth=1.8,
            alpha=0.7,
            label="Initial path",
            zorder=4,
        )
    if replanned_path:
        ax.plot(
            [point[0] for point in replanned_path],
            [point[1] for point in replanned_path],
            linewidth=2.2,
            label="Replanned path",
            zorder=5,
        )
    if travelled_path:
        ax.plot(
            [point[0] for point in travelled_path],
            [point[1] for point in travelled_path],
            linewidth=3.0,
            alpha=0.45,
            label="Travelled route",
            zorder=4,
        )

    ax.scatter(
        detection_point[0],
        detection_point[1],
        marker="X",
        s=110,
        label="Detection point",
        zorder=6,
    )
    ax.set_title("D* Lite Dynamic Replanning")

    _unique_legend(ax)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_dynamic_replanning_animation(
    initial_path: Sequence[tuple[float, float]],
    replanned_path: Sequence[tuple[float, float]],
    travelled_before_detection: Sequence[tuple[float, float]],
    detection_point: tuple[float, float],
    static_scenario: Scenario,
    hidden_obstacles,
    sensor_range: float,
    output_path: str | Path,
    *,
    fps: int = 10,
) -> None:
    """Animate hidden-obstacle discovery and D* Lite path replanning."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    before = list(travelled_before_detection)
    after = list(replanned_path[1:]) if len(replanned_path) > 1 else []
    full_route = before + after
    detection_frame = max(0, len(before) - 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_environment(ax, static_scenario)

    ax.plot(
        [point[0] for point in initial_path],
        [point[1] for point in initial_path],
        linestyle="--",
        linewidth=1.6,
        alpha=0.5,
        label="Initial path",
        zorder=3,
    )

    hidden_patches = []
    hidden_margins = []
    for obstacle in hidden_obstacles:
        patch = plt.Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius,
            alpha=0.0,
            label="Discovered obstacle",
            zorder=4,
        )
        margin = plt.Circle(
            (obstacle.x, obstacle.y),
            obstacle.radius + static_scenario.safety_margin,
            fill=False,
            linestyle="--",
            alpha=0.0,
            label="Discovered safety margin",
            zorder=4,
        )
        ax.add_patch(patch)
        ax.add_patch(margin)
        hidden_patches.append(patch)
        hidden_margins.append(margin)

    replanned_line, = ax.plot([], [], linewidth=2.2, label="Replanned path", zorder=5)
    travelled_line, = ax.plot([], [], linewidth=3.0, alpha=0.55, label="Travelled route", zorder=5)
    agent, = ax.plot([], [], marker="o", markersize=8, linestyle="None", label="Agent", zorder=7)
    sensor = plt.Circle(static_scenario.start, sensor_range, fill=False, linestyle=":", alpha=0.45, label="Sensor range")
    ax.add_patch(sensor)
    detection_marker, = ax.plot([], [], marker="X", markersize=9, linestyle="None", label="Detection point", zorder=7)
    state_text = ax.text(0.02, 0.02, "", transform=ax.transAxes)

    ax.set_title("D* Lite Dynamic Replanning Animation")
    _unique_legend(ax)

    def init():
        travelled_line.set_data([], [])
        replanned_line.set_data([], [])
        agent.set_data([], [])
        detection_marker.set_data([], [])
        state_text.set_text("")
        return (
            travelled_line,
            replanned_line,
            agent,
            sensor,
            detection_marker,
            state_text,
            *hidden_patches,
            *hidden_margins,
        )

    def update(frame_index: int):
        current = full_route[frame_index]
        visited = full_route[: frame_index + 1]
        travelled_line.set_data(
            [point[0] for point in visited],
            [point[1] for point in visited],
        )
        agent.set_data([current[0]], [current[1]])
        sensor.center = current

        if frame_index >= detection_frame:
            for patch in hidden_patches:
                patch.set_alpha(0.7)
            for margin in hidden_margins:
                margin.set_alpha(0.55)
            detection_marker.set_data([detection_point[0]], [detection_point[1]])
            replanned_line.set_data(
                [point[0] for point in replanned_path],
                [point[1] for point in replanned_path],
            )
            state_text.set_text("Obstacle detected - path replanned")
        else:
            state_text.set_text("Following initial path")

        return (
            travelled_line,
            replanned_line,
            agent,
            sensor,
            detection_marker,
            state_text,
            *hidden_patches,
            *hidden_margins,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(full_route),
        init_func=init,
        blit=True,
        interval=1000 / fps,
        repeat=False,
    )
    fig.tight_layout()
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
