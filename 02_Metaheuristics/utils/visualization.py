"""Reusable visualizations for metaheuristic path-planning experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config.scenario import Scenario
from .path_utils import vector_to_path


_ALGORITHM_ORDER = ("ACO", "GA", "GWO", "PSO")


def plot_scenario(ax: plt.Axes, scenario: Scenario) -> None:
    """Draw the common map, physical obstacles, and safety margins."""

    ax.set_xlim(0.0, scenario.width)
    ax.set_ylim(0.0, scenario.height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    for index, obstacle in enumerate(scenario.obstacles):
        physical = plt.Circle(
            obstacle.center,
            obstacle.radius,
            alpha=0.45,
            label="Obstacle" if index == 0 else None,
        )
        safety = plt.Circle(
            obstacle.center,
            obstacle.radius + scenario.safety_margin,
            fill=False,
            linestyle="--",
            alpha=0.7,
            label="Safety margin" if index == 0 else None,
        )
        ax.add_patch(physical)
        ax.add_patch(safety)

    ax.scatter(*scenario.start, marker="o", s=80, label="Start", zorder=5)
    ax.scatter(*scenario.goal, marker="*", s=150, label="Goal", zorder=5)


def save_path_figure(
    path: np.ndarray,
    scenario: Scenario,
    output_path: str | Path,
    *,
    title: str,
    show: bool = False,
) -> None:
    """Save one final path over the common scenario."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 7))
    plot_scenario(axis, scenario)
    axis.plot(path[:, 0], path[:, 1], "-o", linewidth=2, markersize=4, label="Path")
    axis.set_title(title)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_convergence_figure(
    fitness_history: np.ndarray,
    output_path: str | Path,
    *,
    title: str,
    show: bool = False,
) -> None:
    """Save a best-so-far objective convergence curve."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    history = np.asarray(fitness_history, dtype=float)
    axis.plot(np.arange(len(history)), history, linewidth=2)
    finite = history[np.isfinite(history)]
    if finite.size and np.all(finite > 0.0) and finite.max() / finite.min() >= 100.0:
        axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("Iteration / generation")
    axis.set_ylabel("Best-so-far objective")
    axis.grid(True, alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_path_comparison_figure(
    results: Sequence[Any],
    scenario: Scenario,
    output_path: str | Path,
    *,
    show: bool = False,
) -> None:
    """Save a 2x2 final-path comparison using identical map axes."""

    by_name = {result.algorithm: result for result in results}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    for axis, name in zip(axes.flat, _ALGORITHM_ORDER):
        result = by_name[name]
        plot_scenario(axis, scenario)
        path = np.asarray(result.path, dtype=float)
        axis.plot(path[:, 0], path[:, 1], "-o", linewidth=2, markersize=3, label=name)
        status = "success" if result.success else "failed"
        axis.set_title(
            f"{name} | {status} | length={result.metrics.path_length:.2f} | "
            f"planning time={result.runtime:.2f}s"
        )
        axis.legend(loc="best", fontsize=8)

    figure.suptitle("Metaheuristic Path Planning: Final Paths", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def save_convergence_comparison_figure(
    results: Sequence[Any],
    output_path: str | Path,
    *,
    show: bool = False,
) -> None:
    """Save all best-so-far objective curves on one axis."""

    by_name = {result.algorithm: result for result in results}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 6))
    for name in _ALGORITHM_ORDER:
        result = by_name[name]
        history = np.asarray(result.fitness_history, dtype=float)
        finite_mask = np.isfinite(history)
        axis.plot(
            np.arange(len(history))[finite_mask],
            history[finite_mask],
            linewidth=2,
            label=name,
        )

    axis.set_title("Best-so-far Objective Convergence")
    axis.set_xlabel("Iteration / generation")
    axis.set_ylabel("Shared objective value")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


# ---------------------------------------------------------------------------
# Algorithm-specific GIF renderers
# ---------------------------------------------------------------------------

_CANVAS_WIDTH = 840
_CANVAS_HEIGHT = 650
_MAP_MARGIN_LEFT = 55
_MAP_MARGIN_TOP = 60
_MAP_SIZE = 540
_PANEL_LEFT = 625


def _history_indices(length: int, max_frames: int) -> np.ndarray:
    if length < 1:
        raise ValueError("History must contain at least one frame.")
    count = min(length, max_frames)
    return np.unique(np.linspace(0, length - 1, count, dtype=int))


def _to_pixel(
    point: np.ndarray | tuple[float, float], scenario: Scenario
) -> tuple[int, int]:
    x, y = float(point[0]), float(point[1])
    px = _MAP_MARGIN_LEFT + int(round((x / scenario.width) * _MAP_SIZE))
    py = _MAP_MARGIN_TOP + _MAP_SIZE - int(round((y / scenario.height) * _MAP_SIZE))
    return px, py


def _radius_to_pixels(radius: float, scenario: Scenario) -> int:
    scale = min(_MAP_SIZE / scenario.width, _MAP_SIZE / scenario.height)
    return max(1, int(round(radius * scale)))


def _draw_base_map(draw: ImageDraw.ImageDraw, scenario: Scenario) -> None:
    draw.rectangle(
        (
            _MAP_MARGIN_LEFT,
            _MAP_MARGIN_TOP,
            _MAP_MARGIN_LEFT + _MAP_SIZE,
            _MAP_MARGIN_TOP + _MAP_SIZE,
        ),
        fill=(250, 250, 250),
        outline=(70, 70, 70),
        width=2,
    )

    for obstacle in scenario.obstacles:
        cx, cy = _to_pixel(obstacle.center, scenario)
        safety_radius = _radius_to_pixels(
            obstacle.radius + scenario.safety_margin, scenario
        )
        physical_radius = _radius_to_pixels(obstacle.radius, scenario)
        safety_box = (
            cx - safety_radius,
            cy - safety_radius,
            cx + safety_radius,
            cy + safety_radius,
        )
        for angle in range(0, 360, 20):
            draw.arc(safety_box, angle, angle + 10, fill=(135, 135, 135), width=2)
        draw.ellipse(
            (
                cx - physical_radius,
                cy - physical_radius,
                cx + physical_radius,
                cy + physical_radius,
            ),
            fill=(185, 185, 185),
            outline=(85, 85, 85),
            width=2,
        )

    start = _to_pixel(scenario.start, scenario)
    goal = _to_pixel(scenario.goal, scenario)
    draw.ellipse(
        (start[0] - 7, start[1] - 7, start[0] + 7, start[1] + 7),
        fill=(35, 145, 75),
        outline=(20, 90, 45),
    )
    draw.ellipse(
        (goal[0] - 9, goal[1] - 9, goal[0] + 9, goal[1] + 9),
        fill=(205, 55, 55),
        outline=(125, 25, 25),
    )


def _draw_path(
    draw: ImageDraw.ImageDraw,
    path: np.ndarray,
    scenario: Scenario,
    *,
    fill: tuple[int, int, int],
    width: int,
    nodes: bool = False,
) -> None:
    array = np.asarray(path, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2:
        return
    pixels = [_to_pixel(point, scenario) for point in array]
    draw.line(pixels, fill=fill, width=width, joint="curve")
    if nodes:
        for px, py in pixels[1:-1]:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=fill)


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 2,
    dash: int = 7,
) -> None:
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    vector = end_array - start_array
    distance = float(np.linalg.norm(vector))
    if distance < 1e-9:
        return
    direction = vector / distance
    position = 0.0
    while position < distance:
        segment_end = min(position + dash, distance)
        p0 = start_array + direction * position
        p1 = start_array + direction * segment_end
        draw.line([tuple(p0.astype(int)), tuple(p1.astype(int))], fill=fill, width=width)
        position += 2 * dash


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 2,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    vector = np.asarray(start, dtype=float) - np.asarray(end, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return
    unit = vector / norm
    perpendicular = np.asarray([-unit[1], unit[0]])
    tip = np.asarray(end, dtype=float)
    left = tip + 9 * unit + 4 * perpendicular
    right = tip + 9 * unit - 4 * perpendicular
    draw.polygon([end, tuple(left.astype(int)), tuple(right.astype(int))], fill=fill)


def _new_frame(title: str, scenario: Scenario) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (_CANVAS_WIDTH, _CANVAS_HEIGHT), (245, 246, 248))
    draw = ImageDraw.Draw(image)
    draw.text((_MAP_MARGIN_LEFT, 22), title, fill=(20, 20, 24))
    _draw_base_map(draw, scenario)
    draw.rectangle(
        (_PANEL_LEFT - 14, _MAP_MARGIN_TOP, _CANVAS_WIDTH - 24, _MAP_MARGIN_TOP + _MAP_SIZE),
        fill=(255, 255, 255),
        outline=(205, 208, 214),
        width=1,
    )
    return image, draw


def _panel_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[tuple[str, tuple[int, int, int]]],
    *,
    start_y: int = 82,
    gap: int = 25,
) -> None:
    y = start_y
    for text, color in lines:
        draw.text((_PANEL_LEFT, y), text, fill=color)
        y += gap


def _save_gif(frames: Sequence[Image.Image], output_path: str | Path, fps: int) -> None:
    if not frames:
        raise ValueError("At least one GIF frame is required.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(40, int(round(1000 / fps)))
    frames[0].save(
        output,
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_aco_evolution_gif(
    result: Any,
    scenario: Scenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    fps: int = 4,
) -> None:
    """Visualize ant routes and the emergence of a pheromone corridor."""

    pheromone_history = np.asarray(result.pheromone_history, dtype=float)
    indices = _history_indices(len(pheromone_history), max_frames)
    frames: list[Image.Image] = []

    for index in indices:
        image, draw = _new_frame("ACO: pheromone-guided path reinforcement", scenario)
        pheromone = pheromone_history[index]
        positive = pheromone[pheromone > 0]
        scale = float(np.percentile(np.log1p(positive), 97)) if positive.size else 1.0
        scale = max(scale, 1e-9)
        rows, cols = pheromone.shape
        for row in range(rows):
            for col in range(cols):
                value = float(pheromone[row, col])
                if value <= 0.0:
                    continue
                strength = min(1.0, np.log1p(value) / scale)
                if strength < 0.12:
                    continue
                x = col * result.grid_resolution
                y = row * result.grid_resolution
                px, py = _to_pixel((x, y), scenario)
                radius = 1 + int(round(3 * strength))
                color = (
                    235,
                    int(round(205 - 100 * strength)),
                    int(round(90 - 45 * strength)),
                )
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

        colony_paths = result.colony_path_history[index]
        for path in colony_paths[:10]:
            _draw_path(draw, path, scenario, fill=(125, 125, 135), width=1)

        best_path = result.best_path_history[index]
        if np.asarray(best_path).shape[0] >= 2:
            _draw_path(
                draw,
                best_path,
                scenario,
                fill=(30, 85, 185),
                width=5,
                nodes=True,
            )

        _panel_lines(
            draw,
            [
                (f"Colony: {index}/{len(pheromone_history) - 1}", (25, 25, 30)),
                ("Gray: ant routes", (105, 105, 115)),
                ("Orange: pheromone", (205, 115, 35)),
                ("Blue: best route", (30, 85, 185)),
                ("", (0, 0, 0)),
                ("Mechanism", (25, 25, 30)),
                ("1. Ants sample routes", (70, 70, 78)),
                ("2. Good edges deposit", (70, 70, 78)),
                ("3. Pheromone evaporates", (70, 70, 78)),
                ("4. A corridor emerges", (70, 70, 78)),
            ],
        )
        frames.append(image)

    _save_gif(frames, output_path, fps)


def save_ga_evolution_gif(
    result: Any,
    scenario: Scenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    fps: int = 4,
) -> None:
    """Visualize generational replacement, elites, and offspring diversity."""

    populations = np.asarray(result.population_history, dtype=float)
    fitness_history = np.asarray(result.population_fitness_history, dtype=float)
    indices = _history_indices(len(populations), max_frames)
    frames: list[Image.Image] = []

    for frame_number, index in enumerate(indices):
        image, draw = _new_frame("GA: selection, crossover, mutation, and elitism", scenario)
        population = populations[index]
        fitness = fitness_history[index]
        ranked = np.argsort(fitness)

        offspring_indices = [i for i in ranked[:28] if index == 0 or i >= result.elite_count]
        for chromosome_index in offspring_indices[:24]:
            path = vector_to_path(
                population[chromosome_index],
                start=scenario.start_array,
                goal=scenario.goal_array,
                num_waypoints=scenario.num_waypoints,
            )
            _draw_path(draw, path, scenario, fill=(170, 145, 205), width=1)

        if index > 0:
            for chromosome_index in range(min(result.elite_count, len(population))):
                elite_path = vector_to_path(
                    population[chromosome_index],
                    start=scenario.start_array,
                    goal=scenario.goal_array,
                    num_waypoints=scenario.num_waypoints,
                )
                _draw_path(draw, elite_path, scenario, fill=(225, 145, 45), width=3)

        best_path = vector_to_path(
            result.best_vector_history[index],
            start=scenario.start_array,
            goal=scenario.goal_array,
            num_waypoints=scenario.num_waypoints,
        )
        _draw_path(draw, best_path, scenario, fill=(35, 135, 75), width=5, nodes=True)

        _panel_lines(
            draw,
            [
                (f"Generation: {index}/{len(populations) - 1}", (25, 25, 30)),
                ("Purple: offspring", (145, 105, 185)),
                ("Orange: elites", (205, 120, 35)),
                ("Green: best-so-far", (35, 135, 75)),
                ("", (0, 0, 0)),
                ("Generation cycle", (25, 25, 30)),
                ("Tournament selection", (70, 70, 78)),
                ("       down", (115, 115, 120)),
                ("Waypoint crossover", (70, 70, 78)),
                ("       down", (115, 115, 120)),
                ("Gaussian mutation", (70, 70, 78)),
                ("       down", (115, 115, 120)),
                ("Elite survival", (70, 70, 78)),
            ],
        )
        frames.append(image)

    _save_gif(frames, output_path, fps)


def save_pso_evolution_gif(
    result: Any,
    scenario: Scenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    fps: int = 4,
) -> None:
    """Visualize particles moving under personal-best and global-best attraction."""

    populations = np.asarray(result.population_history, dtype=float)
    personal_bests = np.asarray(result.personal_best_history, dtype=float)
    indices = _history_indices(len(populations), max_frames)
    frames: list[Image.Image] = []

    for index in indices:
        image, draw = _new_frame("PSO: particles attracted by pbest and gbest", scenario)
        population = populations[index]
        pbest_population = personal_bests[index]
        display_indices = np.linspace(0, len(population) - 1, 24, dtype=int)

        for particle_index in display_indices:
            path = vector_to_path(
                population[particle_index],
                start=scenario.start_array,
                goal=scenario.goal_array,
                num_waypoints=scenario.num_waypoints,
            )
            _draw_path(draw, path, scenario, fill=(145, 185, 220), width=1)

        gbest_path = vector_to_path(
            result.best_vector_history[index],
            start=scenario.start_array,
            goal=scenario.goal_array,
            num_waypoints=scenario.num_waypoints,
        )
        _draw_path(draw, gbest_path, scenario, fill=(205, 55, 55), width=5, nodes=True)

        central_waypoint = scenario.num_waypoints // 2
        arrow_indices = np.linspace(1, len(population) - 1, 6, dtype=int)
        gbest_points = result.best_vector_history[index].reshape(scenario.num_waypoints, 2)
        gbest_pixel = _to_pixel(gbest_points[central_waypoint], scenario)
        for particle_index in arrow_indices:
            current_points = population[particle_index].reshape(scenario.num_waypoints, 2)
            pbest_points = pbest_population[particle_index].reshape(
                scenario.num_waypoints, 2
            )
            current_pixel = _to_pixel(current_points[central_waypoint], scenario)
            pbest_pixel = _to_pixel(pbest_points[central_waypoint], scenario)
            draw.ellipse(
                (
                    current_pixel[0] - 3,
                    current_pixel[1] - 3,
                    current_pixel[0] + 3,
                    current_pixel[1] + 3,
                ),
                fill=(40, 120, 190),
            )
            _draw_dashed_line(
                draw,
                current_pixel,
                pbest_pixel,
                fill=(225, 150, 40),
                width=2,
            )
            _draw_arrow(
                draw,
                current_pixel,
                gbest_pixel,
                fill=(205, 55, 55),
                width=2,
            )

        _panel_lines(
            draw,
            [
                (f"Iteration: {index}/{len(populations) - 1}", (25, 25, 30)),
                ("Blue: particles", (65, 125, 185)),
                ("Yellow dash: pbest", (205, 130, 25)),
                ("Red arrows: gbest", (190, 45, 45)),
                ("Red path: global best", (190, 45, 45)),
                ("", (0, 0, 0)),
                ("Velocity update", (25, 25, 30)),
                ("inertia", (70, 70, 78)),
                ("+ cognitive pull", (70, 70, 78)),
                ("+ social pull", (70, 70, 78)),
                ("", (0, 0, 0)),
                ("Swarm gathers around", (70, 70, 78)),
                ("the shared best path.", (70, 70, 78)),
            ],
        )
        frames.append(image)

    _save_gif(frames, output_path, fps)


def save_gwo_evolution_gif(
    result: Any,
    scenario: Scenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    fps: int = 4,
) -> None:
    """Visualize omega wolves encircling alpha, beta, and delta leaders."""

    populations = np.asarray(result.population_history, dtype=float)
    leader_history = np.asarray(result.leader_vector_history, dtype=float)
    indices = _history_indices(len(populations), max_frames)
    frames: list[Image.Image] = []
    leader_colors = [(205, 55, 55), (45, 105, 200), (45, 150, 85)]

    for index in indices:
        image, draw = _new_frame("GWO: wolves guided by Alpha, Beta, and Delta", scenario)
        population = populations[index]
        leaders = leader_history[index]
        display_indices = np.linspace(0, len(population) - 1, 26, dtype=int)

        for wolf_index in display_indices:
            path = vector_to_path(
                population[wolf_index],
                start=scenario.start_array,
                goal=scenario.goal_array,
                num_waypoints=scenario.num_waypoints,
            )
            _draw_path(draw, path, scenario, fill=(165, 165, 175), width=1)

        leader_paths = []
        for leader_vector, color in zip(leaders, leader_colors):
            leader_path = vector_to_path(
                leader_vector,
                start=scenario.start_array,
                goal=scenario.goal_array,
                num_waypoints=scenario.num_waypoints,
            )
            leader_paths.append(leader_path)
            _draw_path(draw, leader_path, scenario, fill=color, width=4, nodes=True)

        central_waypoint = scenario.num_waypoints // 2 + 1
        omega_indices = np.linspace(3, len(population) - 1, 4, dtype=int)
        leader_pixels = [
            _to_pixel(path[central_waypoint], scenario) for path in leader_paths
        ]
        for wolf_index in omega_indices:
            wolf_path = vector_to_path(
                population[wolf_index],
                start=scenario.start_array,
                goal=scenario.goal_array,
                num_waypoints=scenario.num_waypoints,
            )
            source = _to_pixel(wolf_path[central_waypoint], scenario)
            for target, color in zip(leader_pixels, leader_colors):
                _draw_dashed_line(draw, source, target, fill=color, width=1, dash=5)

        _panel_lines(
            draw,
            [
                (f"Iteration: {index}/{len(populations) - 1}", (25, 25, 30)),
                ("Gray: omega wolves", (110, 110, 120)),
                ("Red: Alpha", leader_colors[0]),
                ("Blue: Beta", leader_colors[1]),
                ("Green: Delta", leader_colors[2]),
                ("", (0, 0, 0)),
                ("Leader hierarchy", (25, 25, 30)),
                ("Each wolf estimates", (70, 70, 78)),
                ("three target positions", (70, 70, 78)),
                ("and moves toward their", (70, 70, 78)),
                ("combined guidance.", (70, 70, 78)),
                ("", (0, 0, 0)),
                ("The pack encircles", (70, 70, 78)),
                ("the promising region.", (70, 70, 78)),
            ],
        )
        frames.append(image)

    _save_gif(frames, output_path, fps)
