"""Static and animated visualizations for multi-agent joint plans."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from config.scenario import MultiAgentScenario
from .path_utils import JointPlan, decode_joint_vector, sample_joint_trajectories


def plot_scenario(ax: plt.Axes, scenario: MultiAgentScenario) -> None:
    """Draw the shared map, obstacles, safety margins, starts, and goals."""

    ax.set_xlim(0.0, scenario.width)
    ax.set_ylim(0.0, scenario.height)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    margin = scenario.boundary_safety_margin
    if margin > 0.0:
        boundary_margin = plt.Rectangle(
            (margin, margin),
            scenario.width - 2.0 * margin,
            scenario.height - 2.0 * margin,
            fill=False,
            linestyle=":",
            linewidth=1.5,
            color="black",
            alpha=0.65,
            label="Boundary margin",
        )
        ax.add_patch(boundary_margin)

    for index, obstacle in enumerate(scenario.obstacles):
        physical = plt.Circle(
            obstacle.center,
            obstacle.radius,
            alpha=0.42,
            color="gray",
            label="Obstacle" if index == 0 else None,
        )
        safety = plt.Circle(
            obstacle.center,
            obstacle.radius + scenario.obstacle_safety_margin,
            fill=False,
            linestyle="--",
            color="gray",
            alpha=0.7,
            label="Obstacle margin" if index == 0 else None,
        )
        ax.add_patch(physical)
        ax.add_patch(safety)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, task in enumerate(scenario.tasks):
        color = colors[index % len(colors)]
        ax.scatter(*task.start, marker="o", s=75, color=color, zorder=5)
        ax.scatter(*task.goal, marker="*", s=150, color=color, zorder=5)
        ax.text(task.start[0] + 1.0, task.start[1] + 1.0, f"S{index + 1}")
        ax.text(task.goal[0] + 1.0, task.goal[1] + 1.0, f"G{index + 1}")


def save_joint_plan_figure(
    plan: JointPlan,
    scenario: MultiAgentScenario,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Save all agent paths over the common scenario."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 8))
    plot_scenario(axis, scenario)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, path in enumerate(plan.paths):
        axis.plot(
            path[:, 0],
            path[:, 1],
            "-o",
            linewidth=2,
            markersize=3,
            color=colors[index % len(colors)],
            label=(
                f"{scenario.tasks[index].name} | "
                f"delay={plan.start_delays[index]:.1f}s"
            ),
        )
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pil_to_pixel(
    point: np.ndarray | tuple[float, float],
    scenario: MultiAgentScenario,
    *,
    left: int = 50,
    top: int = 60,
    map_size: int = 520,
) -> tuple[int, int]:
    x, y = float(point[0]), float(point[1])
    px = left + int(round((x / scenario.width) * map_size))
    py = top + map_size - int(round((y / scenario.height) * map_size))
    return px, py


def _pil_radius(
    radius: float,
    scenario: MultiAgentScenario,
    *,
    map_size: int = 520,
) -> int:
    scale = min(map_size / scenario.width, map_size / scenario.height)
    return max(1, int(round(radius * scale)))


def _draw_dashed_circle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
    width: int = 2,
) -> None:
    for angle in range(0, 360, 18):
        draw.arc(box, angle, angle + 9, fill=fill, width=width)


def _draw_pil_scenario(
    draw: ImageDraw.ImageDraw,
    scenario: MultiAgentScenario,
    *,
    left: int = 50,
    top: int = 60,
    map_size: int = 520,
) -> None:
    draw.rectangle(
        (left, top, left + map_size, top + map_size),
        fill=(250, 250, 250),
        outline=(60, 60, 60),
        width=2,
    )
    margin = scenario.boundary_safety_margin
    if margin > 0.0:
        lower_left = _pil_to_pixel(
            (margin, margin), scenario, left=left, top=top, map_size=map_size
        )
        upper_right = _pil_to_pixel(
            (scenario.width - margin, scenario.height - margin),
            scenario,
            left=left,
            top=top,
            map_size=map_size,
        )
        box = (
            lower_left[0],
            upper_right[1],
            upper_right[0],
            lower_left[1],
        )
        dash = 8
        x0, y0, x1, y1 = box
        for x in range(x0, x1, 2 * dash):
            draw.line((x, y0, min(x + dash, x1), y0), fill=(95, 95, 95), width=1)
            draw.line((x, y1, min(x + dash, x1), y1), fill=(95, 95, 95), width=1)
        for y in range(y0, y1, 2 * dash):
            draw.line((x0, y, x0, min(y + dash, y1)), fill=(95, 95, 95), width=1)
            draw.line((x1, y, x1, min(y + dash, y1)), fill=(95, 95, 95), width=1)
    for obstacle in scenario.obstacles:
        cx, cy = _pil_to_pixel(
            obstacle.center, scenario, left=left, top=top, map_size=map_size
        )
        physical = _pil_radius(obstacle.radius, scenario, map_size=map_size)
        safety = _pil_radius(
            obstacle.radius + scenario.obstacle_safety_margin,
            scenario,
            map_size=map_size,
        )
        _draw_dashed_circle(
            draw,
            (cx - safety, cy - safety, cx + safety, cy + safety),
            fill=(125, 125, 125),
        )
        draw.ellipse(
            (cx - physical, cy - physical, cx + physical, cy + physical),
            fill=(185, 185, 185),
            outline=(75, 75, 75),
            width=2,
        )


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    path: np.ndarray,
    scenario: MultiAgentScenario,
    *,
    fill: tuple[int, int, int],
    width: int,
    left: int = 50,
    top: int = 60,
    map_size: int = 520,
) -> None:
    points = [
        _pil_to_pixel(point, scenario, left=left, top=top, map_size=map_size)
        for point in np.asarray(path, dtype=float)
    ]
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width, joint="curve")


def save_joint_motion_gif(
    plan: JointPlan,
    scenario: MultiAgentScenario,
    output_path: str | Path,
    *,
    title: str = "Centralized Multi-Agent Plan",
    max_frames: int = 60,
    fps: int = 8,
) -> None:
    """Animate synchronized motion with a fast Pillow renderer."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    times, trajectories = sample_joint_trajectories(plan, scenario)
    frame_indices = np.unique(
        np.linspace(0, len(times) - 1, min(max_frames, len(times)), dtype=int)
    )
    colors = ((35, 110, 190), (225, 120, 35), (40, 155, 85))
    pale = ((155, 190, 225), (235, 190, 145), (155, 210, 175))
    frames: list[Image.Image] = []

    for sample_index in frame_indices:
        image = Image.new("RGB", (620, 630), (244, 246, 248))
        draw = ImageDraw.Draw(image)
        draw.text((50, 18), title, fill=(20, 20, 25))
        draw.text(
            (455, 20), f"time={times[sample_index]:.1f}s", fill=(30, 30, 35)
        )
        _draw_pil_scenario(draw, scenario)

        for agent_index, path in enumerate(plan.paths):
            _draw_polyline(
                draw, path, scenario, fill=pale[agent_index], width=2
            )
            history = trajectories[agent_index, : sample_index + 1]
            _draw_polyline(
                draw, history, scenario, fill=colors[agent_index], width=4
            )
            start = _pil_to_pixel(scenario.tasks[agent_index].start, scenario)
            goal = _pil_to_pixel(scenario.tasks[agent_index].goal, scenario)
            draw.ellipse(
                (start[0] - 6, start[1] - 6, start[0] + 6, start[1] + 6),
                fill=colors[agent_index],
            )
            draw.rectangle(
                (goal[0] - 7, goal[1] - 7, goal[0] + 7, goal[1] + 7),
                outline=colors[agent_index],
                width=3,
            )
            position = _pil_to_pixel(trajectories[agent_index, sample_index], scenario)
            radius = _pil_radius(scenario.agent_radius, scenario) + 3
            draw.ellipse(
                (
                    position[0] - radius,
                    position[1] - radius,
                    position[0] + radius,
                    position[1] + radius,
                ),
                fill=colors[agent_index],
                outline=(255, 255, 255),
                width=2,
            )
            draw.text(
                (50 + agent_index * 185, 600),
                f"A{agent_index + 1}: delay={plan.start_delays[agent_index]:.1f}s",
                fill=colors[agent_index],
            )
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    duration_ms = max(40, int(round(1000 / fps)))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_fitness_convergence_figure(
    fitness_history: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Multi-Agent PSO Best-so-far Objective",
) -> None:
    """Save the best-so-far joint objective history."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(fitness_history, dtype=float)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(np.arange(len(history)), history, linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best-so-far joint objective")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_pso_search_gif(
    result,
    scenario: MultiAgentScenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    displayed_particles: int = 12,
    fps: int = 4,
) -> None:
    """Animate swarm exploration and the best joint plan with Pillow."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(result.population_history, dtype=float)
    best_history = np.asarray(result.best_vector_history, dtype=float)
    fitness_history = np.asarray(result.fitness_history, dtype=float)
    frame_indices = np.unique(
        np.linspace(0, len(history) - 1, min(max_frames, len(history)), dtype=int)
    )
    particle_indices = np.unique(
        np.linspace(
            0,
            history.shape[1] - 1,
            min(displayed_particles, history.shape[1]),
            dtype=int,
        )
    )
    colors = ((35, 110, 190), (225, 120, 35), (40, 155, 85))
    pale = ((190, 210, 230), (235, 210, 185), (185, 220, 200))
    frames: list[Image.Image] = []

    for iteration in frame_indices:
        image = Image.new("RGB", (620, 650), (244, 246, 248))
        draw = ImageDraw.Draw(image)
        draw.text((50, 16), "Multi-Agent PSO Search", fill=(20, 20, 25))
        draw.text(
            (365, 16),
            f"iteration={iteration}/{result.iterations}",
            fill=(30, 30, 35),
        )
        draw.text(
            (50, 36),
            f"best objective={fitness_history[iteration]:.2f}",
            fill=(30, 30, 35),
        )
        _draw_pil_scenario(draw, scenario)

        for particle_index in particle_indices:
            particle_plan = decode_joint_vector(
                history[iteration, particle_index], scenario
            )
            for agent_index, path in enumerate(particle_plan.paths):
                _draw_polyline(
                    draw, path, scenario, fill=pale[agent_index], width=1
                )

        best_plan = decode_joint_vector(best_history[iteration], scenario)
        for agent_index, path in enumerate(best_plan.paths):
            _draw_polyline(
                draw, path, scenario, fill=colors[agent_index], width=5
            )
            for point in path[1:-1]:
                px, py = _pil_to_pixel(point, scenario)
                draw.ellipse(
                    (px - 3, py - 3, px + 3, py + 3),
                    fill=colors[agent_index],
                )
            draw.text(
                (50 + agent_index * 185, 615),
                f"A{agent_index + 1}: delay={best_plan.start_delays[agent_index]:.1f}s",
                fill=colors[agent_index],
            )
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    duration_ms = max(50, int(round(1000 / fps)))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )



def save_gwo_search_gif(
    result,
    scenario: MultiAgentScenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    displayed_wolves: int = 10,
    fps: int = 4,
) -> None:
    """Animate Omega exploration and Alpha/Beta/Delta joint-plan guidance."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(result.population_history, dtype=float)
    leader_history = np.asarray(result.leader_vector_history, dtype=float)
    best_history = np.asarray(result.best_vector_history, dtype=float)
    fitness_history = np.asarray(result.fitness_history, dtype=float)
    frame_indices = np.unique(
        np.linspace(0, len(history) - 1, min(max_frames, len(history)), dtype=int)
    )
    wolf_indices = np.unique(
        np.linspace(
            0,
            history.shape[1] - 1,
            min(displayed_wolves, history.shape[1]),
            dtype=int,
        )
    )
    omega_color = (190, 193, 200)
    leader_colors = ((210, 55, 55), (45, 105, 205), (35, 150, 85))
    agent_colors = ((35, 110, 190), (225, 120, 35), (40, 155, 85))
    frames: list[Image.Image] = []

    for iteration in frame_indices:
        image = Image.new("RGB", (620, 675), (244, 246, 248))
        draw = ImageDraw.Draw(image)
        draw.text((50, 14), "Multi-Agent GWO Search", fill=(20, 20, 25))
        draw.text(
            (355, 14),
            f"iteration={iteration}/{result.iterations}",
            fill=(30, 30, 35),
        )
        draw.text(
            (50, 34),
            f"best objective={fitness_history[iteration]:.2f}",
            fill=(30, 30, 35),
        )
        draw.text((50, 52), "Omega", fill=omega_color)
        draw.text((125, 52), "Alpha", fill=leader_colors[0])
        draw.text((200, 52), "Beta", fill=leader_colors[1])
        draw.text((265, 52), "Delta", fill=leader_colors[2])
        _draw_pil_scenario(draw, scenario, top=78, map_size=500)

        for wolf_index in wolf_indices:
            wolf_plan = decode_joint_vector(history[iteration, wolf_index], scenario)
            for path in wolf_plan.paths:
                _draw_polyline(
                    draw,
                    path,
                    scenario,
                    fill=omega_color,
                    width=1,
                    top=78,
                    map_size=500,
                )

        for leader_index, leader_vector in enumerate(leader_history[iteration]):
            leader_plan = decode_joint_vector(leader_vector, scenario)
            for path in leader_plan.paths:
                _draw_polyline(
                    draw,
                    path,
                    scenario,
                    fill=leader_colors[leader_index],
                    width=3,
                    top=78,
                    map_size=500,
                )

        best_plan = decode_joint_vector(best_history[iteration], scenario)
        for agent_index, path in enumerate(best_plan.paths):
            for point in path[1:-1]:
                px, py = _pil_to_pixel(
                    point, scenario, top=78, map_size=500
                )
                draw.ellipse(
                    (px - 3, py - 3, px + 3, py + 3),
                    fill=agent_colors[agent_index],
                )
            draw.text(
                (50 + agent_index * 185, 615),
                f"A{agent_index + 1}: delay={best_plan.start_delays[agent_index]:.1f}s",
                fill=agent_colors[agent_index],
            )
        draw.text(
            (50, 642),
            "Alpha, Beta, and Delta guide the full 3-agent joint plan.",
            fill=(45, 45, 50),
        )
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    duration_ms = max(50, int(round(1000 / fps)))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_ga_search_gif(
    result,
    scenario: MultiAgentScenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    displayed_chromosomes: int = 12,
    fps: int = 4,
) -> None:
    """Animate population evolution, elite survival, and best-so-far plan."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    history = np.asarray(result.population_history, dtype=float)
    fitness_history = np.asarray(result.population_fitness_history, dtype=float)
    best_history = np.asarray(result.best_vector_history, dtype=float)
    best_fitness_history = np.asarray(result.fitness_history, dtype=float)
    frame_indices = np.unique(
        np.linspace(0, len(history) - 1, min(max_frames, len(history)), dtype=int)
    )
    offspring_color = (198, 180, 225)
    elite_color = (230, 135, 45)
    best_color = (30, 145, 75)
    agent_colors = ((35, 110, 190), (225, 120, 35), (40, 155, 85))
    frames: list[Image.Image] = []

    for generation in frame_indices:
        image = Image.new("RGB", (620, 675), (244, 246, 248))
        draw = ImageDraw.Draw(image)
        draw.text((50, 14), "Multi-Agent GA Evolution", fill=(20, 20, 25))
        draw.text(
            (350, 14),
            f"generation={generation}/{result.iterations}",
            fill=(30, 30, 35),
        )
        draw.text(
            (50, 34),
            f"best objective={best_fitness_history[generation]:.2f}",
            fill=(30, 30, 35),
        )
        draw.text((50, 52), "Population", fill=offspring_color)
        draw.text((145, 52), "Elites", fill=elite_color)
        draw.text((210, 52), "Best-so-far", fill=best_color)
        _draw_pil_scenario(draw, scenario, top=78, map_size=500)

        ranked = np.argsort(fitness_history[generation], kind="stable")
        elite_indices = ranked[: result.elite_count]
        non_elites = ranked[result.elite_count :]
        if len(non_elites):
            sample_positions = np.unique(
                np.linspace(
                    0,
                    len(non_elites) - 1,
                    min(displayed_chromosomes, len(non_elites)),
                    dtype=int,
                )
            )
            sampled_indices = non_elites[sample_positions]
        else:
            sampled_indices = np.empty(0, dtype=int)

        for chromosome_index in sampled_indices:
            plan = decode_joint_vector(history[generation, chromosome_index], scenario)
            for path in plan.paths:
                _draw_polyline(
                    draw,
                    path,
                    scenario,
                    fill=offspring_color,
                    width=1,
                    top=78,
                    map_size=500,
                )

        for chromosome_index in elite_indices:
            plan = decode_joint_vector(history[generation, chromosome_index], scenario)
            for path in plan.paths:
                _draw_polyline(
                    draw,
                    path,
                    scenario,
                    fill=elite_color,
                    width=2,
                    top=78,
                    map_size=500,
                )

        best_plan = decode_joint_vector(best_history[generation], scenario)
        for agent_index, path in enumerate(best_plan.paths):
            _draw_polyline(
                draw,
                path,
                scenario,
                fill=best_color,
                width=5,
                top=78,
                map_size=500,
            )
            for point in path[1:-1]:
                px, py = _pil_to_pixel(point, scenario, top=78, map_size=500)
                draw.ellipse(
                    (px - 3, py - 3, px + 3, py + 3),
                    fill=agent_colors[agent_index],
                )
            draw.text(
                (50 + agent_index * 185, 615),
                f"A{agent_index + 1}: delay={best_plan.start_delays[agent_index]:.1f}s",
                fill=agent_colors[agent_index],
            )
        draw.text(
            (50, 642),
            "Selection, structured crossover, mutation, and elite survival.",
            fill=(45, 45, 50),
        )
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    duration_ms = max(50, int(round(1000 / fps)))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def save_aco_search_gif(
    result,
    scenario: MultiAgentScenario,
    output_path: str | Path,
    *,
    max_frames: int = 18,
    displayed_joint_ants: int = 6,
    fps: int = 4,
) -> None:
    """Animate agent-specific pheromone fields and joint-ant plans."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pheromone_history = np.asarray(result.pheromone_history, dtype=float)
    fitness_history = np.asarray(result.fitness_history, dtype=float)
    frame_indices = np.unique(
        np.linspace(
            0,
            len(pheromone_history) - 1,
            min(max_frames, len(pheromone_history)),
            dtype=int,
        )
    )
    agent_colors = ((35, 110, 190), (225, 120, 35), (40, 155, 85))
    pale_colors = ((175, 205, 232), (238, 205, 170), (175, 220, 195))
    frames: list[Image.Image] = []

    for iteration in frame_indices:
        image = Image.new("RGB", (620, 680), (244, 246, 248))
        draw = ImageDraw.Draw(image)
        draw.text((50, 14), "Multi-Agent ACO", fill=(20, 20, 25))
        draw.text(
            (360, 14),
            f"colony={iteration}/{result.iterations}",
            fill=(30, 30, 35),
        )
        draw.text(
            (50, 34),
            f"best objective={fitness_history[iteration]:.2f}",
            fill=(30, 30, 35),
        )
        draw.text((50, 52), "Pheromone + reservation-aware joint ants", fill=(65, 65, 70))
        _draw_pil_scenario(draw, scenario, top=82, map_size=500)

        pheromone = pheromone_history[iteration]
        for agent_index in range(min(scenario.num_agents, pheromone.shape[0])):
            values = pheromone[agent_index]
            positive = values[values > 0.0]
            if positive.size == 0:
                continue
            transformed = np.log1p(values)
            scale = float(np.percentile(np.log1p(positive), 97))
            scale = max(scale, 1e-12)
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    strength = min(1.0, float(transformed[row, col]) / scale)
                    if strength < 0.22:
                        continue
                    point = (
                        col * result.grid_resolution,
                        row * result.grid_resolution,
                    )
                    px, py = _pil_to_pixel(
                        point, scenario, top=82, map_size=500
                    )
                    offset = agent_index - 1
                    radius = 1 + int(round(2 * strength))
                    color = agent_colors[agent_index]
                    draw.ellipse(
                        (
                            px - radius + 2 * offset,
                            py - radius,
                            px + radius + 2 * offset,
                            py + radius,
                        ),
                        fill=color,
                    )

        colony_plans = result.colony_plan_history[iteration]
        for plan in colony_plans[:displayed_joint_ants]:
            for agent_index, path in enumerate(plan.paths):
                _draw_polyline(
                    draw,
                    path,
                    scenario,
                    fill=pale_colors[agent_index],
                    width=1,
                    top=82,
                    map_size=500,
                )

        best_plan = result.best_plan_history[iteration]
        for agent_index, path in enumerate(best_plan.paths):
            _draw_polyline(
                draw,
                path,
                scenario,
                fill=agent_colors[agent_index],
                width=5,
                top=82,
                map_size=500,
            )
            draw.text(
                (50 + agent_index * 185, 620),
                f"A{agent_index + 1}: delay={best_plan.start_delays[agent_index]:.1f}s",
                fill=agent_colors[agent_index],
            )
        draw.text(
            (50, 648),
            "Earlier paths reserve space-time; good routes reinforce pheromone.",
            fill=(45, 45, 50),
        )
        frames.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    duration_ms = max(50, int(round(1000 / fps)))
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
