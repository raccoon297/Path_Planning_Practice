"""Serializable metadata shared by algorithm reports and comparisons."""

from __future__ import annotations

from dataclasses import asdict

from config.scenario import MultiAgentObjectiveWeights, MultiAgentScenario


def scenario_as_dict(scenario: MultiAgentScenario) -> dict[str, object]:
    """Return a JSON-serializable description of one planning scenario."""

    return {
        "width": scenario.width,
        "height": scenario.height,
        "tasks": [
            {
                "name": task.name,
                "start": list(task.start),
                "goal": list(task.goal),
            }
            for task in scenario.tasks
        ],
        "obstacles": [
            {
                "center": list(obstacle.center),
                "radius": obstacle.radius,
            }
            for obstacle in scenario.obstacles
        ],
        "obstacle_safety_margin": scenario.obstacle_safety_margin,
        "boundary_safety_margin": scenario.boundary_safety_margin,
        "num_waypoints": scenario.num_waypoints,
        "speed": scenario.speed,
        "agent_radius": scenario.agent_radius,
        "minimum_agent_separation": scenario.minimum_agent_separation,
        "max_start_delay": scenario.max_start_delay,
        "time_step": scenario.time_step,
        "continuous_dimension": scenario.dimension,
    }


def objective_weights_as_dict(
    weights: MultiAgentObjectiveWeights,
) -> dict[str, float]:
    """Return objective weights in a stable JSON representation."""

    return {key: float(value) for key, value in asdict(weights).items()}
