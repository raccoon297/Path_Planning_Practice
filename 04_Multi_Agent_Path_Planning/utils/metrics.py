"""Human-readable metrics for a complete multi-agent joint plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config.scenario import MultiAgentScenario
from .collision import (
    boundary_safety_violation_count,
    boundary_violation_amount,
    inter_agent_separation_report,
    minimum_boundary_clearance,
    segment_obstacle_clearances,
)
from .path_utils import (
    JointPlan,
    backtracking_cost,
    completion_times,
    path_length,
    sample_joint_trajectories,
    smoothness_cost,
    waypoint_spacing_cost,
)


@dataclass(frozen=True)
class JointPlanMetrics:
    success: bool
    obstacle_collision_free: bool
    obstacle_margin_satisfied: bool
    inter_agent_collision_free: bool
    inter_agent_separation_satisfied: bool
    within_bounds: bool
    boundary_margin_satisfied: bool
    endpoints_valid: bool
    total_path_length: float
    makespan: float
    sum_start_delay: float
    minimum_obstacle_clearance: float
    minimum_boundary_clearance: float
    minimum_inter_agent_distance: float
    obstacle_collision_count: int
    obstacle_safety_violation_count: int
    boundary_safety_violation_count: int
    inter_agent_collision_episodes: int
    inter_agent_separation_violation_episodes: int
    total_smoothness: float
    total_backtracking: float
    waypoint_spacing_imbalance: float

    def as_dict(self) -> dict[str, bool | float | int]:
        return asdict(self)


def compute_joint_plan_metrics(
    plan: JointPlan, scenario: MultiAgentScenario
) -> JointPlanMetrics:
    """Compute common feasibility and quality metrics for a joint plan."""

    if len(plan.paths) != scenario.num_agents:
        raise ValueError("Joint plan does not match scenario.num_agents.")

    obstacle_collision_count = 0
    obstacle_safety_violation_count = 0
    minimum_obstacle_clearance = float("inf")
    boundary_amount = 0.0
    boundary_violation_count = 0
    minimum_wall_clearance = float("inf")
    endpoint_valid = True
    total_length = 0.0
    total_smoothness = 0.0
    total_backtracking = 0.0
    waypoint_spacing_imbalance = 0.0

    for index, path in enumerate(plan.paths):
        clearances = segment_obstacle_clearances(path, scenario.obstacles)
        if clearances.size:
            obstacle_collision_count += int(np.count_nonzero(clearances <= 0.0))
            obstacle_safety_violation_count += int(
                np.count_nonzero(clearances < scenario.obstacle_safety_margin)
            )
            minimum_obstacle_clearance = min(
                minimum_obstacle_clearance, float(clearances.min())
            )
        boundary_amount += boundary_violation_amount(
            path, scenario.width, scenario.height
        )
        boundary_violation_count += boundary_safety_violation_count(
            path,
            scenario.width,
            scenario.height,
            scenario.boundary_safety_margin,
        )
        minimum_wall_clearance = min(
            minimum_wall_clearance,
            minimum_boundary_clearance(path, scenario.width, scenario.height),
        )
        endpoint_valid = endpoint_valid and bool(
            np.allclose(path[0], scenario.tasks[index].start_array, atol=1e-9)
            and np.allclose(path[-1], scenario.tasks[index].goal_array, atol=1e-9)
        )
        total_length += path_length(path)
        total_smoothness += smoothness_cost(path)
        total_backtracking += backtracking_cost(path)
        waypoint_spacing_imbalance += waypoint_spacing_cost(path)

    _, trajectories = sample_joint_trajectories(plan, scenario)
    separation = inter_agent_separation_report(
        trajectories,
        physical_collision_distance=scenario.physical_agent_collision_distance,
        minimum_separation=scenario.minimum_agent_separation,
        time_step=scenario.time_step,
    )

    obstacle_collision_free = obstacle_collision_count == 0
    obstacle_margin_satisfied = obstacle_safety_violation_count == 0
    inter_agent_collision_free = separation.physical_collision_episode_count == 0
    inter_agent_separation_satisfied = (
        separation.separation_violation_episode_count == 0
    )
    within_bounds = boundary_amount == 0.0
    boundary_margin_satisfied = boundary_violation_count == 0
    success = (
        obstacle_collision_free
        and obstacle_margin_satisfied
        and inter_agent_collision_free
        and inter_agent_separation_satisfied
        and within_bounds
        and boundary_margin_satisfied
        and endpoint_valid
    )

    arrivals = completion_times(plan, scenario)
    return JointPlanMetrics(
        success=success,
        obstacle_collision_free=obstacle_collision_free,
        obstacle_margin_satisfied=obstacle_margin_satisfied,
        inter_agent_collision_free=inter_agent_collision_free,
        inter_agent_separation_satisfied=inter_agent_separation_satisfied,
        within_bounds=within_bounds,
        boundary_margin_satisfied=boundary_margin_satisfied,
        endpoints_valid=endpoint_valid,
        total_path_length=float(total_length),
        makespan=float(arrivals.max()),
        sum_start_delay=float(plan.start_delays.sum()),
        minimum_obstacle_clearance=float(minimum_obstacle_clearance),
        minimum_boundary_clearance=float(minimum_wall_clearance),
        minimum_inter_agent_distance=float(separation.minimum_distance),
        obstacle_collision_count=obstacle_collision_count,
        obstacle_safety_violation_count=obstacle_safety_violation_count,
        boundary_safety_violation_count=boundary_violation_count,
        inter_agent_collision_episodes=separation.physical_collision_episode_count,
        inter_agent_separation_violation_episodes=(
            separation.separation_violation_episode_count
        ),
        total_smoothness=float(total_smoothness),
        total_backtracking=float(total_backtracking),
        waypoint_spacing_imbalance=float(waypoint_spacing_imbalance),
    )
