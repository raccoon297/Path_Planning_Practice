"""Shared objective for centralized multi-agent path planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from config.scenario import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    MultiAgentObjectiveWeights,
    MultiAgentScenario,
)
from .collision import (
    boundary_safety_penalty,
    boundary_violation_amount,
    inter_agent_separation_report,
    segment_obstacle_clearances,
)
from .path_utils import (
    JointPlan,
    backtracking_cost,
    completion_times,
    decode_joint_vector,
    path_length,
    sample_joint_trajectories,
    smoothness_cost,
    waypoint_spacing_cost,
)


@dataclass(frozen=True)
class JointObjectiveResult:
    total: float
    length: float
    obstacle_collision: float
    obstacle_clearance: float
    inter_agent_collision: float
    inter_agent_clearance: float
    smoothness: float
    backtracking: float
    waypoint_spacing: float
    boundary: float
    start_delay: float
    makespan: float
    obstacle_collision_count: int
    obstacle_safety_violation_count: int
    inter_agent_collision_episode_count: int
    inter_agent_separation_violation_episode_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_joint_plan(
    plan: JointPlan,
    scenario: MultiAgentScenario,
    weights: MultiAgentObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    *,
    include_waypoint_regularization: bool = False,
) -> JointObjectiveResult:
    """Evaluate paths and departure times as one centralized joint decision."""

    length_raw = 0.0
    obstacle_collision_count = 0
    obstacle_safety_violation_count = 0
    obstacle_clearance_shortfall = 0.0
    smoothness_raw = 0.0
    backtracking_raw = 0.0
    waypoint_spacing_raw = 0.0
    boundary_raw = 0.0

    for path in plan.paths:
        length_raw += path_length(path)
        smoothness_raw += smoothness_cost(path)
        if include_waypoint_regularization:
            backtracking_raw += backtracking_cost(path)
            waypoint_spacing_raw += waypoint_spacing_cost(path)
        boundary_raw += boundary_violation_amount(
            path, scenario.width, scenario.height
        )
        boundary_raw += boundary_safety_penalty(
            path,
            scenario.width,
            scenario.height,
            scenario.boundary_safety_margin,
        )
        clearances = segment_obstacle_clearances(path, scenario.obstacles)
        if clearances.size:
            obstacle_collision_count += int(np.count_nonzero(clearances <= 0.0))
            shortfalls = np.clip(
                scenario.obstacle_safety_margin - clearances, 0.0, None
            )
            obstacle_safety_violation_count += int(np.count_nonzero(shortfalls > 0.0))
            obstacle_clearance_shortfall += float(
                np.count_nonzero(shortfalls > 0.0) + np.square(shortfalls).sum()
            )

    _, trajectories = sample_joint_trajectories(plan, scenario)
    separation = inter_agent_separation_report(
        trajectories,
        physical_collision_distance=scenario.physical_agent_collision_distance,
        minimum_separation=scenario.minimum_agent_separation,
        time_step=scenario.time_step,
    )

    inter_collision_raw = float(
        separation.physical_collision_episode_count
        + separation.physical_overlap_integral
    )
    inter_clearance_raw = float(
        separation.separation_violation_episode_count
        + separation.separation_shortfall_integral
    )
    delay_raw = float(plan.start_delays.sum())
    makespan_raw = float(completion_times(plan, scenario).max())

    weighted = {
        "length": weights.length * length_raw,
        "obstacle_collision": weights.obstacle_collision * obstacle_collision_count,
        "obstacle_clearance": weights.obstacle_clearance
        * obstacle_clearance_shortfall,
        "inter_agent_collision": weights.inter_agent_collision
        * inter_collision_raw,
        "inter_agent_clearance": weights.inter_agent_clearance
        * inter_clearance_raw,
        "smoothness": weights.smoothness * smoothness_raw,
        "backtracking": weights.backtracking * backtracking_raw,
        "waypoint_spacing": weights.waypoint_spacing * waypoint_spacing_raw,
        "boundary": weights.boundary * boundary_raw,
        "start_delay": weights.start_delay * delay_raw,
        "makespan": weights.makespan * makespan_raw,
    }
    return JointObjectiveResult(
        total=float(sum(weighted.values())),
        length=float(weighted["length"]),
        obstacle_collision=float(weighted["obstacle_collision"]),
        obstacle_clearance=float(weighted["obstacle_clearance"]),
        inter_agent_collision=float(weighted["inter_agent_collision"]),
        inter_agent_clearance=float(weighted["inter_agent_clearance"]),
        smoothness=float(weighted["smoothness"]),
        backtracking=float(weighted["backtracking"]),
        waypoint_spacing=float(weighted["waypoint_spacing"]),
        boundary=float(weighted["boundary"]),
        start_delay=float(weighted["start_delay"]),
        makespan=float(weighted["makespan"]),
        obstacle_collision_count=obstacle_collision_count,
        obstacle_safety_violation_count=obstacle_safety_violation_count,
        inter_agent_collision_episode_count=(
            separation.physical_collision_episode_count
        ),
        inter_agent_separation_violation_episode_count=(
            separation.separation_violation_episode_count
        ),
    )


def evaluate_joint_vector(
    vector: np.ndarray,
    scenario: MultiAgentScenario,
    weights: MultiAgentObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
) -> JointObjectiveResult:
    """Decode and evaluate one optimizer candidate vector."""

    return evaluate_joint_plan(
        decode_joint_vector(vector, scenario),
        scenario,
        weights,
        include_waypoint_regularization=True,
    )
