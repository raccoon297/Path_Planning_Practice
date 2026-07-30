"""Path representation and common time-parameterization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config.scenario import MultiAgentScenario


@dataclass(frozen=True)
class JointPlan:
    """Complete paths and normalized start delays for all agents."""

    paths: tuple[np.ndarray, ...]
    start_delays: np.ndarray

    def __post_init__(self) -> None:
        if len(self.paths) != len(self.start_delays):
            raise ValueError("The number of paths and delays must match.")


def ensure_path_array(path: np.ndarray) -> np.ndarray:
    """Validate and return a floating-point path with shape ``(N, 2)``."""

    array = np.asarray(path, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Path must have shape (N, 2).")
    if len(array) < 2:
        raise ValueError("Path must contain at least two points.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Path contains NaN or infinite values.")
    return array


def path_length(path: np.ndarray) -> float:
    """Return total Euclidean length of a polyline."""

    array = ensure_path_array(path)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def turning_angles(path: np.ndarray) -> np.ndarray:
    """Return unsigned turning angles in radians for interior path points."""

    array = ensure_path_array(path)
    if len(array) < 3:
        return np.empty(0, dtype=float)

    previous = array[1:-1] - array[:-2]
    following = array[2:] - array[1:-1]
    previous_norm = np.linalg.norm(previous, axis=1)
    following_norm = np.linalg.norm(following, axis=1)
    valid = (previous_norm > 1e-12) & (following_norm > 1e-12)

    angles = np.full(len(previous), np.pi, dtype=float)
    if np.any(valid):
        cosine = np.sum(previous[valid] * following[valid], axis=1)
        cosine /= previous_norm[valid] * following_norm[valid]
        angles[valid] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return angles


def smoothness_cost(path: np.ndarray) -> float:
    """Return the sum of squared turning angles."""

    return float(np.square(turning_angles(path)).sum())


def backtracking_cost(path: np.ndarray) -> float:
    """Penalize motion opposite to the path's start-to-goal direction.

    The raw backward progress is squared and normalized by the direct
    start-to-goal distance. A monotonic path therefore has zero cost, while
    short loops and explicit reversals receive a positive cost.
    """

    array = ensure_path_array(path)
    direct = array[-1] - array[0]
    direct_length = float(np.linalg.norm(direct))
    if direct_length <= 1e-12:
        return 0.0
    direction = direct / direct_length
    progress = np.diff(array, axis=0) @ direction
    backward = np.clip(-progress, 0.0, None)
    return float(np.square(backward).sum() / direct_length)


def waypoint_spacing_cost(path: np.ndarray) -> float:
    """Measure how unevenly intermediate points cover start-goal progress.

    Each intermediate point is projected onto the start-to-goal axis and
    compared with an evenly spaced reference fraction. Lateral detours remain
    possible, but waypoint clustering near the start or goal is discouraged.
    """

    array = ensure_path_array(path)
    if len(array) <= 2:
        return 0.0
    direct = array[-1] - array[0]
    squared_length = float(np.dot(direct, direct))
    if squared_length <= 1e-12:
        return 0.0
    fractions = ((array[1:-1] - array[0]) @ direct) / squared_length
    ideal = np.linspace(0.0, 1.0, len(array))[1:-1]
    return float(np.square(fractions - ideal).sum())


def normalize_start_delays(
    delays: np.ndarray, max_start_delay: float
) -> np.ndarray:
    """Clip delays and remove their common offset so one agent starts at zero."""

    array = np.asarray(delays, dtype=float).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Start delays contain NaN or infinite values.")
    clipped = np.clip(array, 0.0, max_start_delay)
    normalized = clipped - float(clipped.min())
    return np.clip(normalized, 0.0, max_start_delay)


def encode_joint_plan(
    intermediate_waypoints: Sequence[np.ndarray],
    start_delays: np.ndarray,
    scenario: MultiAgentScenario,
) -> np.ndarray:
    """Encode per-agent waypoint arrays and delays into one candidate vector."""

    if len(intermediate_waypoints) != scenario.num_agents:
        raise ValueError("Waypoint groups must match scenario.num_agents.")
    delays = normalize_start_delays(start_delays, scenario.max_start_delay)
    blocks: list[np.ndarray] = []
    for index, waypoints in enumerate(intermediate_waypoints):
        array = np.asarray(waypoints, dtype=float)
        expected_shape = (scenario.num_waypoints, 2)
        if array.shape != expected_shape:
            raise ValueError(
                f"Agent {index} waypoints must have shape {expected_shape}."
            )
        blocks.append(np.concatenate([array.reshape(-1), [delays[index]]]))
    return np.concatenate(blocks)


def decode_joint_vector(
    vector: np.ndarray,
    scenario: MultiAgentScenario,
    *,
    normalize_delays: bool = True,
) -> JointPlan:
    """Decode a complete joint-plan candidate vector."""

    flat = np.asarray(vector, dtype=float).reshape(-1)
    if flat.size != scenario.dimension:
        raise ValueError(
            f"Expected joint vector of size {scenario.dimension}, got {flat.size}."
        )
    if not np.all(np.isfinite(flat)):
        raise ValueError("Joint vector contains NaN or infinite values.")

    paths: list[np.ndarray] = []
    delays: list[float] = []
    block_size = scenario.agent_block_dimension
    waypoint_values = 2 * scenario.num_waypoints
    for index, task in enumerate(scenario.tasks):
        block = flat[index * block_size : (index + 1) * block_size]
        waypoints = block[:waypoint_values].reshape(scenario.num_waypoints, 2)
        path = np.vstack([task.start_array, waypoints, task.goal_array])
        paths.append(path)
        delays.append(float(block[-1]))

    delay_array = np.asarray(delays, dtype=float)
    if normalize_delays:
        delay_array = normalize_start_delays(
            delay_array, scenario.max_start_delay
        )
    return JointPlan(paths=tuple(paths), start_delays=delay_array)


def candidate_bounds(scenario: MultiAgentScenario) -> tuple[np.ndarray, np.ndarray]:
    """Alias for the scenario candidate bounds used by optimizers."""

    return scenario.candidate_bounds()


def completion_times(
    plan: JointPlan, scenario: MultiAgentScenario
) -> np.ndarray:
    """Return each agent's goal-arrival time under constant path speed."""

    lengths = np.asarray([path_length(path) for path in plan.paths], dtype=float)
    return plan.start_delays + lengths / scenario.speed


def build_common_time_axis(
    plan: JointPlan, scenario: MultiAgentScenario
) -> np.ndarray:
    """Create one time axis covering every agent until all have arrived."""

    makespan = float(completion_times(plan, scenario).max())
    step_count = int(np.ceil(makespan / scenario.time_step))
    return np.linspace(0.0, step_count * scenario.time_step, step_count + 1)


def sample_path_trajectory(
    path: np.ndarray,
    start_delay: float,
    speed: float,
    times: np.ndarray,
) -> np.ndarray:
    """Sample a constant-speed polyline trajectory on the supplied time axis."""

    array = ensure_path_array(path)
    time_array = np.asarray(times, dtype=float).reshape(-1)
    if speed <= 0.0:
        raise ValueError("speed must be positive.")

    segment_lengths = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    travelled = np.clip((time_array - start_delay) * speed, 0.0, cumulative[-1])

    indices = np.searchsorted(cumulative, travelled, side="right") - 1
    indices = np.clip(indices, 0, len(segment_lengths) - 1)
    denominators = segment_lengths[indices]
    fractions = np.divide(
        travelled - cumulative[indices],
        denominators,
        out=np.zeros_like(travelled),
        where=denominators > 1e-12,
    )
    return (
        array[indices] * (1.0 - fractions[:, None])
        + array[indices + 1] * fractions[:, None]
    )


def sample_joint_trajectories(
    plan: JointPlan,
    scenario: MultiAgentScenario,
    *,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return common times and trajectories with shape ``(A, T, 2)``."""

    if len(plan.paths) != scenario.num_agents:
        raise ValueError("Joint plan does not match scenario.num_agents.")
    time_axis = build_common_time_axis(plan, scenario) if times is None else np.asarray(times)
    trajectories = np.stack(
        [
            sample_path_trajectory(
                path,
                float(plan.start_delays[index]),
                scenario.speed,
                time_axis,
            )
            for index, path in enumerate(plan.paths)
        ],
        axis=0,
    )
    return time_axis, trajectories


def simplify_path_line_of_sight(
    path: np.ndarray,
    obstacles,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Remove unnecessary grid points while preserving obstacle clearance."""

    if margin < 0.0:
        raise ValueError("margin cannot be negative.")

    from .collision import segment_circle_clearance

    array = ensure_path_array(path)
    if len(array) <= 2:
        return array.copy()

    simplified = [array[0]]
    anchor_index = 0
    final_index = len(array) - 1

    while anchor_index < final_index:
        next_index = final_index
        while next_index > anchor_index + 1:
            start = array[anchor_index]
            end = array[next_index]
            edge_is_free = all(
                segment_circle_clearance(start, end, obstacle) > margin
                for obstacle in obstacles
            )
            if edge_is_free:
                break
            next_index -= 1
        simplified.append(array[next_index])
        anchor_index = next_index

    return np.asarray(simplified, dtype=float)
