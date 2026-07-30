"""Obstacle and time-indexed inter-agent collision calculations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from config.scenario import CircleObstacle
from .path_utils import ensure_path_array


@dataclass(frozen=True)
class InterAgentSeparationReport:
    minimum_distance: float
    physical_collision_sample_count: int
    separation_violation_sample_count: int
    physical_collision_episode_count: int
    separation_violation_episode_count: int
    physical_overlap_integral: float
    separation_shortfall_integral: float


def point_to_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    """Return the shortest Euclidean distance from a point to a segment."""

    point_array = np.asarray(point, dtype=float)
    start = np.asarray(segment_start, dtype=float)
    end = np.asarray(segment_end, dtype=float)
    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1e-24:
        return float(np.linalg.norm(point_array - start))
    projection = float(np.dot(point_array - start, segment) / squared_length)
    projection = float(np.clip(projection, 0.0, 1.0))
    nearest = start + projection * segment
    return float(np.linalg.norm(point_array - nearest))


def segment_circle_clearance(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    obstacle: CircleObstacle,
) -> float:
    """Return signed clearance from a segment to a physical obstacle."""

    return (
        point_to_segment_distance(
            obstacle.center_array, segment_start, segment_end
        )
        - obstacle.radius
    )


def segment_obstacle_clearances(
    path: np.ndarray, obstacles: Sequence[CircleObstacle]
) -> np.ndarray:
    """Return signed clearance for every segment-obstacle pair."""

    array = ensure_path_array(path)
    if not obstacles:
        return np.empty((len(array) - 1, 0), dtype=float)
    clearances = np.empty((len(array) - 1, len(obstacles)), dtype=float)
    for segment_index, (start, end) in enumerate(zip(array[:-1], array[1:])):
        for obstacle_index, obstacle in enumerate(obstacles):
            clearances[segment_index, obstacle_index] = segment_circle_clearance(
                start, end, obstacle
            )
    return clearances


def boundary_violation_amount(path: np.ndarray, width: float, height: float) -> float:
    """Return squared distance of out-of-map coordinates from valid bounds."""

    array = ensure_path_array(path)
    x = array[:, 0]
    y = array[:, 1]
    return float(
        np.square(np.clip(-x, 0.0, None)).sum()
        + np.square(np.clip(x - width, 0.0, None)).sum()
        + np.square(np.clip(-y, 0.0, None)).sum()
        + np.square(np.clip(y - height, 0.0, None)).sum()
    )


def point_boundary_clearances(
    path: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Return each path point's signed distance to the nearest map wall."""

    array = ensure_path_array(path)
    x = array[:, 0]
    y = array[:, 1]
    return np.minimum.reduce((x, width - x, y, height - y))


def minimum_boundary_clearance(
    path: np.ndarray, width: float, height: float
) -> float:
    """Return the minimum signed clearance from a path to the rectangular wall."""

    return float(point_boundary_clearances(path, width, height).min())


def boundary_safety_violation_count(
    path: np.ndarray,
    width: float,
    height: float,
    margin: float,
) -> int:
    """Count path vertices that enter the requested wall-safety region."""

    if margin < 0.0:
        raise ValueError("Boundary safety margin cannot be negative.")
    clearances = point_boundary_clearances(path, width, height)
    return int(np.count_nonzero(clearances < margin))


def boundary_safety_penalty(
    path: np.ndarray,
    width: float,
    height: float,
    margin: float,
) -> float:
    """Return a fixed-plus-squared penalty for wall-safety violations."""

    if margin < 0.0:
        raise ValueError("Boundary safety margin cannot be negative.")
    clearances = point_boundary_clearances(path, width, height)
    shortfalls = np.clip(margin - clearances, 0.0, None)
    return float(np.count_nonzero(shortfalls > 0.0) + np.square(shortfalls).sum())


def count_true_episodes(mask: np.ndarray) -> int:
    """Count contiguous True regions in a one-dimensional boolean mask."""

    boolean = np.asarray(mask, dtype=bool).reshape(-1)
    if boolean.size == 0:
        return 0
    starts = boolean & np.concatenate([[True], ~boolean[:-1]])
    return int(np.count_nonzero(starts))


def pairwise_distance_series(trajectories: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Return synchronized pairwise distance histories for all agent pairs."""

    array = np.asarray(trajectories, dtype=float)
    if array.ndim != 3 or array.shape[2] != 2:
        raise ValueError("trajectories must have shape (agents, time, 2).")
    return {
        (first, second): np.linalg.norm(array[first] - array[second], axis=1)
        for first, second in combinations(range(array.shape[0]), 2)
    }


def inter_agent_separation_report(
    trajectories: np.ndarray,
    *,
    physical_collision_distance: float,
    minimum_separation: float,
    time_step: float,
) -> InterAgentSeparationReport:
    """Summarize synchronized physical collisions and safety violations."""

    if physical_collision_distance <= 0.0:
        raise ValueError("physical_collision_distance must be positive.")
    if minimum_separation < physical_collision_distance:
        raise ValueError("minimum_separation cannot be below collision distance.")
    if time_step <= 0.0:
        raise ValueError("time_step must be positive.")

    distance_series = pairwise_distance_series(trajectories)
    if not distance_series:
        return InterAgentSeparationReport(
            minimum_distance=float("inf"),
            physical_collision_sample_count=0,
            separation_violation_sample_count=0,
            physical_collision_episode_count=0,
            separation_violation_episode_count=0,
            physical_overlap_integral=0.0,
            separation_shortfall_integral=0.0,
        )

    minimum_distance = float("inf")
    collision_samples = 0
    separation_samples = 0
    collision_episodes = 0
    separation_episodes = 0
    physical_overlap = 0.0
    separation_shortfall = 0.0

    for distances in distance_series.values():
        minimum_distance = min(minimum_distance, float(distances.min()))
        collision_mask = distances <= physical_collision_distance
        separation_mask = distances < minimum_separation
        collision_samples += int(np.count_nonzero(collision_mask))
        separation_samples += int(np.count_nonzero(separation_mask))
        collision_episodes += count_true_episodes(collision_mask)
        separation_episodes += count_true_episodes(separation_mask)
        physical_overlap += float(
            np.square(
                np.clip(physical_collision_distance - distances, 0.0, None)
            ).sum()
            * time_step
        )
        separation_shortfall += float(
            np.square(np.clip(minimum_separation - distances, 0.0, None)).sum()
            * time_step
        )

    return InterAgentSeparationReport(
        minimum_distance=minimum_distance,
        physical_collision_sample_count=collision_samples,
        separation_violation_sample_count=separation_samples,
        physical_collision_episode_count=collision_episodes,
        separation_violation_episode_count=separation_episodes,
        physical_overlap_integral=physical_overlap,
        separation_shortfall_integral=separation_shortfall,
    )
