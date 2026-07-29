"""Exact line-segment collision and clearance calculations for circles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from config.scenario import CircleObstacle
from .path_utils import ensure_path_array


def point_to_segment_distance(
    point: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
) -> float:
    """Return the shortest Euclidean distance from a point to a segment."""

    point = np.asarray(point, dtype=float)
    start = np.asarray(segment_start, dtype=float)
    end = np.asarray(segment_end, dtype=float)

    segment = end - start
    squared_length = float(np.dot(segment, segment))
    if squared_length <= 1e-24:
        return float(np.linalg.norm(point - start))

    projection = float(np.dot(point - start, segment) / squared_length)
    projection = float(np.clip(projection, 0.0, 1.0))
    nearest = start + projection * segment
    return float(np.linalg.norm(point - nearest))


def segment_circle_clearance(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    obstacle: CircleObstacle,
) -> float:
    """Return signed clearance from a segment to an obstacle surface.

    Positive values indicate free space, zero indicates tangency, and negative
    values indicate penetration into the physical obstacle.
    """

    center_distance = point_to_segment_distance(
        obstacle.center_array, segment_start, segment_end
    )
    return center_distance - obstacle.radius


def segment_collides_circle(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    obstacle: CircleObstacle,
    margin: float = 0.0,
) -> bool:
    """Return whether a segment intersects an obstacle expanded by ``margin``."""

    if margin < 0:
        raise ValueError("Collision margin cannot be negative.")
    return segment_circle_clearance(segment_start, segment_end, obstacle) <= margin


def segment_obstacle_clearances(
    path: np.ndarray,
    obstacles: Sequence[CircleObstacle],
) -> np.ndarray:
    """Return signed clearance for every path-segment/obstacle pair."""

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


def count_path_collisions(
    path: np.ndarray,
    obstacles: Sequence[CircleObstacle],
    margin: float = 0.0,
) -> int:
    """Count path-segment/obstacle intersections at the requested margin."""

    clearances = segment_obstacle_clearances(path, obstacles)
    if clearances.size == 0:
        return 0
    return int(np.count_nonzero(clearances <= margin))


def minimum_obstacle_clearance(
    path: np.ndarray,
    obstacles: Sequence[CircleObstacle],
) -> float:
    """Return minimum signed clearance from the path to any obstacle."""

    clearances = segment_obstacle_clearances(path, obstacles)
    if clearances.size == 0:
        return float("inf")
    return float(clearances.min())


def path_is_collision_free(
    path: np.ndarray,
    obstacles: Sequence[CircleObstacle],
    margin: float = 0.0,
) -> bool:
    """Return whether every path segment avoids all expanded obstacles."""

    return count_path_collisions(path, obstacles, margin=margin) == 0


def boundary_violation_amount(path: np.ndarray, width: float, height: float) -> float:
    """Return squared distance of out-of-map coordinates from valid bounds."""

    array = ensure_path_array(path)
    x = array[:, 0]
    y = array[:, 1]

    violation = (
        np.square(np.clip(-x, 0.0, None)).sum()
        + np.square(np.clip(x - width, 0.0, None)).sum()
        + np.square(np.clip(-y, 0.0, None)).sum()
        + np.square(np.clip(y - height, 0.0, None)).sum()
    )
    return float(violation)


def path_is_within_bounds(path: np.ndarray, width: float, height: float) -> bool:
    """Return whether every path point lies inside the rectangular map."""

    return boundary_violation_amount(path, width, height) == 0.0
