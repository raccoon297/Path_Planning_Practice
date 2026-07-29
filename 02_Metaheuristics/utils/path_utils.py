"""Path representation and geometric path utilities."""

from __future__ import annotations

import numpy as np


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


def vector_to_path(
    vector: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    num_waypoints: int,
) -> np.ndarray:
    """Convert a flattened waypoint vector into a complete polyline path."""

    flat = np.asarray(vector, dtype=float).reshape(-1)
    expected_size = num_waypoints * 2
    if flat.size != expected_size:
        raise ValueError(
            f"Expected waypoint vector of size {expected_size}, got {flat.size}."
        )
    if not np.all(np.isfinite(flat)):
        raise ValueError("Waypoint vector contains NaN or infinite values.")

    waypoints = flat.reshape(num_waypoints, 2)
    return np.vstack(
        [np.asarray(start, dtype=float), waypoints, np.asarray(goal, dtype=float)]
    )


def path_to_vector(path: np.ndarray) -> np.ndarray:
    """Flatten intermediate waypoints while excluding start and goal."""

    array = ensure_path_array(path)
    if len(array) <= 2:
        return np.empty(0, dtype=float)
    return array[1:-1].reshape(-1).copy()


def clip_waypoint_vector(
    vector: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Clip a waypoint vector to optimizer search bounds."""

    flat = np.asarray(vector, dtype=float).reshape(-1)
    lower = np.asarray(lower_bounds, dtype=float).reshape(-1)
    upper = np.asarray(upper_bounds, dtype=float).reshape(-1)
    if flat.shape != lower.shape or flat.shape != upper.shape:
        raise ValueError("Vector and bound arrays must have identical shapes.")
    return np.clip(flat, lower, upper)


def path_length(path: np.ndarray) -> float:
    """Return total Euclidean length of a polyline."""

    array = ensure_path_array(path)
    return float(np.linalg.norm(np.diff(array, axis=0), axis=1).sum())


def turning_angles(path: np.ndarray) -> np.ndarray:
    """Return unsigned turning angles in radians for interior path points."""

    array = ensure_path_array(path)
    if len(array) < 3:
        return np.empty(0, dtype=float)

    previous_vectors = array[1:-1] - array[:-2]
    next_vectors = array[2:] - array[1:-1]
    previous_norms = np.linalg.norm(previous_vectors, axis=1)
    next_norms = np.linalg.norm(next_vectors, axis=1)

    valid = (previous_norms > 1e-12) & (next_norms > 1e-12)
    angles = np.zeros(len(previous_vectors), dtype=float)
    if np.any(valid):
        cosine = np.sum(previous_vectors[valid] * next_vectors[valid], axis=1)
        cosine /= previous_norms[valid] * next_norms[valid]
        cosine = np.clip(cosine, -1.0, 1.0)
        angles[valid] = np.arccos(cosine)

    # Duplicate consecutive waypoints create a degenerate turn and should be
    # discouraged rather than silently treated as perfectly smooth.
    angles[~valid] = np.pi
    return angles


def smoothness_cost(path: np.ndarray) -> float:
    """Return the sum of squared turning angles."""

    angles = turning_angles(path)
    return float(np.square(angles).sum())


def simplify_path_line_of_sight(
    path: np.ndarray,
    obstacles,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Greedily remove intermediate points while preserving collision clearance.

    This deterministic post-processing is especially useful for grid paths:
    it removes staircase points that are unnecessary in the original continuous
    geometry. The first and last path points are always preserved.
    """

    if margin < 0:
        raise ValueError("margin cannot be negative.")

    # Local import avoids a module-level cycle: collision.py already imports
    # ensure_path_array from this module.
    from .collision import segment_collides_circle

    array = ensure_path_array(path)
    if len(array) <= 2:
        return array.copy()

    simplified = [array[0]]
    anchor_index = 0
    final_index = len(array) - 1

    while anchor_index < final_index:
        next_index = final_index
        while next_index > anchor_index + 1:
            if not any(
                segment_collides_circle(
                    array[anchor_index],
                    array[next_index],
                    obstacle,
                    margin=margin,
                )
                for obstacle in obstacles
            ):
                break
            next_index -= 1

        simplified.append(array[next_index])
        anchor_index = next_index

    return np.asarray(simplified, dtype=float)
