"""Collision and map-boundary utilities shared by all planners."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

from config.scenarios import CircleObstacle, Point, Scenario


def is_within_bounds(point: Point, width: float, height: float) -> bool:
    """Return whether a point lies inside the map, including its boundary."""

    x, y = point
    return 0.0 <= x <= width and 0.0 <= y <= height


def point_in_collision(
    point: Point,
    obstacles: Iterable[CircleObstacle],
    safety_margin: float = 0.0,
) -> bool:
    """Return whether a point lies inside any inflated circular obstacle."""

    x, y = point
    for obstacle in obstacles:
        distance = math.hypot(x - obstacle.x, y - obstacle.y)
        if distance <= obstacle.radius + safety_margin:
            return True
    return False


def segment_in_collision(
    start: Point,
    end: Point,
    obstacles: Iterable[CircleObstacle],
    safety_margin: float = 0.0,
) -> bool:
    """Return whether a line segment intersects any inflated circle."""

    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    denominator = dx * dx + dy * dy

    for obstacle in obstacles:
        if denominator == 0.0:
            closest_x, closest_y = x1, y1
        else:
            t = ((obstacle.x - x1) * dx + (obstacle.y - y1) * dy) / denominator
            t = min(1.0, max(0.0, t))
            closest_x = x1 + t * dx
            closest_y = y1 + t * dy

        distance = math.hypot(closest_x - obstacle.x, closest_y - obstacle.y)
        if distance <= obstacle.radius + safety_margin:
            return True

    return False


def create_occupancy_grid(scenario: Scenario) -> np.ndarray:
    """Rasterize the continuous scenario into a Boolean occupancy grid.

    The first array axis represents x and the second represents y to preserve
    the coordinate convention used in the original practice code.
    """

    resolution = scenario.grid_resolution
    x_count = int(round(scenario.width / resolution)) + 1
    y_count = int(round(scenario.height / resolution)) + 1
    grid = np.zeros((x_count, y_count), dtype=bool)

    for x_index in range(x_count):
        for y_index in range(y_count):
            point = (x_index * resolution, y_index * resolution)
            if point_in_collision(point, scenario.obstacles, scenario.safety_margin):
                grid[x_index, y_index] = True

    return grid


def point_to_grid(point: Point, resolution: float) -> Tuple[int, int]:
    """Convert continuous coordinates to the nearest grid index."""

    return int(round(point[0] / resolution)), int(round(point[1] / resolution))


def grid_to_point(index: Tuple[int, int], resolution: float) -> Point:
    """Convert a grid index to continuous coordinates."""

    return index[0] * resolution, index[1] * resolution
