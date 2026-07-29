"""Ant Colony Optimization for grid-based path planning.

The implementation follows the central Ant System mechanism introduced by
Dorigo, Maniezzo, and Colorni: ants construct solutions probabilistically from
pheromone and heuristic information, pheromone evaporates after each colony,
and successful solutions deposit new pheromone. In this project, one ant is a
virtual search agent that constructs one complete start-to-goal path on a grid;
it is not a physical drone.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import thread_time

import numpy as np

from config.scenario import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    ObjectiveWeights,
    Scenario,
)
from utils.collision import segment_collides_circle
from utils.metrics import PathMetrics, compute_path_metrics
from utils.objective import ObjectiveResult, evaluate_path
from utils.path_utils import simplify_path_line_of_sight


# Clockwise 8-connected neighbourhood. Direction indices are also pheromone
# array columns, so each adjacent-edge lookup is constant time.
_DIRECTIONS = np.asarray(
    [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ],
    dtype=int,
)
_OPPOSITE_DIRECTION = np.asarray([4, 5, 6, 7, 0, 1, 2, 3], dtype=int)
_DIRECTION_UNIT_VECTORS = _DIRECTIONS[:, ::-1].astype(float)
_DIRECTION_UNIT_VECTORS /= np.linalg.norm(
    _DIRECTION_UNIT_VECTORS, axis=1, keepdims=True
)


@dataclass(frozen=True)
class ACOConfig:
    """Hyperparameters for the grid-based Ant System baseline."""

    num_ants: int = 80
    max_iterations: int = 200
    grid_resolution: float = 2.5
    alpha: float = 1.0
    beta: float = 6.0
    evaporation_rate: float = 0.30
    pheromone_deposit: float = 100.0
    initial_pheromone: float = 1.0
    minimum_pheromone: float = 1e-6
    max_steps_factor: float = 2.5
    continuous_check_top_k: int = 5

    def __post_init__(self) -> None:
        if self.num_ants < 1:
            raise ValueError("num_ants must be positive.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive.")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("alpha and beta cannot be negative.")
        if not (0.0 < self.evaporation_rate < 1.0):
            raise ValueError("evaporation_rate must be in (0, 1).")
        if self.pheromone_deposit <= 0:
            raise ValueError("pheromone_deposit must be positive.")
        if self.initial_pheromone <= 0 or self.minimum_pheromone <= 0:
            raise ValueError("Pheromone values must be positive.")
        if self.max_steps_factor < 1.0:
            raise ValueError("max_steps_factor must be at least 1.")
        if not (1 <= self.continuous_check_top_k <= self.num_ants):
            raise ValueError("continuous_check_top_k must be between 1 and num_ants.")


@dataclass(frozen=True)
class GridGraph:
    """Precomputed 8-connected graph derived from the continuous scenario."""

    coordinates: np.ndarray
    free_nodes: np.ndarray
    neighbours: np.ndarray
    edge_lengths: np.ndarray
    start_index: int
    goal_index: int
    rows: int
    cols: int

    @property
    def num_nodes(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class _AntRoute:
    """Internal compact representation of one successful ant route."""

    nodes: np.ndarray
    directions: np.ndarray
    grid_fitness: float


@dataclass(frozen=True)
class ACOResult:
    """Commonly useful outputs from one ACO optimization run."""

    algorithm: str
    path: np.ndarray
    raw_path: np.ndarray
    best_fitness: float
    fitness_history: np.ndarray
    best_path_history: tuple[np.ndarray, ...]
    pheromone_history: np.ndarray
    colony_path_history: tuple[tuple[np.ndarray, ...], ...]
    iterations: int
    evaluations: int
    runtime: float
    seed: int
    objective: ObjectiveResult
    metrics: PathMetrics
    successful_candidates: int
    graph_rows: int
    graph_cols: int
    grid_resolution: float

    @property
    def success(self) -> bool:
        return self.metrics.success


def _axis_coordinates(length: float, resolution: float) -> np.ndarray:
    """Return an evenly spaced axis including both rectangular boundaries."""

    interval_count = int(round(length / resolution))
    if interval_count < 1 or not np.isclose(
        interval_count * resolution, length, atol=1e-9
    ):
        raise ValueError(
            "Map dimensions must be integer multiples of grid_resolution."
        )
    return np.linspace(0.0, length, interval_count + 1)


def _point_to_grid_index(
    point: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> int:
    """Map an exactly representable scenario point to a flattened grid index."""

    x_index = int(np.argmin(np.abs(x_coordinates - point[0])))
    y_index = int(np.argmin(np.abs(y_coordinates - point[1])))
    if not np.isclose(x_coordinates[x_index], point[0], atol=1e-9) or not np.isclose(
        y_coordinates[y_index], point[1], atol=1e-9
    ):
        raise ValueError(
            "Start and goal must lie on grid nodes for the selected resolution."
        )
    return y_index * len(x_coordinates) + x_index


def _point_is_free(point: np.ndarray, scenario: Scenario) -> bool:
    """Return whether a grid node lies outside every expanded obstacle."""

    return all(
        np.linalg.norm(point - obstacle.center_array)
        > obstacle.radius + scenario.safety_margin
        for obstacle in scenario.obstacles
    )


def _edge_is_free(start: np.ndarray, end: np.ndarray, scenario: Scenario) -> bool:
    """Validate an edge in the original continuous geometry."""

    return not any(
        segment_collides_circle(
            start,
            end,
            obstacle,
            margin=scenario.safety_margin,
        )
        for obstacle in scenario.obstacles
    )


def build_grid_graph(scenario: Scenario, config: ACOConfig) -> GridGraph:
    """Convert the continuous scenario into an exact-collision-checked grid."""

    x_coordinates = _axis_coordinates(scenario.width, config.grid_resolution)
    y_coordinates = _axis_coordinates(scenario.height, config.grid_resolution)
    rows = len(y_coordinates)
    cols = len(x_coordinates)

    mesh_x, mesh_y = np.meshgrid(x_coordinates, y_coordinates)
    coordinates = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)])
    free_nodes = np.asarray(
        [_point_is_free(point, scenario) for point in coordinates], dtype=bool
    )

    start_index = _point_to_grid_index(
        scenario.start_array, x_coordinates, y_coordinates
    )
    goal_index = _point_to_grid_index(scenario.goal_array, x_coordinates, y_coordinates)
    if not free_nodes[start_index] or not free_nodes[goal_index]:
        raise ValueError("Start and goal must be outside expanded obstacles.")

    neighbours = np.full((rows * cols, len(_DIRECTIONS)), -1, dtype=int)
    edge_lengths = np.zeros_like(neighbours, dtype=float)

    for row in range(rows):
        for col in range(cols):
            node_index = row * cols + col
            if not free_nodes[node_index]:
                continue

            for direction_index, (delta_row, delta_col) in enumerate(_DIRECTIONS):
                next_row = row + int(delta_row)
                next_col = col + int(delta_col)
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue

                neighbour_index = next_row * cols + next_col
                if not free_nodes[neighbour_index]:
                    continue
                if not _edge_is_free(
                    coordinates[node_index], coordinates[neighbour_index], scenario
                ):
                    continue

                neighbours[node_index, direction_index] = neighbour_index
                edge_lengths[node_index, direction_index] = float(
                    np.linalg.norm(
                        coordinates[neighbour_index] - coordinates[node_index]
                    )
                )

    if np.all(neighbours[start_index] < 0) or np.all(neighbours[goal_index] < 0):
        raise ValueError("Start or goal has no valid grid connection.")

    return GridGraph(
        coordinates=coordinates,
        free_nodes=free_nodes,
        neighbours=neighbours,
        edge_lengths=edge_lengths,
        start_index=start_index,
        goal_index=goal_index,
        rows=rows,
        cols=cols,
    )


def _heuristic_matrix(graph: GridGraph) -> np.ndarray:
    """Return edge desirability based on edge length and goal distance."""

    heuristic = np.zeros_like(graph.edge_lengths, dtype=float)
    goal = graph.coordinates[graph.goal_index]
    valid_sources, valid_directions = np.nonzero(graph.neighbours >= 0)
    valid_targets = graph.neighbours[valid_sources, valid_directions]
    remaining = np.linalg.norm(graph.coordinates[valid_targets] - goal, axis=1)
    edge = graph.edge_lengths[valid_sources, valid_directions]
    heuristic[valid_sources, valid_directions] = 1.0 / (edge + remaining + 1e-12)
    return heuristic


def _grid_route_fitness(
    directions: np.ndarray,
    edge_lengths: np.ndarray,
    weights: ObjectiveWeights,
) -> float:
    """Compute the shared objective efficiently for a guaranteed-safe grid path.

    Grid construction already guarantees boundary, physical-collision, and
    safety-margin feasibility. Therefore those shared objective terms are
    exactly zero, leaving path length plus squared turning-angle smoothness.
    """

    length_value = float(edge_lengths.sum())
    if len(directions) < 2:
        smoothness_value = 0.0
    else:
        previous = _DIRECTION_UNIT_VECTORS[directions[:-1]]
        following = _DIRECTION_UNIT_VECTORS[directions[1:]]
        cosine = np.sum(previous * following, axis=1)
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))
        smoothness_value = float(np.square(angles).sum())
    return weights.length * length_value + weights.smoothness * smoothness_value


def _construct_ant_route(
    graph: GridGraph,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    config: ACOConfig,
    weights: ObjectiveWeights,
    rng: np.random.Generator,
) -> _AntRoute | None:
    """Construct one loop-free grid route with roulette-wheel selection."""

    current = graph.start_index
    route_nodes = [current]
    route_directions: list[int] = []
    route_edge_lengths: list[float] = []
    visited = np.zeros(graph.num_nodes, dtype=bool)
    visited[current] = True
    max_steps = min(
        graph.num_nodes - 1,
        int(np.ceil(config.max_steps_factor * (graph.rows + graph.cols))),
    )

    for _ in range(max_steps):
        if current == graph.goal_index:
            directions_array = np.asarray(route_directions, dtype=int)
            lengths_array = np.asarray(route_edge_lengths, dtype=float)
            return _AntRoute(
                nodes=np.asarray(route_nodes, dtype=int),
                directions=directions_array,
                grid_fitness=float(
                    _grid_route_fitness(directions_array, lengths_array, weights)
                ),
            )

        candidate_directions = np.flatnonzero(graph.neighbours[current] >= 0)
        if candidate_directions.size == 0:
            return None

        candidate_nodes = graph.neighbours[current, candidate_directions]
        unvisited_mask = ~visited[candidate_nodes]
        candidate_directions = candidate_directions[unvisited_mask]
        candidate_nodes = candidate_nodes[unvisited_mask]
        if candidate_directions.size == 0:
            return None

        desirability = np.power(
            pheromone[current, candidate_directions], config.alpha
        ) * np.power(heuristic[current, candidate_directions], config.beta)
        desirability_sum = float(desirability.sum())
        if not np.isfinite(desirability_sum) or desirability_sum <= 0.0:
            probabilities = np.full(
                candidate_directions.size, 1.0 / candidate_directions.size
            )
        else:
            probabilities = desirability / desirability_sum

        selected = int(rng.choice(candidate_directions.size, p=probabilities))
        selected_direction = int(candidate_directions[selected])
        current = int(candidate_nodes[selected])

        route_nodes.append(current)
        route_directions.append(selected_direction)
        route_edge_lengths.append(
            float(graph.edge_lengths[route_nodes[-2], selected_direction])
        )
        visited[current] = True

    if current != graph.goal_index:
        return None

    directions_array = np.asarray(route_directions, dtype=int)
    lengths_array = np.asarray(route_edge_lengths, dtype=float)
    return _AntRoute(
        nodes=np.asarray(route_nodes, dtype=int),
        directions=directions_array,
        grid_fitness=float(
            _grid_route_fitness(directions_array, lengths_array, weights)
        ),
    )


def _route_to_paths(
    route: _AntRoute,
    graph: GridGraph,
    scenario: Scenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw grid path and deterministic line-of-sight simplification."""

    raw_path = graph.coordinates[route.nodes]
    simplified_path = simplify_path_line_of_sight(
        raw_path,
        scenario.obstacles,
        margin=scenario.safety_margin,
    )
    return raw_path, simplified_path


def _node_pheromone_map(pheromone: np.ndarray, graph: GridGraph) -> np.ndarray:
    """Aggregate directional pheromone into a 2D node-intensity map."""

    valid_edges = graph.neighbours >= 0
    node_values = np.where(valid_edges, pheromone, 0.0).sum(axis=1)
    return node_values.reshape(graph.rows, graph.cols)


def _update_pheromone(
    pheromone: np.ndarray,
    graph: GridGraph,
    successful_routes: list[_AntRoute],
    config: ACOConfig,
) -> None:
    """Apply global evaporation and Ant System deposits from successful ants."""

    valid_edges = graph.neighbours >= 0
    pheromone[valid_edges] *= 1.0 - config.evaporation_rate
    pheromone[valid_edges] = np.maximum(
        pheromone[valid_edges], config.minimum_pheromone
    )

    for route in successful_routes:
        deposit = config.pheromone_deposit / max(route.grid_fitness, 1e-12)
        sources = route.nodes[:-1]
        targets = route.nodes[1:]
        directions = route.directions
        reverse_directions = _OPPOSITE_DIRECTION[directions]
        pheromone[sources, directions] += deposit
        pheromone[targets, reverse_directions] += deposit


def _run_colony(
    graph: GridGraph,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    scenario: Scenario,
    weights: ObjectiveWeights,
    config: ACOConfig,
    rng: np.random.Generator,
) -> tuple[list[_AntRoute], tuple[np.ndarray, np.ndarray, float] | None]:
    """Construct one colony and continuously validate its best grid routes."""

    successful_routes: list[_AntRoute] = []
    for _ in range(config.num_ants):
        route = _construct_ant_route(
            graph,
            pheromone,
            heuristic,
            config,
            weights,
            rng,
        )
        if route is not None:
            successful_routes.append(route)

    if not successful_routes:
        return successful_routes, None

    successful_routes.sort(key=lambda item: item.grid_fitness)
    colony_best: tuple[np.ndarray, np.ndarray, float] | None = None
    for route in successful_routes[: config.continuous_check_top_k]:
        raw_path, simplified_path = _route_to_paths(route, graph, scenario)
        continuous_fitness = evaluate_path(simplified_path, scenario, weights).total
        if colony_best is None or continuous_fitness < colony_best[2]:
            colony_best = (raw_path, simplified_path, float(continuous_fitness))

    return successful_routes, colony_best


def run_aco(
    scenario: Scenario,
    config: ACOConfig = ACOConfig(),
    *,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> ACOResult:
    """Optimize a collision-aware path with grid-based Ant System.

    Notes
    -----
    - One ant constructs one virtual candidate path; ants are not physical drones.
    - The continuous scenario is rasterized only for search. Grid edges are
      validated against the original circles expanded by the safety margin.
    - Each ant is counted as one path-construction attempt. Ants that do not
      reach the goal are treated as failed candidates. Successful grid routes
      receive the safe-path length and smoothness terms, and the best few are
      simplified and re-evaluated in continuous space for final selection.
    - ``fitness_history`` stores best-so-far continuous fitness, including the
      initialization colony.
    """

    rng = np.random.default_rng(seed)
    start_time = thread_time()
    graph = build_grid_graph(scenario, config)
    heuristic = _heuristic_matrix(graph)

    pheromone = np.zeros_like(graph.edge_lengths, dtype=float)
    valid_edges = graph.neighbours >= 0
    pheromone[valid_edges] = config.initial_pheromone

    best_raw_path: np.ndarray | None = None
    best_path: np.ndarray | None = None
    best_fitness = float("inf")
    best_path_history: list[np.ndarray] = []
    fitness_history: list[float] = []
    pheromone_history: list[np.ndarray] = []
    colony_path_history: list[tuple[np.ndarray, ...]] = []
    successful_candidates = 0
    evaluations = 0

    colony_count = config.max_iterations + 1
    for _ in range(colony_count):
        successful_routes, colony_best = _run_colony(
            graph,
            pheromone,
            heuristic,
            scenario,
            weights,
            config,
            rng,
        )
        evaluations += config.num_ants
        successful_candidates += len(successful_routes)

        if colony_best is not None and colony_best[2] < best_fitness:
            best_raw_path, best_path, best_fitness = colony_best

        colony_path_history.append(
            tuple(
                graph.coordinates[route.nodes].copy()
                for route in successful_routes[:12]
            )
        )
        _update_pheromone(pheromone, graph, successful_routes, config)
        pheromone_history.append(_node_pheromone_map(pheromone, graph).copy())
        fitness_history.append(best_fitness)
        best_path_history.append(
            best_path.copy() if best_path is not None else np.empty((0, 2), dtype=float)
        )

    runtime = thread_time() - start_time
    if best_path is None or best_raw_path is None or not np.isfinite(best_fitness):
        raise RuntimeError(
            "ACO did not construct a complete start-to-goal path. "
            "Increase ants/iterations or revise grid settings."
        )

    objective = evaluate_path(best_path, scenario, weights)
    metrics = compute_path_metrics(best_path, scenario)

    return ACOResult(
        algorithm="ACO",
        path=best_path,
        raw_path=best_raw_path,
        best_fitness=float(best_fitness),
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_path_history=tuple(best_path_history),
        pheromone_history=np.asarray(pheromone_history, dtype=np.float32),
        colony_path_history=tuple(colony_path_history),
        iterations=config.max_iterations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=objective,
        metrics=metrics,
        successful_candidates=successful_candidates,
        graph_rows=graph.rows,
        graph_cols=graph.cols,
        grid_resolution=config.grid_resolution,
    )
