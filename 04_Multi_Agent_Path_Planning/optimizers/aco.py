"""Reservation-aware Ant Colony Optimization for multi-agent path planning.

One joint ant constructs paths for all agents on a shared 8-connected grid.
Agents are planned in a sampled priority order. Paths already created by the
same ant become time-indexed reservations, so later agents reduce the
probability of moves that approach an occupied place at a similar time.
Successful joint plans reinforce both traversed grid edges and selected start
-delay bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from config.scenario import (
    DEFAULT_OBJECTIVE_WEIGHTS,
    MultiAgentObjectiveWeights,
    MultiAgentScenario,
)
from utils.collision import segment_circle_clearance
from utils.metrics import JointPlanMetrics, compute_joint_plan_metrics
from utils.objective import JointObjectiveResult, evaluate_joint_plan
from utils.path_utils import (
    JointPlan,
    normalize_start_delays,
    path_length,
    sample_path_trajectory,
    simplify_path_line_of_sight,
)

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


@dataclass(frozen=True)
class MultiAgentACOConfig:
    """Hyperparameters for reservation-aware joint Ant System."""

    num_ants: int = 40
    max_iterations: int = 40
    grid_resolution: float = 5.0
    alpha: float = 1.0
    beta: float = 7.0
    reservation_strength: float = 5.0
    delay_beta: float = 5.0
    delay_step: float = 1.0
    delay_evaluation_step: float = 0.5
    evaporation_rate: float = 0.25
    pheromone_deposit: float = 300.0
    initial_pheromone: float = 1.0
    minimum_pheromone: float = 1e-5
    max_steps_factor: float = 1.8
    elite_fraction: float = 0.15
    global_best_weight: float = 2.0
    recorded_colony_plans: int = 8

    def __post_init__(self) -> None:
        if self.num_ants < 1:
            raise ValueError("num_ants must be positive.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive.")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("alpha and beta cannot be negative.")
        if self.reservation_strength < 0.0 or self.delay_beta < 0.0:
            raise ValueError("Reservation and delay strengths cannot be negative.")
        if self.delay_step <= 0.0 or self.delay_evaluation_step <= 0.0:
            raise ValueError("Delay steps must be positive.")
        if not (0.0 < self.evaporation_rate < 1.0):
            raise ValueError("evaporation_rate must be in (0, 1).")
        if self.pheromone_deposit <= 0.0:
            raise ValueError("pheromone_deposit must be positive.")
        if self.initial_pheromone <= 0.0 or self.minimum_pheromone <= 0.0:
            raise ValueError("Pheromone values must be positive.")
        if self.max_steps_factor < 1.0:
            raise ValueError("max_steps_factor must be at least 1.")
        if not (0.0 < self.elite_fraction <= 1.0):
            raise ValueError("elite_fraction must be in (0, 1].")
        if self.global_best_weight < 0.0:
            raise ValueError("global_best_weight cannot be negative.")
        if self.recorded_colony_plans < 1:
            raise ValueError("recorded_colony_plans must be positive.")


@dataclass(frozen=True)
class GridGraph:
    coordinates: np.ndarray
    free_nodes: np.ndarray
    neighbours: np.ndarray
    edge_lengths: np.ndarray
    start_indices: np.ndarray
    goal_indices: np.ndarray
    rows: int
    cols: int

    @property
    def num_nodes(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class _Route:
    nodes: np.ndarray
    directions: np.ndarray
    edge_lengths: np.ndarray
    raw_path: np.ndarray
    simplified_path: np.ndarray


@dataclass(frozen=True)
class _TimedReservation:
    path: np.ndarray
    start_delay: float
    cumulative_lengths: np.ndarray

    @classmethod
    def from_path(cls, path: np.ndarray, start_delay: float) -> "_TimedReservation":
        segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        return cls(path=path, start_delay=float(start_delay), cumulative_lengths=cumulative)

    def position_at(self, time_value: float, speed: float) -> np.ndarray:
        if time_value <= self.start_delay:
            return self.path[0]
        travelled = min(
            max((time_value - self.start_delay) * speed, 0.0),
            float(self.cumulative_lengths[-1]),
        )
        if travelled >= self.cumulative_lengths[-1] - 1e-12:
            return self.path[-1]
        index = int(np.searchsorted(self.cumulative_lengths, travelled, side="right") - 1)
        index = max(0, min(index, len(self.path) - 2))
        denominator = self.cumulative_lengths[index + 1] - self.cumulative_lengths[index]
        fraction = 0.0 if denominator <= 1e-12 else (
            travelled - self.cumulative_lengths[index]
        ) / denominator
        return self.path[index] * (1.0 - fraction) + self.path[index + 1] * fraction


@dataclass(frozen=True)
class _AntSolution:
    plan: JointPlan
    routes: tuple[_Route, ...]
    delay_bins: np.ndarray
    priority: np.ndarray
    fitness: float
    objective: JointObjectiveResult
    metrics: JointPlanMetrics


@dataclass(frozen=True)
class MultiAgentACOResult:
    algorithm: str
    plan: JointPlan
    raw_paths: tuple[np.ndarray, ...]
    best_fitness: float
    fitness_history: np.ndarray
    best_plan_history: tuple[JointPlan, ...]
    pheromone_history: np.ndarray
    colony_plan_history: tuple[tuple[JointPlan, ...], ...]
    iterations: int
    evaluations: int
    runtime: float
    seed: int
    objective: JointObjectiveResult
    metrics: JointPlanMetrics
    successful_candidates: int
    graph_rows: int
    graph_cols: int
    grid_resolution: float
    delay_step: float

    @property
    def success(self) -> bool:
        return self.metrics.success


def _axis_coordinates(length: float, resolution: float) -> np.ndarray:
    interval_count = int(round(length / resolution))
    if interval_count < 1 or not np.isclose(interval_count * resolution, length):
        raise ValueError("Map dimensions must be integer multiples of grid_resolution.")
    return np.linspace(0.0, length, interval_count + 1)


def _point_to_grid_index(
    point: np.ndarray,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> int:
    x_index = int(np.argmin(np.abs(x_coordinates - point[0])))
    y_index = int(np.argmin(np.abs(y_coordinates - point[1])))
    if not np.isclose(x_coordinates[x_index], point[0]) or not np.isclose(
        y_coordinates[y_index], point[1]
    ):
        raise ValueError("All starts and goals must lie on grid nodes.")
    return y_index * len(x_coordinates) + x_index


def _point_is_free(point: np.ndarray, scenario: MultiAgentScenario) -> bool:
    x, y = float(point[0]), float(point[1])
    wall_clearance = min(x, scenario.width - x, y, scenario.height - y)
    return wall_clearance >= scenario.boundary_safety_margin and all(
        np.linalg.norm(point - obstacle.center_array)
        > obstacle.radius + scenario.obstacle_safety_margin
        for obstacle in scenario.obstacles
    )


def _edge_is_free(
    start: np.ndarray,
    end: np.ndarray,
    scenario: MultiAgentScenario,
) -> bool:
    return all(
        segment_circle_clearance(start, end, obstacle)
        > scenario.obstacle_safety_margin
        for obstacle in scenario.obstacles
    )


def build_grid_graph(
    scenario: MultiAgentScenario,
    config: MultiAgentACOConfig,
) -> GridGraph:
    """Build an exact-collision-checked 8-connected graph."""

    x_coordinates = _axis_coordinates(scenario.width, config.grid_resolution)
    y_coordinates = _axis_coordinates(scenario.height, config.grid_resolution)
    rows = len(y_coordinates)
    cols = len(x_coordinates)
    mesh_x, mesh_y = np.meshgrid(x_coordinates, y_coordinates)
    coordinates = np.column_stack([mesh_x.reshape(-1), mesh_y.reshape(-1)])
    free_nodes = np.asarray(
        [_point_is_free(point, scenario) for point in coordinates], dtype=bool
    )

    start_indices = np.asarray(
        [
            _point_to_grid_index(task.start_array, x_coordinates, y_coordinates)
            for task in scenario.tasks
        ],
        dtype=int,
    )
    goal_indices = np.asarray(
        [
            _point_to_grid_index(task.goal_array, x_coordinates, y_coordinates)
            for task in scenario.tasks
        ],
        dtype=int,
    )
    if not np.all(free_nodes[start_indices]) or not np.all(free_nodes[goal_indices]):
        raise ValueError("All starts and goals must be outside expanded obstacles.")

    neighbours = np.full((rows * cols, len(_DIRECTIONS)), -1, dtype=int)
    edge_lengths = np.zeros_like(neighbours, dtype=float)
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if not free_nodes[node]:
                continue
            for direction_index, (delta_row, delta_col) in enumerate(_DIRECTIONS):
                next_row = row + int(delta_row)
                next_col = col + int(delta_col)
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                target = next_row * cols + next_col
                if not free_nodes[target]:
                    continue
                if not _edge_is_free(coordinates[node], coordinates[target], scenario):
                    continue
                neighbours[node, direction_index] = target
                edge_lengths[node, direction_index] = float(
                    np.linalg.norm(coordinates[target] - coordinates[node])
                )

    for index in np.concatenate([start_indices, goal_indices]):
        if np.all(neighbours[index] < 0):
            raise ValueError("A start or goal has no valid grid connection.")

    return GridGraph(
        coordinates=coordinates,
        free_nodes=free_nodes,
        neighbours=neighbours,
        edge_lengths=edge_lengths,
        start_indices=start_indices,
        goal_indices=goal_indices,
        rows=rows,
        cols=cols,
    )


def _heuristic_tensor(graph: GridGraph, scenario: MultiAgentScenario) -> np.ndarray:
    heuristic = np.zeros(
        (scenario.num_agents, graph.num_nodes, len(_DIRECTIONS)), dtype=float
    )
    sources, directions = np.nonzero(graph.neighbours >= 0)
    targets = graph.neighbours[sources, directions]
    edge = graph.edge_lengths[sources, directions]
    for agent_index, goal_index in enumerate(graph.goal_indices):
        goal = graph.coordinates[goal_index]
        remaining = np.linalg.norm(graph.coordinates[targets] - goal, axis=1)
        heuristic[agent_index, sources, directions] = 1.0 / (
            edge + remaining + 1e-12
        )
    return heuristic


def _roulette_choice(weights: np.ndarray, rng: np.random.Generator) -> int:
    values = np.asarray(weights, dtype=float)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        return int(rng.integers(0, len(values)))
    return int(rng.choice(len(values), p=values / total))


def _sample_provisional_delay(
    delay_pheromone: np.ndarray,
    delay_values: np.ndarray,
    agent_index: int,
    config: MultiAgentACOConfig,
    rng: np.random.Generator,
) -> float:
    desirability = np.power(delay_pheromone[agent_index], config.alpha) / np.power(
        1.0 + delay_values, 1.5
    )
    return float(delay_values[_roulette_choice(desirability, rng)])


def _reservation_factor(
    start_point: np.ndarray,
    end_point: np.ndarray,
    start_time: float,
    end_time: float,
    reservations: list[_TimedReservation],
    scenario: MultiAgentScenario,
    config: MultiAgentACOConfig,
) -> float:
    if not reservations:
        return 1.0

    midpoint = 0.5 * (start_point + end_point)
    midpoint_time = 0.5 * (start_time + end_time)
    minimum_distance = float("inf")
    for reservation in reservations:
        for point, time_value in ((midpoint, midpoint_time), (end_point, end_time)):
            distance = float(
                np.linalg.norm(
                    point - reservation.position_at(time_value, scenario.speed)
                )
            )
            minimum_distance = min(minimum_distance, distance)

    if minimum_distance <= scenario.physical_agent_collision_distance:
        return 1e-9
    if minimum_distance < scenario.minimum_agent_separation:
        shortfall_ratio = (
            scenario.minimum_agent_separation - minimum_distance
        ) / scenario.minimum_agent_separation
        return float(np.exp(-config.reservation_strength * shortfall_ratio))
    return 1.0


def _construct_route(
    agent_index: int,
    graph: GridGraph,
    pheromone: np.ndarray,
    heuristic: np.ndarray,
    provisional_delay: float,
    reservations: list[_TimedReservation],
    scenario: MultiAgentScenario,
    config: MultiAgentACOConfig,
    rng: np.random.Generator,
) -> _Route | None:
    current = int(graph.start_indices[agent_index])
    goal = int(graph.goal_indices[agent_index])
    nodes = [current]
    directions: list[int] = []
    lengths: list[float] = []
    visited = np.zeros(graph.num_nodes, dtype=bool)
    visited[current] = True
    cumulative_length = 0.0
    max_steps = min(
        graph.num_nodes - 1,
        int(np.ceil(config.max_steps_factor * (graph.rows + graph.cols))),
    )

    for _ in range(max_steps):
        if current == goal:
            break
        candidate_directions = np.flatnonzero(graph.neighbours[current] >= 0)
        candidate_nodes = graph.neighbours[current, candidate_directions]
        unvisited = ~visited[candidate_nodes]
        candidate_directions = candidate_directions[unvisited]
        candidate_nodes = candidate_nodes[unvisited]
        if candidate_directions.size == 0:
            return None

        factors = np.empty(candidate_directions.size, dtype=float)
        current_point = graph.coordinates[current]
        for candidate_index, direction_index in enumerate(candidate_directions):
            edge_length = float(graph.edge_lengths[current, direction_index])
            start_time = provisional_delay + cumulative_length / scenario.speed
            end_time = provisional_delay + (
                cumulative_length + edge_length
            ) / scenario.speed
            factors[candidate_index] = _reservation_factor(
                current_point,
                graph.coordinates[candidate_nodes[candidate_index]],
                start_time,
                end_time,
                reservations,
                scenario,
                config,
            )

        desirability = (
            np.power(
                pheromone[agent_index, current, candidate_directions], config.alpha
            )
            * np.power(
                heuristic[agent_index, current, candidate_directions], config.beta
            )
            * factors
        )
        selected_local = _roulette_choice(desirability, rng)
        selected_direction = int(candidate_directions[selected_local])
        target = int(candidate_nodes[selected_local])
        edge_length = float(graph.edge_lengths[current, selected_direction])
        directions.append(selected_direction)
        lengths.append(edge_length)
        nodes.append(target)
        cumulative_length += edge_length
        current = target
        visited[current] = True

    if current != goal:
        return None

    node_array = np.asarray(nodes, dtype=int)
    raw_path = graph.coordinates[node_array]
    simplified = simplify_path_line_of_sight(
        raw_path,
        scenario.obstacles,
        margin=scenario.obstacle_safety_margin,
    )
    return _Route(
        nodes=node_array,
        directions=np.asarray(directions, dtype=int),
        edge_lengths=np.asarray(lengths, dtype=float),
        raw_path=raw_path,
        simplified_path=simplified,
    )


def _delay_penalties(
    path: np.ndarray,
    delay_values: np.ndarray,
    reservations: list[_TimedReservation],
    scenario: MultiAgentScenario,
    config: MultiAgentACOConfig,
) -> np.ndarray:
    if not reservations:
        penalties = 0.15 * delay_values
        penalties[0] = 0.0
        return penalties

    current_duration = path_length(path) / scenario.speed
    horizon = float(delay_values[-1]) + current_duration
    for reservation in reservations:
        horizon = max(
            horizon,
            reservation.start_delay
            + float(reservation.cumulative_lengths[-1]) / scenario.speed,
        )
    count = int(np.ceil(horizon / config.delay_evaluation_step))
    times = np.linspace(0.0, count * config.delay_evaluation_step, count + 1)
    reserved_trajectories = [
        sample_path_trajectory(
            reservation.path,
            reservation.start_delay,
            scenario.speed,
            times,
        )
        for reservation in reservations
    ]

    penalties = np.empty(len(delay_values), dtype=float)
    for delay_index, delay in enumerate(delay_values):
        current_trajectory = sample_path_trajectory(
            path, float(delay), scenario.speed, times
        )
        physical_samples = 0
        separation_samples = 0
        physical_overlap = 0.0
        separation_shortfall = 0.0
        for other_trajectory in reserved_trajectories:
            distances = np.linalg.norm(current_trajectory - other_trajectory, axis=1)
            physical_shortfall = np.clip(
                scenario.physical_agent_collision_distance - distances, 0.0, None
            )
            separation_gap = np.clip(
                scenario.minimum_agent_separation - distances, 0.0, None
            )
            physical_samples += int(np.count_nonzero(physical_shortfall > 0.0))
            separation_samples += int(np.count_nonzero(separation_gap > 0.0))
            physical_overlap += float(np.square(physical_shortfall).sum())
            separation_shortfall += float(np.square(separation_gap).sum())

        penalties[delay_index] = (
            500.0 * physical_samples
            + 20.0 * separation_samples
            + 100.0 * physical_overlap
            + 5.0 * separation_shortfall
            + 0.15 * float(delay)
        )
    return penalties


def _select_delay(
    path: np.ndarray,
    agent_index: int,
    delay_pheromone: np.ndarray,
    delay_values: np.ndarray,
    reservations: list[_TimedReservation],
    scenario: MultiAgentScenario,
    config: MultiAgentACOConfig,
    rng: np.random.Generator,
) -> tuple[float, int]:
    if not reservations:
        return 0.0, 0

    penalties = _delay_penalties(
        path, delay_values, reservations, scenario, config
    )
    heuristic = 1.0 / (1.0 + penalties)
    desirability = np.power(delay_pheromone[agent_index], config.alpha) * np.power(
        heuristic, config.delay_beta
    )
    selected = _roulette_choice(desirability, rng)
    return float(delay_values[selected]), int(selected)


def _construct_joint_ant(
    graph: GridGraph,
    pheromone: np.ndarray,
    delay_pheromone: np.ndarray,
    heuristic: np.ndarray,
    delay_values: np.ndarray,
    scenario: MultiAgentScenario,
    config: MultiAgentACOConfig,
    weights: MultiAgentObjectiveWeights,
    rng: np.random.Generator,
) -> _AntSolution | None:
    priority = rng.permutation(scenario.num_agents)
    routes: list[_Route | None] = [None] * scenario.num_agents
    delays = np.zeros(scenario.num_agents, dtype=float)
    selected_delay_bins = np.zeros(scenario.num_agents, dtype=int)
    reservations: list[_TimedReservation] = []

    for order_index, agent_index_value in enumerate(priority):
        agent_index = int(agent_index_value)
        provisional_delay = 0.0 if order_index == 0 else _sample_provisional_delay(
            delay_pheromone, delay_values, agent_index, config, rng
        )
        route = _construct_route(
            agent_index,
            graph,
            pheromone,
            heuristic,
            provisional_delay,
            reservations,
            scenario,
            config,
            rng,
        )
        if route is None:
            return None
        delay, delay_bin = _select_delay(
            route.simplified_path,
            agent_index,
            delay_pheromone,
            delay_values,
            reservations,
            scenario,
            config,
            rng,
        )
        routes[agent_index] = route
        delays[agent_index] = delay
        selected_delay_bins[agent_index] = delay_bin
        reservations.append(
            _TimedReservation.from_path(route.simplified_path, delay)
        )

    if any(route is None for route in routes):
        return None
    normalized_delays = normalize_start_delays(delays, scenario.max_start_delay)
    complete_routes = tuple(route for route in routes if route is not None)
    plan = JointPlan(
        paths=tuple(route.simplified_path for route in complete_routes),
        start_delays=normalized_delays,
    )
    objective = evaluate_joint_plan(plan, scenario, weights)
    metrics = compute_joint_plan_metrics(plan, scenario)
    normalized_bins = np.rint(normalized_delays / config.delay_step).astype(int)
    normalized_bins = np.clip(normalized_bins, 0, len(delay_values) - 1)
    return _AntSolution(
        plan=plan,
        routes=complete_routes,
        delay_bins=normalized_bins,
        priority=priority,
        fitness=float(objective.total),
        objective=objective,
        metrics=metrics,
    )


def _node_pheromone_map(
    pheromone: np.ndarray,
    graph: GridGraph,
) -> np.ndarray:
    valid_edges = graph.neighbours >= 0
    return np.stack(
        [
            np.where(valid_edges, pheromone[agent_index], 0.0)
            .sum(axis=1)
            .reshape(graph.rows, graph.cols)
            for agent_index in range(pheromone.shape[0])
        ],
        axis=0,
    )


def _deposit_solution(
    pheromone: np.ndarray,
    delay_pheromone: np.ndarray,
    solution: _AntSolution,
    amount: float,
) -> None:
    for agent_index, route in enumerate(solution.routes):
        sources = route.nodes[:-1]
        targets = route.nodes[1:]
        directions = route.directions
        reverse = _OPPOSITE_DIRECTION[directions]
        pheromone[agent_index, sources, directions] += amount
        pheromone[agent_index, targets, reverse] += amount
        delay_pheromone[agent_index, solution.delay_bins[agent_index]] += amount


def _update_pheromone(
    pheromone: np.ndarray,
    delay_pheromone: np.ndarray,
    graph: GridGraph,
    solutions: list[_AntSolution],
    global_best: _AntSolution | None,
    config: MultiAgentACOConfig,
) -> None:
    valid_edges = graph.neighbours >= 0
    pheromone[:, valid_edges] *= 1.0 - config.evaporation_rate
    pheromone[:, valid_edges] = np.maximum(
        pheromone[:, valid_edges], config.minimum_pheromone
    )
    delay_pheromone *= 1.0 - config.evaporation_rate
    np.maximum(delay_pheromone, config.minimum_pheromone, out=delay_pheromone)

    if solutions:
        elite_count = max(1, int(np.ceil(config.elite_fraction * len(solutions))))
        for rank, solution in enumerate(solutions[:elite_count]):
            rank_weight = (elite_count - rank) / elite_count
            feasibility_weight = 2.0 if solution.metrics.success else 1.0
            amount = (
                config.pheromone_deposit
                * rank_weight
                * feasibility_weight
                / max(solution.fitness, 1e-12)
            )
            _deposit_solution(pheromone, delay_pheromone, solution, amount)

    if global_best is not None and config.global_best_weight > 0.0:
        amount = (
            config.global_best_weight
            * config.pheromone_deposit
            / max(global_best.fitness, 1e-12)
        )
        _deposit_solution(pheromone, delay_pheromone, global_best, amount)


def run_multi_agent_aco(
    scenario: MultiAgentScenario,
    config: MultiAgentACOConfig = MultiAgentACOConfig(),
    *,
    weights: MultiAgentObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> MultiAgentACOResult:
    """Construct and reinforce reservation-aware joint plans with Ant System."""

    rng = np.random.default_rng(seed)
    start_time = perf_counter()
    graph = build_grid_graph(scenario, config)
    heuristic = _heuristic_tensor(graph, scenario)
    delay_values = np.arange(
        0.0,
        scenario.max_start_delay + 0.5 * config.delay_step,
        config.delay_step,
        dtype=float,
    )

    pheromone = np.zeros(
        (scenario.num_agents, graph.num_nodes, len(_DIRECTIONS)), dtype=float
    )
    valid_edges = graph.neighbours >= 0
    pheromone[:, valid_edges] = config.initial_pheromone
    delay_pheromone = np.full(
        (scenario.num_agents, len(delay_values)),
        config.initial_pheromone,
        dtype=float,
    )

    global_best: _AntSolution | None = None
    fitness_history: list[float] = []
    best_plan_history: list[JointPlan] = []
    pheromone_history: list[np.ndarray] = []
    colony_plan_history: list[tuple[JointPlan, ...]] = []
    successful_candidates = 0
    evaluations = 0

    colony_count = config.max_iterations + 1
    for _ in range(colony_count):
        solutions: list[_AntSolution] = []
        for _ant_index in range(config.num_ants):
            solution = _construct_joint_ant(
                graph,
                pheromone,
                delay_pheromone,
                heuristic,
                delay_values,
                scenario,
                config,
                weights,
                rng,
            )
            evaluations += 1
            if solution is not None:
                solutions.append(solution)
                successful_candidates += int(solution.metrics.success)

        solutions.sort(key=lambda item: item.fitness)
        if solutions and (
            global_best is None or solutions[0].fitness < global_best.fitness
        ):
            global_best = solutions[0]

        _update_pheromone(
            pheromone,
            delay_pheromone,
            graph,
            solutions,
            global_best,
            config,
        )
        pheromone_history.append(_node_pheromone_map(pheromone, graph).astype(np.float32))
        colony_plan_history.append(
            tuple(
                solution.plan
                for solution in solutions[: config.recorded_colony_plans]
            )
        )
        fitness_history.append(
            float(global_best.fitness) if global_best is not None else float("inf")
        )
        if global_best is None:
            empty_paths = tuple(
                np.vstack([task.start_array, task.goal_array])
                for task in scenario.tasks
            )
            best_plan_history.append(
                JointPlan(paths=empty_paths, start_delays=np.zeros(scenario.num_agents))
            )
        else:
            best_plan_history.append(global_best.plan)

    runtime = perf_counter() - start_time
    if global_best is None:
        raise RuntimeError(
            "ACO did not construct any complete joint plan. Increase ants or revise grid settings."
        )

    return MultiAgentACOResult(
        algorithm="ACO",
        plan=global_best.plan,
        raw_paths=tuple(route.raw_path for route in global_best.routes),
        best_fitness=float(global_best.fitness),
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_plan_history=tuple(best_plan_history),
        pheromone_history=np.asarray(pheromone_history, dtype=np.float32),
        colony_plan_history=tuple(colony_plan_history),
        iterations=config.max_iterations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=global_best.objective,
        metrics=global_best.metrics,
        successful_candidates=successful_candidates,
        graph_rows=graph.rows,
        graph_cols=graph.cols,
        grid_resolution=config.grid_resolution,
        delay_step=config.delay_step,
    )
