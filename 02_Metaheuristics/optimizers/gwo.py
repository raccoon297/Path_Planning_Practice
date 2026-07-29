"""Grey Wolf Optimizer for continuous waypoint path planning.

The position update follows the original GWO equations proposed by Mirjalili,
Mirjalili, and Lewis. In this project, one wolf represents one complete path:
its position vector stores all intermediate waypoint coordinates. Alpha, beta,
and delta are the three best path candidates used to guide the remaining
population.
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
from utils.metrics import PathMetrics, compute_path_metrics
from utils.objective import ObjectiveResult, evaluate_waypoint_vector
from utils.path_utils import vector_to_path


@dataclass(frozen=True)
class GWOConfig:
    """Hyperparameters for the waypoint-based GWO baseline."""

    num_wolves: int = 80
    max_iterations: int = 200
    initialization_noise_fraction: float = 0.18

    def __post_init__(self) -> None:
        if self.num_wolves < 3:
            raise ValueError("num_wolves must be at least 3.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.initialization_noise_fraction < 0:
            raise ValueError("initialization_noise_fraction cannot be negative.")


@dataclass(frozen=True)
class GWOResult:
    """Commonly useful outputs from one GWO optimization run."""

    algorithm: str
    path: np.ndarray
    best_vector: np.ndarray
    best_fitness: float
    fitness_history: np.ndarray
    best_vector_history: np.ndarray
    population_history: np.ndarray
    leader_vector_history: np.ndarray
    iterations: int
    evaluations: int
    runtime: float
    seed: int
    objective: ObjectiveResult
    metrics: PathMetrics

    @property
    def success(self) -> bool:
        return self.metrics.success


def _straight_line_waypoints(scenario: Scenario) -> np.ndarray:
    """Return evenly spaced intermediate waypoints from start to goal."""

    fractions = np.linspace(0.0, 1.0, scenario.num_waypoints + 2)[1:-1]
    points = scenario.start_array + fractions[:, None] * (
        scenario.goal_array - scenario.start_array
    )
    return points.reshape(-1)


def _initialize_pack(
    scenario: Scenario,
    config: GWOConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize wolves around the start-goal corridor.

    The first wolf is the unperturbed straight-line path. Remaining wolves are
    Gaussian perturbations around that path. This is a path-planning
    initialization choice; the GWO leader hierarchy and update equations are
    unchanged.
    """

    lower_bounds, upper_bounds = scenario.waypoint_bounds()
    base = _straight_line_waypoints(scenario)
    coordinate_scale = np.tile(
        np.array([scenario.width, scenario.height], dtype=float),
        scenario.num_waypoints,
    )
    noise_scale = config.initialization_noise_fraction * coordinate_scale

    positions = np.empty((config.num_wolves, scenario.dimension), dtype=float)
    positions[0] = base
    if config.num_wolves > 1:
        positions[1:] = base + rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=(config.num_wolves - 1, scenario.dimension),
        )
    return np.clip(positions, lower_bounds, upper_bounds)


def _evaluate_population(
    positions: np.ndarray,
    scenario: Scenario,
    weights: ObjectiveWeights,
) -> np.ndarray:
    """Evaluate the shared objective for every wolf position."""

    return np.asarray(
        [
            evaluate_waypoint_vector(position, scenario, weights).total
            for position in positions
        ],
        dtype=float,
    )


def _leader_indices(fitness: np.ndarray) -> tuple[int, int, int]:
    """Return indices of the alpha, beta, and delta wolves."""

    if fitness.ndim != 1 or fitness.size < 3:
        raise ValueError("fitness must contain at least three scalar values.")
    ranked = np.argsort(fitness, kind="stable")
    return int(ranked[0]), int(ranked[1]), int(ranked[2])


def _control_parameter(iteration: int, config: GWOConfig) -> float:
    """Linearly decrease ``a`` from 2 to 0.

    In the original GWO formulation, values of |A| greater than one encourage
    exploration, while values below one increasingly emphasize exploitation
    around the three leaders.
    """

    if config.max_iterations == 1:
        return 0.0
    progress = iteration / (config.max_iterations - 1)
    return 2.0 * (1.0 - progress)


def _update_positions(
    positions: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: np.ndarray,
    a: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the original three-leader GWO position update."""

    r1_alpha = rng.random(size=positions.shape)
    r2_alpha = rng.random(size=positions.shape)
    a_alpha = 2.0 * a * r1_alpha - a
    c_alpha = 2.0 * r2_alpha
    distance_alpha = np.abs(c_alpha * alpha - positions)
    x1 = alpha - a_alpha * distance_alpha

    r1_beta = rng.random(size=positions.shape)
    r2_beta = rng.random(size=positions.shape)
    a_beta = 2.0 * a * r1_beta - a
    c_beta = 2.0 * r2_beta
    distance_beta = np.abs(c_beta * beta - positions)
    x2 = beta - a_beta * distance_beta

    r1_delta = rng.random(size=positions.shape)
    r2_delta = rng.random(size=positions.shape)
    a_delta = 2.0 * a * r1_delta - a
    c_delta = 2.0 * r2_delta
    distance_delta = np.abs(c_delta * delta - positions)
    x3 = delta - a_delta * distance_delta

    return (x1 + x2 + x3) / 3.0


def run_gwo(
    scenario: Scenario,
    config: GWOConfig = GWOConfig(),
    *,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> GWOResult:
    """Optimize a collision-aware waypoint path with Grey Wolf Optimizer.

    Notes
    -----
    - A wolf position is a flattened vector of intermediate waypoint pairs.
    - Lower objective values are better.
    - Alpha, beta, and delta are selected from the current population before
      each position update.
    - ``fitness_history`` stores the best-so-far fitness, including
      initialization, so it can be compared directly with the PSO history.
    """

    rng = np.random.default_rng(seed)
    lower_bounds, upper_bounds = scenario.waypoint_bounds()

    start_time = thread_time()
    positions = _initialize_pack(scenario, config, rng)
    fitness = _evaluate_population(positions, scenario, weights)
    evaluations = config.num_wolves

    alpha_index, beta_index, delta_index = _leader_indices(fitness)
    alpha_position = positions[alpha_index].copy()
    beta_position = positions[beta_index].copy()
    delta_position = positions[delta_index].copy()

    best_position = alpha_position.copy()
    best_fitness = float(fitness[alpha_index])
    fitness_history = [best_fitness]
    best_vector_history = [best_position.copy()]
    population_history = [positions.copy()]
    leader_vector_history = [
        np.stack([alpha_position, beta_position, delta_position], axis=0)
    ]

    for iteration in range(config.max_iterations):
        a = _control_parameter(iteration, config)
        positions = _update_positions(
            positions,
            alpha_position,
            beta_position,
            delta_position,
            a,
            rng,
        )
        positions = np.clip(positions, lower_bounds, upper_bounds)

        fitness = _evaluate_population(positions, scenario, weights)
        evaluations += config.num_wolves

        alpha_index, beta_index, delta_index = _leader_indices(fitness)
        alpha_position = positions[alpha_index].copy()
        beta_position = positions[beta_index].copy()
        delta_position = positions[delta_index].copy()
        alpha_fitness = float(fitness[alpha_index])

        if alpha_fitness < best_fitness:
            best_fitness = alpha_fitness
            best_position = alpha_position.copy()

        fitness_history.append(best_fitness)
        best_vector_history.append(best_position.copy())
        population_history.append(positions.copy())
        leader_vector_history.append(
            np.stack([alpha_position, beta_position, delta_position], axis=0)
        )

    runtime = thread_time() - start_time
    best_path = vector_to_path(
        best_position,
        start=scenario.start_array,
        goal=scenario.goal_array,
        num_waypoints=scenario.num_waypoints,
    )
    objective = evaluate_waypoint_vector(best_position, scenario, weights)
    metrics = compute_path_metrics(best_path, scenario)

    return GWOResult(
        algorithm="GWO",
        path=best_path,
        best_vector=best_position,
        best_fitness=best_fitness,
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_vector_history=np.asarray(best_vector_history, dtype=float),
        population_history=np.asarray(population_history, dtype=float),
        leader_vector_history=np.asarray(leader_vector_history, dtype=float),
        iterations=config.max_iterations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=objective,
        metrics=metrics,
    )
