"""Particle Swarm Optimization for continuous waypoint path planning.

The PSO update follows the particle-position and particle-velocity formulation
introduced by Kennedy and Eberhart, with the inertia-weight extension described
by Shi and Eberhart. In this project, one particle represents one complete path:
its position vector stores all intermediate waypoint coordinates.
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
class PSOConfig:
    """Hyperparameters for the waypoint-based PSO baseline."""

    num_particles: int = 80
    max_iterations: int = 200
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    cognitive_coefficient: float = 2.0
    social_coefficient: float = 2.0
    velocity_fraction: float = 0.20
    initialization_noise_fraction: float = 0.18

    def __post_init__(self) -> None:
        if self.num_particles < 2:
            raise ValueError("num_particles must be at least 2.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.inertia_start < 0 or self.inertia_end < 0:
            raise ValueError("Inertia weights cannot be negative.")
        if self.cognitive_coefficient < 0 or self.social_coefficient < 0:
            raise ValueError("PSO acceleration coefficients cannot be negative.")
        if not (0.0 < self.velocity_fraction <= 1.0):
            raise ValueError("velocity_fraction must be in (0, 1].")
        if self.initialization_noise_fraction < 0:
            raise ValueError("initialization_noise_fraction cannot be negative.")


@dataclass(frozen=True)
class PSOResult:
    """Commonly useful outputs from one PSO optimization run."""

    algorithm: str
    path: np.ndarray
    best_vector: np.ndarray
    best_fitness: float
    fitness_history: np.ndarray
    best_vector_history: np.ndarray
    population_history: np.ndarray
    personal_best_history: np.ndarray
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


def _initialize_swarm(
    scenario: Scenario,
    config: PSOConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize particles around the start-goal corridor.

    The first particle is the unperturbed straight-line path. Remaining
    particles are Gaussian perturbations around that path. This is a
    path-planning initialization choice, not a modification of the PSO update.
    """

    lower_bounds, upper_bounds = scenario.waypoint_bounds()
    base = _straight_line_waypoints(scenario)
    coordinate_scale = np.tile(
        np.array([scenario.width, scenario.height], dtype=float),
        scenario.num_waypoints,
    )
    noise_scale = config.initialization_noise_fraction * coordinate_scale

    positions = np.empty((config.num_particles, scenario.dimension), dtype=float)
    positions[0] = base
    if config.num_particles > 1:
        positions[1:] = base + rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=(config.num_particles - 1, scenario.dimension),
        )
    positions = np.clip(positions, lower_bounds, upper_bounds)

    max_velocity = config.velocity_fraction * (upper_bounds - lower_bounds)
    velocities = rng.uniform(
        low=-max_velocity,
        high=max_velocity,
        size=(config.num_particles, scenario.dimension),
    )
    return positions, velocities


def _evaluate_population(
    positions: np.ndarray,
    scenario: Scenario,
    weights: ObjectiveWeights,
) -> np.ndarray:
    """Evaluate the shared objective for every particle position."""

    return np.asarray(
        [
            evaluate_waypoint_vector(position, scenario, weights).total
            for position in positions
        ],
        dtype=float,
    )


def _inertia_weight(iteration: int, config: PSOConfig) -> float:
    """Linearly decrease inertia from exploration to exploitation."""

    if config.max_iterations == 1:
        return config.inertia_end
    progress = iteration / (config.max_iterations - 1)
    return config.inertia_start + progress * (
        config.inertia_end - config.inertia_start
    )


def run_pso(
    scenario: Scenario,
    config: PSOConfig = PSOConfig(),
    *,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> PSOResult:
    """Optimize a collision-aware waypoint path with Particle Swarm Optimization.

    Notes
    -----
    - A particle position is a flattened vector of intermediate waypoint pairs.
    - Lower objective values are better.
    - Search bounds are enforced as hard coordinate constraints.
    - ``fitness_history`` stores best-so-far fitness, including initialization.
    """

    rng = np.random.default_rng(seed)
    lower_bounds, upper_bounds = scenario.waypoint_bounds()
    max_velocity = config.velocity_fraction * (upper_bounds - lower_bounds)

    start_time = thread_time()
    positions, velocities = _initialize_swarm(scenario, config, rng)
    fitness = _evaluate_population(positions, scenario, weights)
    evaluations = config.num_particles

    personal_best_positions = positions.copy()
    personal_best_fitness = fitness.copy()

    global_best_index = int(np.argmin(personal_best_fitness))
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_fitness = float(personal_best_fitness[global_best_index])

    fitness_history = [global_best_fitness]
    best_vector_history = [global_best_position.copy()]
    population_history = [positions.copy()]
    personal_best_history = [personal_best_positions.copy()]

    for iteration in range(config.max_iterations):
        inertia = _inertia_weight(iteration, config)
        random_cognitive = rng.random(size=positions.shape)
        random_social = rng.random(size=positions.shape)

        velocities = (
            inertia * velocities
            + config.cognitive_coefficient
            * random_cognitive
            * (personal_best_positions - positions)
            + config.social_coefficient
            * random_social
            * (global_best_position - positions)
        )
        velocities = np.clip(velocities, -max_velocity, max_velocity)

        proposed_positions = positions + velocities
        clipped_positions = np.clip(proposed_positions, lower_bounds, upper_bounds)

        # A clipped coordinate has hit a hard map boundary. Removing its outward
        # velocity avoids repeatedly pushing the particle beyond the same bound.
        hit_boundary = proposed_positions != clipped_positions
        velocities[hit_boundary] = 0.0
        positions = clipped_positions

        fitness = _evaluate_population(positions, scenario, weights)
        evaluations += config.num_particles

        improved = fitness < personal_best_fitness
        personal_best_fitness[improved] = fitness[improved]
        personal_best_positions[improved] = positions[improved]

        candidate_index = int(np.argmin(personal_best_fitness))
        candidate_fitness = float(personal_best_fitness[candidate_index])
        if candidate_fitness < global_best_fitness:
            global_best_fitness = candidate_fitness
            global_best_position = personal_best_positions[candidate_index].copy()

        fitness_history.append(global_best_fitness)
        best_vector_history.append(global_best_position.copy())
        population_history.append(positions.copy())
        personal_best_history.append(personal_best_positions.copy())

    runtime = thread_time() - start_time
    best_path = vector_to_path(
        global_best_position,
        start=scenario.start_array,
        goal=scenario.goal_array,
        num_waypoints=scenario.num_waypoints,
    )
    objective = evaluate_waypoint_vector(global_best_position, scenario, weights)
    metrics = compute_path_metrics(best_path, scenario)

    return PSOResult(
        algorithm="PSO",
        path=best_path,
        best_vector=global_best_position,
        best_fitness=global_best_fitness,
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_vector_history=np.asarray(best_vector_history, dtype=float),
        population_history=np.asarray(population_history, dtype=float),
        personal_best_history=np.asarray(personal_best_history, dtype=float),
        iterations=config.max_iterations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=objective,
        metrics=metrics,
    )
