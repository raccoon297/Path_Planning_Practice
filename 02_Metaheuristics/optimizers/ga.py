"""Real-coded Genetic Algorithm for continuous waypoint path planning.

The optimizer uses the standard genetic-algorithm cycle of selection,
crossover, mutation, and elitist survival. In this project, one chromosome
represents one complete path: its real-valued genes store all intermediate
waypoint coordinates.
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
class GAConfig:
    """Hyperparameters for the waypoint-based real-coded GA baseline."""

    population_size: int = 80
    max_generations: int = 200
    tournament_size: int = 3
    crossover_rate: float = 0.90
    mutation_rate: float = 0.35
    mutation_sigma_start: float = 0.12
    mutation_sigma_end: float = 0.02
    elite_count: int = 4
    initialization_noise_fraction: float = 0.18
    uniform_initialization_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4.")
        if self.max_generations < 1:
            raise ValueError("max_generations must be positive.")
        if not (2 <= self.tournament_size <= self.population_size):
            raise ValueError("tournament_size must be between 2 and population_size.")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1].")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1].")
        if self.mutation_sigma_start < 0 or self.mutation_sigma_end < 0:
            raise ValueError("Mutation sigma fractions cannot be negative.")
        if not (0 <= self.elite_count < self.population_size):
            raise ValueError("elite_count must be in [0, population_size).")
        if self.initialization_noise_fraction < 0:
            raise ValueError("initialization_noise_fraction cannot be negative.")
        if not (0.0 <= self.uniform_initialization_fraction <= 1.0):
            raise ValueError("uniform_initialization_fraction must be in [0, 1].")


@dataclass(frozen=True)
class GAResult:
    """Commonly useful outputs from one GA optimization run."""

    algorithm: str
    path: np.ndarray
    best_vector: np.ndarray
    best_fitness: float
    fitness_history: np.ndarray
    best_vector_history: np.ndarray
    population_history: np.ndarray
    population_fitness_history: np.ndarray
    elite_count: int
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


def _initialize_population(
    scenario: Scenario,
    config: GAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize chromosomes around the start-goal corridor with diversity.

    Most chromosomes are Gaussian perturbations around the direct route. A
    small fraction is sampled uniformly over the map to preserve exploration.
    This is a path-planning initialization choice; it does not change the GA
    selection, crossover, or mutation mechanisms.
    """

    lower_bounds, upper_bounds = scenario.waypoint_bounds()
    base = _straight_line_waypoints(scenario)
    coordinate_scale = np.tile(
        np.array([scenario.width, scenario.height], dtype=float),
        scenario.num_waypoints,
    )
    noise_scale = config.initialization_noise_fraction * coordinate_scale

    population = np.empty((config.population_size, scenario.dimension), dtype=float)
    population[0] = base

    remaining = config.population_size - 1
    uniform_count = int(round(remaining * config.uniform_initialization_fraction))
    corridor_count = remaining - uniform_count

    if corridor_count > 0:
        population[1 : 1 + corridor_count] = base + rng.normal(
            loc=0.0,
            scale=noise_scale,
            size=(corridor_count, scenario.dimension),
        )
    if uniform_count > 0:
        population[1 + corridor_count :] = rng.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(uniform_count, scenario.dimension),
        )

    return np.clip(population, lower_bounds, upper_bounds)


def _evaluate_population(
    population: np.ndarray,
    scenario: Scenario,
    weights: ObjectiveWeights,
) -> np.ndarray:
    """Evaluate the shared objective for every chromosome."""

    return np.asarray(
        [
            evaluate_waypoint_vector(chromosome, scenario, weights).total
            for chromosome in population
        ],
        dtype=float,
    )


def _tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one copied parent selected by minimization tournament."""

    indices = rng.integers(0, len(population), size=tournament_size)
    winner = indices[int(np.argmin(fitness[indices]))]
    return population[winner].copy()


def _waypoint_two_point_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    num_waypoints: int,
    crossover_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Exchange complete waypoint blocks between two real-valued parents."""

    child_a = parent_a.copy()
    child_b = parent_b.copy()
    if num_waypoints < 2 or rng.random() >= crossover_rate:
        return child_a, child_b

    parent_a_points = parent_a.reshape(num_waypoints, 2)
    parent_b_points = parent_b.reshape(num_waypoints, 2)
    child_a_points = parent_a_points.copy()
    child_b_points = parent_b_points.copy()

    if num_waypoints == 2:
        left, right = 1, 2
    else:
        cuts = np.sort(rng.choice(np.arange(1, num_waypoints), size=2, replace=False))
        left, right = int(cuts[0]), int(cuts[1])
        if left == right:
            right = min(num_waypoints, left + 1)

    child_a_points[left:right] = parent_b_points[left:right]
    child_b_points[left:right] = parent_a_points[left:right]
    return child_a_points.reshape(-1), child_b_points.reshape(-1)


def _mutation_sigma(generation: int, config: GAConfig) -> float:
    """Linearly decrease Gaussian mutation scale during optimization."""

    if config.max_generations == 1:
        return config.mutation_sigma_end
    progress = generation / (config.max_generations - 1)
    return config.mutation_sigma_start + progress * (
        config.mutation_sigma_end - config.mutation_sigma_start
    )


def _mutate_waypoints(
    chromosome: np.ndarray,
    scenario: Scenario,
    mutation_rate: float,
    sigma_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply independent Gaussian mutation to selected waypoint pairs."""

    points = chromosome.reshape(scenario.num_waypoints, 2).copy()
    mutation_mask = rng.random(scenario.num_waypoints) < mutation_rate
    if np.any(mutation_mask):
        sigma = sigma_fraction * np.array([scenario.width, scenario.height], dtype=float)
        points[mutation_mask] += rng.normal(
            loc=0.0,
            scale=sigma,
            size=(int(np.count_nonzero(mutation_mask)), 2),
        )

    lower_bounds, upper_bounds = scenario.waypoint_bounds()
    return np.clip(points.reshape(-1), lower_bounds, upper_bounds)


def _create_next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    scenario: Scenario,
    config: GAConfig,
    generation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create one full generation using elitism and genetic operators."""

    ranked_indices = np.argsort(fitness)
    elites = population[ranked_indices[: config.elite_count]].copy()
    offspring: list[np.ndarray] = [elite.copy() for elite in elites]
    sigma_fraction = _mutation_sigma(generation, config)

    while len(offspring) < config.population_size:
        parent_a = _tournament_select(
            population, fitness, config.tournament_size, rng
        )
        parent_b = _tournament_select(
            population, fitness, config.tournament_size, rng
        )
        child_a, child_b = _waypoint_two_point_crossover(
            parent_a,
            parent_b,
            scenario.num_waypoints,
            config.crossover_rate,
            rng,
        )
        child_a = _mutate_waypoints(
            child_a,
            scenario,
            config.mutation_rate,
            sigma_fraction,
            rng,
        )
        offspring.append(child_a)

        if len(offspring) < config.population_size:
            child_b = _mutate_waypoints(
                child_b,
                scenario,
                config.mutation_rate,
                sigma_fraction,
                rng,
            )
            offspring.append(child_b)

    return np.asarray(offspring, dtype=float)


def run_ga(
    scenario: Scenario,
    config: GAConfig = GAConfig(),
    *,
    weights: ObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> GAResult:
    """Optimize a collision-aware waypoint path with a real-coded GA.

    Notes
    -----
    - A chromosome is a flattened vector of intermediate waypoint pairs.
    - Lower objective values are better.
    - Tournament selection chooses parents.
    - Two-point crossover exchanges complete waypoint blocks.
    - Gaussian mutation perturbs selected waypoint pairs.
    - Elitism copies the best chromosomes into the next generation.
    - ``fitness_history`` stores best-so-far fitness, including initialization.
    """

    rng = np.random.default_rng(seed)

    start_time = thread_time()
    population = _initialize_population(scenario, config, rng)
    fitness = _evaluate_population(population, scenario, weights)
    evaluations = config.population_size

    best_index = int(np.argmin(fitness))
    best_vector = population[best_index].copy()
    best_fitness = float(fitness[best_index])
    fitness_history = [best_fitness]
    best_vector_history = [best_vector.copy()]
    population_history = [population.copy()]
    population_fitness_history = [fitness.copy()]

    for generation in range(config.max_generations):
        population = _create_next_generation(
            population,
            fitness,
            scenario,
            config,
            generation,
            rng,
        )
        fitness = _evaluate_population(population, scenario, weights)
        evaluations += config.population_size

        candidate_index = int(np.argmin(fitness))
        candidate_fitness = float(fitness[candidate_index])
        if candidate_fitness < best_fitness:
            best_fitness = candidate_fitness
            best_vector = population[candidate_index].copy()

        fitness_history.append(best_fitness)
        best_vector_history.append(best_vector.copy())
        population_history.append(population.copy())
        population_fitness_history.append(fitness.copy())

    runtime = thread_time() - start_time
    best_path = vector_to_path(
        best_vector,
        start=scenario.start_array,
        goal=scenario.goal_array,
        num_waypoints=scenario.num_waypoints,
    )
    objective = evaluate_waypoint_vector(best_vector, scenario, weights)
    metrics = compute_path_metrics(best_path, scenario)

    return GAResult(
        algorithm="GA",
        path=best_path,
        best_vector=best_vector,
        best_fitness=best_fitness,
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_vector_history=np.asarray(best_vector_history, dtype=float),
        population_history=np.asarray(population_history, dtype=float),
        population_fitness_history=np.asarray(population_fitness_history, dtype=float),
        elite_count=config.elite_count,
        iterations=config.max_generations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=objective,
        metrics=metrics,
    )
