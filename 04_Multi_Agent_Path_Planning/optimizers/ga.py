"""Genetic Algorithm for centralized multi-agent path planning.

One chromosome represents one complete joint plan containing every agent's
intermediate waypoint coordinates and normalized start delay. The GA preserves
that structure with two complementary crossover modes: complete agent-plan
exchange and waypoint-block exchange inside each agent plan.
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
from utils.metrics import JointPlanMetrics, compute_joint_plan_metrics
from utils.objective import JointObjectiveResult, evaluate_joint_vector
from utils.path_utils import JointPlan, decode_joint_vector


@dataclass(frozen=True)
class MultiAgentGAConfig:
    """Hyperparameters for the centralized joint-plan real-coded GA."""

    population_size: int = 80
    max_generations: int = 150
    tournament_size: int = 3
    crossover_rate: float = 0.90
    agent_block_crossover_probability: float = 0.40
    waypoint_mutation_rate: float = 0.30
    delay_mutation_rate: float = 0.40
    mutation_sigma_start: float = 0.10
    mutation_sigma_end: float = 0.02
    delay_sigma_start: float = 0.15
    delay_sigma_end: float = 0.03
    elite_count: int = 4
    corridor_fraction: float = 0.85
    maximum_curve_offset_fraction: float = 0.28
    waypoint_noise_fraction: float = 0.06

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4.")
        if self.max_generations < 1:
            raise ValueError("max_generations must be positive.")
        if not (2 <= self.tournament_size <= self.population_size):
            raise ValueError("tournament_size must be between 2 and population_size.")
        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1].")
        if not (0.0 <= self.agent_block_crossover_probability <= 1.0):
            raise ValueError(
                "agent_block_crossover_probability must be in [0, 1]."
            )
        if not (0.0 <= self.waypoint_mutation_rate <= 1.0):
            raise ValueError("waypoint_mutation_rate must be in [0, 1].")
        if not (0.0 <= self.delay_mutation_rate <= 1.0):
            raise ValueError("delay_mutation_rate must be in [0, 1].")
        if self.mutation_sigma_start < 0.0 or self.mutation_sigma_end < 0.0:
            raise ValueError("Waypoint mutation sigmas cannot be negative.")
        if self.delay_sigma_start < 0.0 or self.delay_sigma_end < 0.0:
            raise ValueError("Delay mutation sigmas cannot be negative.")
        if not (0 <= self.elite_count < self.population_size):
            raise ValueError("elite_count must be in [0, population_size).")
        if not (0.0 <= self.corridor_fraction <= 1.0):
            raise ValueError("corridor_fraction must be in [0, 1].")
        if self.maximum_curve_offset_fraction < 0.0:
            raise ValueError("maximum_curve_offset_fraction cannot be negative.")
        if self.waypoint_noise_fraction < 0.0:
            raise ValueError("waypoint_noise_fraction cannot be negative.")


@dataclass(frozen=True)
class MultiAgentGAResult:
    """Outputs from one centralized multi-agent GA run."""

    algorithm: str
    plan: JointPlan
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
    objective: JointObjectiveResult
    metrics: JointPlanMetrics

    @property
    def success(self) -> bool:
        return self.metrics.success


def _straight_line_waypoints(
    start: np.ndarray,
    goal: np.ndarray,
    num_waypoints: int,
) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, num_waypoints + 2)[1:-1]
    return start + fractions[:, None] * (goal - start)


def _curved_corridor_waypoints(
    start: np.ndarray,
    goal: np.ndarray,
    num_waypoints: int,
    offset: float,
    noise_scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    fractions = np.linspace(0.0, 1.0, num_waypoints + 2)[1:-1]
    direct = start + fractions[:, None] * (goal - start)
    direction = goal - start
    norm = float(np.linalg.norm(direction))
    normal = (
        np.array([0.0, 1.0], dtype=float)
        if norm <= 1e-12
        else np.array([-direction[1], direction[0]], dtype=float) / norm
    )
    curve = np.sin(np.pi * fractions)[:, None] * normal[None, :] * offset
    noise = rng.normal(0.0, noise_scale, size=(num_waypoints, 2))
    return direct + curve + noise


def _delay_indices(scenario: MultiAgentScenario) -> np.ndarray:
    return np.arange(
        scenario.agent_block_dimension - 1,
        scenario.dimension,
        scenario.agent_block_dimension,
    )


def _normalize_delay_genes(
    population: np.ndarray,
    scenario: MultiAgentScenario,
) -> np.ndarray:
    """Remove each chromosome's common delay offset in-place."""

    indices = _delay_indices(scenario)
    delays = population[:, indices]
    delays -= delays.min(axis=1, keepdims=True)
    np.clip(delays, 0.0, scenario.max_start_delay, out=delays)
    population[:, indices] = delays
    return population


def _initialize_population(
    scenario: MultiAgentScenario,
    config: MultiAgentGAConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize joint plans with corridor-biased and uniform chromosomes."""

    lower_bounds, upper_bounds = scenario.candidate_bounds()
    population = np.empty((config.population_size, scenario.dimension), dtype=float)
    map_scale = np.array([scenario.width, scenario.height], dtype=float)
    noise_scale = config.waypoint_noise_fraction * map_scale
    maximum_offset = config.maximum_curve_offset_fraction * min(
        scenario.width, scenario.height
    )
    corridor_count = max(
        1,
        min(
            config.population_size,
            int(round(config.population_size * config.corridor_fraction)),
        ),
    )

    blocks: list[np.ndarray] = []
    initial_delays = np.linspace(0.0, scenario.max_start_delay, scenario.num_agents)
    for index, task in enumerate(scenario.tasks):
        waypoints = _straight_line_waypoints(
            task.start_array, task.goal_array, scenario.num_waypoints
        )
        blocks.append(np.concatenate([waypoints.reshape(-1), [initial_delays[index]]]))
    population[0] = np.concatenate(blocks)

    for chromosome_index in range(1, corridor_count):
        blocks = []
        for task in scenario.tasks:
            offset = rng.uniform(-maximum_offset, maximum_offset)
            waypoints = _curved_corridor_waypoints(
                task.start_array,
                task.goal_array,
                scenario.num_waypoints,
                offset,
                noise_scale,
                rng,
            )
            delay = rng.uniform(0.0, scenario.max_start_delay)
            blocks.append(np.concatenate([waypoints.reshape(-1), [delay]]))
        population[chromosome_index] = np.concatenate(blocks)

    if corridor_count < config.population_size:
        population[corridor_count:] = rng.uniform(
            lower_bounds,
            upper_bounds,
            size=(config.population_size - corridor_count, scenario.dimension),
        )

    population = np.clip(population, lower_bounds, upper_bounds)
    return _normalize_delay_genes(population, scenario)


def _evaluate_population(
    population: np.ndarray,
    scenario: MultiAgentScenario,
    weights: MultiAgentObjectiveWeights,
) -> np.ndarray:
    return np.asarray(
        [evaluate_joint_vector(chromosome, scenario, weights).total for chromosome in population],
        dtype=float,
    )


def _tournament_select(
    population: np.ndarray,
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    indices = rng.integers(0, len(population), size=tournament_size)
    winner = indices[int(np.argmin(fitness[indices]))]
    return population[winner].copy()


def _agent_block_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    scenario: MultiAgentScenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Exchange one or more complete agent plans between two parents."""

    child_a = parent_a.copy()
    child_b = parent_b.copy()
    block_size = scenario.agent_block_dimension
    maximum_swaps = max(1, scenario.num_agents - 1)
    swap_count = int(rng.integers(1, maximum_swaps + 1))
    selected_agents = rng.choice(
        scenario.num_agents, size=swap_count, replace=False
    )
    for agent_index in np.atleast_1d(selected_agents):
        left = int(agent_index) * block_size
        right = left + block_size
        child_a[left:right], child_b[left:right] = (
            parent_b[left:right],
            parent_a[left:right],
        )
    return child_a, child_b


def _waypoint_block_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    scenario: MultiAgentScenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Exchange contiguous waypoint blocks while blending delay genes."""

    child_a = parent_a.copy()
    child_b = parent_b.copy()
    block_size = scenario.agent_block_dimension
    waypoint_values = 2 * scenario.num_waypoints

    for agent_index in range(scenario.num_agents):
        left = agent_index * block_size
        points_a = parent_a[left : left + waypoint_values].reshape(
            scenario.num_waypoints, 2
        )
        points_b = parent_b[left : left + waypoint_values].reshape(
            scenario.num_waypoints, 2
        )
        offspring_a = points_a.copy()
        offspring_b = points_b.copy()

        if scenario.num_waypoints == 1:
            cut_left, cut_right = 0, 1
        elif scenario.num_waypoints == 2:
            cut_left, cut_right = 1, 2
        else:
            cuts = np.sort(
                rng.choice(
                    np.arange(1, scenario.num_waypoints), size=2, replace=False
                )
            )
            cut_left, cut_right = int(cuts[0]), int(cuts[1])
            if cut_left == cut_right:
                cut_right = min(scenario.num_waypoints, cut_left + 1)

        offspring_a[cut_left:cut_right] = points_b[cut_left:cut_right]
        offspring_b[cut_left:cut_right] = points_a[cut_left:cut_right]
        child_a[left : left + waypoint_values] = offspring_a.reshape(-1)
        child_b[left : left + waypoint_values] = offspring_b.reshape(-1)

        delay_a = float(parent_a[left + waypoint_values])
        delay_b = float(parent_b[left + waypoint_values])
        blend = float(rng.random())
        child_a[left + waypoint_values] = blend * delay_a + (1.0 - blend) * delay_b
        child_b[left + waypoint_values] = blend * delay_b + (1.0 - blend) * delay_a

    return child_a, child_b


def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    scenario: MultiAgentScenario,
    config: MultiAgentGAConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() >= config.crossover_rate:
        return parent_a.copy(), parent_b.copy()
    if rng.random() < config.agent_block_crossover_probability:
        return _agent_block_crossover(parent_a, parent_b, scenario, rng)
    return _waypoint_block_crossover(parent_a, parent_b, scenario, rng)


def _scheduled_value(
    generation: int,
    maximum_generations: int,
    start: float,
    end: float,
) -> float:
    if maximum_generations == 1:
        return end
    progress = generation / (maximum_generations - 1)
    return start + progress * (end - start)


def _mutate(
    chromosome: np.ndarray,
    scenario: MultiAgentScenario,
    config: MultiAgentGAConfig,
    generation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mutate waypoint pairs and delay genes with separate scales."""

    mutated = chromosome.copy()
    block_size = scenario.agent_block_dimension
    waypoint_values = 2 * scenario.num_waypoints
    waypoint_sigma_fraction = _scheduled_value(
        generation,
        config.max_generations,
        config.mutation_sigma_start,
        config.mutation_sigma_end,
    )
    delay_sigma_fraction = _scheduled_value(
        generation,
        config.max_generations,
        config.delay_sigma_start,
        config.delay_sigma_end,
    )
    waypoint_sigma = waypoint_sigma_fraction * np.array(
        [scenario.width, scenario.height], dtype=float
    )
    delay_sigma = delay_sigma_fraction * scenario.max_start_delay

    for agent_index in range(scenario.num_agents):
        left = agent_index * block_size
        points = mutated[left : left + waypoint_values].reshape(
            scenario.num_waypoints, 2
        )
        mutation_mask = (
            rng.random(scenario.num_waypoints) < config.waypoint_mutation_rate
        )
        if np.any(mutation_mask):
            points[mutation_mask] += rng.normal(
                0.0,
                waypoint_sigma,
                size=(int(np.count_nonzero(mutation_mask)), 2),
            )
        mutated[left : left + waypoint_values] = points.reshape(-1)

        if rng.random() < config.delay_mutation_rate:
            mutated[left + waypoint_values] += rng.normal(0.0, delay_sigma)

    lower_bounds, upper_bounds = scenario.candidate_bounds()
    mutated = np.clip(mutated, lower_bounds, upper_bounds)
    return _normalize_delay_genes(mutated[None, :], scenario)[0]


def _create_next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    scenario: MultiAgentScenario,
    config: MultiAgentGAConfig,
    generation: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ranked = np.argsort(fitness, kind="stable")
    offspring: list[np.ndarray] = [
        population[index].copy() for index in ranked[: config.elite_count]
    ]

    while len(offspring) < config.population_size:
        parent_a = _tournament_select(
            population, fitness, config.tournament_size, rng
        )
        parent_b = _tournament_select(
            population, fitness, config.tournament_size, rng
        )
        child_a, child_b = _crossover(
            parent_a, parent_b, scenario, config, rng
        )
        child_a = _mutate(child_a, scenario, config, generation, rng)
        offspring.append(child_a)
        if len(offspring) < config.population_size:
            child_b = _mutate(child_b, scenario, config, generation, rng)
            offspring.append(child_b)

    return np.asarray(offspring, dtype=float)


def run_multi_agent_ga(
    scenario: MultiAgentScenario,
    config: MultiAgentGAConfig = MultiAgentGAConfig(),
    *,
    weights: MultiAgentObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> MultiAgentGAResult:
    """Evolve joint spatial paths and normalized departure delays with GA."""

    rng = np.random.default_rng(seed)
    start_time = perf_counter()
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

    runtime = perf_counter() - start_time
    best_plan = decode_joint_vector(best_vector, scenario)
    objective = evaluate_joint_vector(best_vector, scenario, weights)
    metrics = compute_joint_plan_metrics(best_plan, scenario)

    return MultiAgentGAResult(
        algorithm="Multi-Agent GA",
        plan=best_plan,
        best_vector=best_vector,
        best_fitness=best_fitness,
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_vector_history=np.asarray(best_vector_history, dtype=float),
        population_history=np.asarray(population_history, dtype=np.float32),
        population_fitness_history=np.asarray(
            population_fitness_history, dtype=np.float32
        ),
        elite_count=config.elite_count,
        iterations=config.max_generations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=objective,
        metrics=metrics,
    )
