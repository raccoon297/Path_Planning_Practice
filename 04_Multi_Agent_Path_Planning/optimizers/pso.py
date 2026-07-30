"""Particle Swarm Optimization for centralized multi-agent path planning.

One particle represents one complete joint plan. Its position stores every
agent's intermediate waypoint coordinates together with one start-delay gene
per agent. The standard PSO velocity update therefore improves spatial paths
and temporal departure decisions at the same time.
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
class MultiAgentPSOConfig:
    """Hyperparameters for the centralized joint-plan PSO."""

    num_particles: int = 80
    max_iterations: int = 150
    inertia_start: float = 0.90
    inertia_end: float = 0.40
    cognitive_coefficient: float = 2.0
    social_coefficient: float = 2.0
    velocity_fraction: float = 0.20
    corridor_fraction: float = 0.85
    maximum_curve_offset_fraction: float = 0.28
    waypoint_noise_fraction: float = 0.06

    def __post_init__(self) -> None:
        if self.num_particles < 2:
            raise ValueError("num_particles must be at least 2.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if self.inertia_start < 0.0 or self.inertia_end < 0.0:
            raise ValueError("Inertia weights cannot be negative.")
        if self.cognitive_coefficient < 0.0 or self.social_coefficient < 0.0:
            raise ValueError("PSO acceleration coefficients cannot be negative.")
        if not (0.0 < self.velocity_fraction <= 1.0):
            raise ValueError("velocity_fraction must be in (0, 1].")
        if not (0.0 <= self.corridor_fraction <= 1.0):
            raise ValueError("corridor_fraction must be in [0, 1].")
        if self.maximum_curve_offset_fraction < 0.0:
            raise ValueError("maximum_curve_offset_fraction cannot be negative.")
        if self.waypoint_noise_fraction < 0.0:
            raise ValueError("waypoint_noise_fraction cannot be negative.")


@dataclass(frozen=True)
class MultiAgentPSOResult:
    """Outputs from one centralized multi-agent PSO run."""

    algorithm: str
    plan: JointPlan
    best_vector: np.ndarray
    best_fitness: float
    fitness_history: np.ndarray
    best_vector_history: np.ndarray
    population_history: np.ndarray
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
    """Sample a smooth curved corridor around the direct start-goal route."""

    fractions = np.linspace(0.0, 1.0, num_waypoints + 2)[1:-1]
    direct = start + fractions[:, None] * (goal - start)
    direction = goal - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        normal = np.array([0.0, 1.0], dtype=float)
    else:
        normal = np.array([-direction[1], direction[0]], dtype=float) / norm
    curve = np.sin(np.pi * fractions)[:, None] * normal[None, :] * offset
    noise = rng.normal(0.0, noise_scale, size=(num_waypoints, 2))
    return direct + curve + noise


def _normalize_delay_genes(
    positions: np.ndarray,
    scenario: MultiAgentScenario,
) -> np.ndarray:
    """Remove each particle's common delay offset in-place and return it."""

    delay_indices = np.arange(
        scenario.agent_block_dimension - 1,
        scenario.dimension,
        scenario.agent_block_dimension,
    )
    delays = positions[:, delay_indices]
    delays -= delays.min(axis=1, keepdims=True)
    np.clip(delays, 0.0, scenario.max_start_delay, out=delays)
    positions[:, delay_indices] = delays
    return positions


def _initialize_swarm(
    scenario: MultiAgentScenario,
    config: MultiAgentPSOConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize diverse joint plans around curved start-goal corridors."""

    lower_bounds, upper_bounds = scenario.candidate_bounds()
    positions = np.empty((config.num_particles, scenario.dimension), dtype=float)
    block_size = scenario.agent_block_dimension
    waypoint_values = 2 * scenario.num_waypoints
    map_scale = np.array([scenario.width, scenario.height], dtype=float)
    noise_scale = config.waypoint_noise_fraction * map_scale
    maximum_offset = config.maximum_curve_offset_fraction * min(
        scenario.width, scenario.height
    )

    corridor_count = max(
        1,
        min(
            config.num_particles,
            int(round(config.num_particles * config.corridor_fraction)),
        ),
    )

    # The first particle is the direct joint plan. It deliberately gives PSO a
    # simple baseline but does not inject a hand-crafted feasible solution.
    blocks: list[np.ndarray] = []
    initial_delays = np.linspace(0.0, scenario.max_start_delay, scenario.num_agents)
    for index, task in enumerate(scenario.tasks):
        waypoints = _straight_line_waypoints(
            task.start_array, task.goal_array, scenario.num_waypoints
        )
        blocks.append(np.concatenate([waypoints.reshape(-1), [initial_delays[index]]]))
    positions[0] = np.concatenate(blocks)

    for particle_index in range(1, corridor_count):
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
        positions[particle_index] = np.concatenate(blocks)

    if corridor_count < config.num_particles:
        positions[corridor_count:] = rng.uniform(
            lower_bounds,
            upper_bounds,
            size=(config.num_particles - corridor_count, scenario.dimension),
        )

    positions = np.clip(positions, lower_bounds, upper_bounds)
    positions = _normalize_delay_genes(positions, scenario)

    maximum_velocity = config.velocity_fraction * (upper_bounds - lower_bounds)
    velocities = rng.uniform(
        -maximum_velocity,
        maximum_velocity,
        size=(config.num_particles, scenario.dimension),
    )
    return positions, velocities


def _evaluate_population(
    positions: np.ndarray,
    scenario: MultiAgentScenario,
    weights: MultiAgentObjectiveWeights,
) -> np.ndarray:
    return np.asarray(
        [evaluate_joint_vector(position, scenario, weights).total for position in positions],
        dtype=float,
    )


def _inertia_weight(iteration: int, config: MultiAgentPSOConfig) -> float:
    if config.max_iterations == 1:
        return config.inertia_end
    progress = iteration / (config.max_iterations - 1)
    return config.inertia_start + progress * (
        config.inertia_end - config.inertia_start
    )


def run_multi_agent_pso(
    scenario: MultiAgentScenario,
    config: MultiAgentPSOConfig = MultiAgentPSOConfig(),
    *,
    weights: MultiAgentObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> MultiAgentPSOResult:
    """Optimize spatial paths and normalized start delays with PSO."""

    rng = np.random.default_rng(seed)
    lower_bounds, upper_bounds = scenario.candidate_bounds()
    maximum_velocity = config.velocity_fraction * (upper_bounds - lower_bounds)

    start_time = perf_counter()
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
        velocities = np.clip(velocities, -maximum_velocity, maximum_velocity)

        proposed_positions = positions + velocities
        clipped_positions = np.clip(proposed_positions, lower_bounds, upper_bounds)
        hit_boundary = proposed_positions != clipped_positions
        velocities[hit_boundary] = 0.0
        positions = _normalize_delay_genes(clipped_positions, scenario)

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

    runtime = perf_counter() - start_time
    best_plan = decode_joint_vector(global_best_position, scenario)
    objective = evaluate_joint_vector(global_best_position, scenario, weights)
    metrics = compute_joint_plan_metrics(best_plan, scenario)

    return MultiAgentPSOResult(
        algorithm="Multi-Agent PSO",
        plan=best_plan,
        best_vector=global_best_position,
        best_fitness=global_best_fitness,
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_vector_history=np.asarray(best_vector_history, dtype=float),
        population_history=np.asarray(population_history, dtype=np.float32),
        iterations=config.max_iterations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=objective,
        metrics=metrics,
    )
