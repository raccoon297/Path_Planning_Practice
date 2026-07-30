"""Grey Wolf Optimizer for centralized multi-agent path planning.

One wolf represents one complete joint plan containing every agent's waypoint
coordinates and normalized start delay. Alpha, Beta, and Delta therefore guide
spatial route choices and temporal departure decisions at the same time.
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
class MultiAgentGWOConfig:
    """Hyperparameters for the centralized joint-plan GWO."""

    num_wolves: int = 80
    max_iterations: int = 150
    corridor_fraction: float = 0.85
    maximum_curve_offset_fraction: float = 0.28
    waypoint_noise_fraction: float = 0.06

    def __post_init__(self) -> None:
        if self.num_wolves < 3:
            raise ValueError("num_wolves must be at least 3.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive.")
        if not (0.0 <= self.corridor_fraction <= 1.0):
            raise ValueError("corridor_fraction must be in [0, 1].")
        if self.maximum_curve_offset_fraction < 0.0:
            raise ValueError("maximum_curve_offset_fraction cannot be negative.")
        if self.waypoint_noise_fraction < 0.0:
            raise ValueError("waypoint_noise_fraction cannot be negative.")


@dataclass(frozen=True)
class MultiAgentGWOResult:
    """Outputs from one centralized multi-agent GWO run."""

    algorithm: str
    plan: JointPlan
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
    """Sample a smooth curved corridor around one direct start-goal route."""

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


def _normalize_delay_genes(
    positions: np.ndarray,
    scenario: MultiAgentScenario,
) -> np.ndarray:
    """Remove each wolf's common delay offset in-place and return it."""

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


def _reflect_bounds(
    positions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Reflect out-of-range coordinates back into the feasible interval.

    Unlike clipping, reflection does not collapse many wolves onto exactly the
    same map wall. The periodic form also handles values that overshoot by more
    than one full interval width.
    """

    values = np.asarray(positions, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    span = upper - lower
    if np.any(span <= 0.0):
        raise ValueError("Every search interval must have positive width.")
    shifted = np.mod(values - lower, 2.0 * span)
    return np.where(
        shifted <= span,
        lower + shifted,
        upper - (shifted - span),
    )


def _initialize_pack(
    scenario: MultiAgentScenario,
    config: MultiAgentGWOConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize diverse wolves around smooth start-goal corridors."""

    lower_bounds, upper_bounds = scenario.candidate_bounds()
    positions = np.empty((config.num_wolves, scenario.dimension), dtype=float)
    map_scale = np.array([scenario.width, scenario.height], dtype=float)
    noise_scale = config.waypoint_noise_fraction * map_scale
    maximum_offset = config.maximum_curve_offset_fraction * min(
        scenario.width, scenario.height
    )
    corridor_count = max(
        1,
        min(
            config.num_wolves,
            int(round(config.num_wolves * config.corridor_fraction)),
        ),
    )

    blocks: list[np.ndarray] = []
    initial_delays = np.linspace(0.0, scenario.max_start_delay, scenario.num_agents)
    for index, task in enumerate(scenario.tasks):
        waypoints = _straight_line_waypoints(
            task.start_array, task.goal_array, scenario.num_waypoints
        )
        blocks.append(np.concatenate([waypoints.reshape(-1), [initial_delays[index]]]))
    positions[0] = np.concatenate(blocks)

    for wolf_index in range(1, corridor_count):
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
        positions[wolf_index] = np.concatenate(blocks)

    if corridor_count < config.num_wolves:
        positions[corridor_count:] = rng.uniform(
            lower_bounds,
            upper_bounds,
            size=(config.num_wolves - corridor_count, scenario.dimension),
        )

    positions = _reflect_bounds(positions, lower_bounds, upper_bounds)
    return _normalize_delay_genes(positions, scenario)


def _evaluate_population(
    positions: np.ndarray,
    scenario: MultiAgentScenario,
    weights: MultiAgentObjectiveWeights,
) -> np.ndarray:
    return np.asarray(
        [evaluate_joint_vector(position, scenario, weights).total for position in positions],
        dtype=float,
    )


def _leader_indices(fitness: np.ndarray) -> tuple[int, int, int]:
    if fitness.ndim != 1 or fitness.size < 3:
        raise ValueError("fitness must contain at least three scalar values.")
    ranked = np.argsort(fitness, kind="stable")
    return int(ranked[0]), int(ranked[1]), int(ranked[2])


def _update_leader_archive(
    leader_positions: np.ndarray,
    leader_fitness: np.ndarray,
    positions: np.ndarray,
    fitness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the three best distinct leaders discovered so far.

    The canonical GWO leadership hierarchy is cumulative: a newly evaluated
    wolf replaces Alpha, Beta, or Delta only when it improves the archived
    hierarchy. Combining the previous archive with the current pack prevents
    a good corridor from disappearing after one exploratory update.
    """

    combined_positions = np.vstack([leader_positions, positions])
    combined_fitness = np.concatenate([leader_fitness, fitness])
    ranked = np.argsort(combined_fitness, kind="stable")

    selected_positions: list[np.ndarray] = []
    selected_fitness: list[float] = []
    for index in ranked:
        candidate = combined_positions[int(index)]
        if any(np.allclose(candidate, saved, rtol=0.0, atol=1e-10) for saved in selected_positions):
            continue
        selected_positions.append(candidate.copy())
        selected_fitness.append(float(combined_fitness[int(index)]))
        if len(selected_positions) == 3:
            break

    if len(selected_positions) < 3:
        raise RuntimeError("GWO leader archive could not retain three distinct wolves.")
    return np.asarray(selected_positions), np.asarray(selected_fitness)


def _control_parameter(iteration: int, config: MultiAgentGWOConfig) -> float:
    """Linearly decrease the standard GWO parameter a from 2 to 0."""

    if config.max_iterations == 1:
        return 0.0
    progress = iteration / (config.max_iterations - 1)
    return 2.0 * (1.0 - progress)


def _leader_guided_position(
    positions: np.ndarray,
    leader: np.ndarray,
    a: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r1 = rng.random(size=positions.shape)
    r2 = rng.random(size=positions.shape)
    coefficient_a = 2.0 * a * r1 - a
    coefficient_c = 2.0 * r2
    distance = np.abs(coefficient_c * leader - positions)
    return leader - coefficient_a * distance


def _update_positions(
    positions: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    delta: np.ndarray,
    a: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the original three-leader GWO position update."""

    x1 = _leader_guided_position(positions, alpha, a, rng)
    x2 = _leader_guided_position(positions, beta, a, rng)
    x3 = _leader_guided_position(positions, delta, a, rng)
    return (x1 + x2 + x3) / 3.0


def run_multi_agent_gwo(
    scenario: MultiAgentScenario,
    config: MultiAgentGWOConfig = MultiAgentGWOConfig(),
    *,
    weights: MultiAgentObjectiveWeights = DEFAULT_OBJECTIVE_WEIGHTS,
    seed: int = 42,
) -> MultiAgentGWOResult:
    """Optimize joint spatial paths and start delays with GWO."""

    rng = np.random.default_rng(seed)
    lower_bounds, upper_bounds = scenario.candidate_bounds()

    start_time = perf_counter()
    positions = _initialize_pack(scenario, config, rng)
    fitness = _evaluate_population(positions, scenario, weights)
    evaluations = config.num_wolves

    alpha_index, beta_index, delta_index = _leader_indices(fitness)
    leader_positions = np.stack(
        [
            positions[alpha_index].copy(),
            positions[beta_index].copy(),
            positions[delta_index].copy(),
        ],
        axis=0,
    )
    leader_fitness = np.asarray(
        [fitness[alpha_index], fitness[beta_index], fitness[delta_index]],
        dtype=float,
    )
    alpha_position, beta_position, delta_position = leader_positions

    best_position = alpha_position.copy()
    best_fitness = float(leader_fitness[0])
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
        positions = _reflect_bounds(positions, lower_bounds, upper_bounds)
        positions = _normalize_delay_genes(positions, scenario)

        fitness = _evaluate_population(positions, scenario, weights)
        evaluations += config.num_wolves

        leader_positions, leader_fitness = _update_leader_archive(
            leader_positions,
            leader_fitness,
            positions,
            fitness,
        )
        alpha_position, beta_position, delta_position = leader_positions
        best_position = alpha_position.copy()
        best_fitness = float(leader_fitness[0])

        fitness_history.append(best_fitness)
        best_vector_history.append(best_position.copy())
        population_history.append(positions.copy())
        leader_vector_history.append(
            np.stack([alpha_position, beta_position, delta_position], axis=0)
        )

    runtime = perf_counter() - start_time
    best_plan = decode_joint_vector(best_position, scenario)
    objective = evaluate_joint_vector(best_position, scenario, weights)
    metrics = compute_joint_plan_metrics(best_plan, scenario)

    return MultiAgentGWOResult(
        algorithm="Multi-Agent GWO",
        plan=best_plan,
        best_vector=best_position,
        best_fitness=best_fitness,
        fitness_history=np.asarray(fitness_history, dtype=float),
        best_vector_history=np.asarray(best_vector_history, dtype=float),
        population_history=np.asarray(population_history, dtype=np.float32),
        leader_vector_history=np.asarray(leader_vector_history, dtype=np.float32),
        iterations=config.max_iterations,
        evaluations=evaluations,
        runtime=float(runtime),
        seed=seed,
        objective=objective,
        metrics=metrics,
    )
