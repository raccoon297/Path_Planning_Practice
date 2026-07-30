"""Common 3D path-planning environment for DQN and PPO.

The workspace, obstacles, observations, rewards, collision checks, and episode
termination rules are shared. Only the action representation and motion model
differ:

- DQN: six axis-aligned position increments.
- PPO: continuous 3D acceleration with velocity dynamics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.scenario import SCENARIO_CONFIG

Array = np.ndarray
ControlMode = Literal["dqn", "ppo"]

_EPS = 1e-9


@dataclass(frozen=True)
class SphereObstacle:
    """Spherical obstacle represented by a center and radius."""

    center: Array
    radius: float
    name: str = "sphere"

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError("Sphere center must have shape (3,).")
        if self.radius <= 0:
            raise ValueError("Sphere radius must be positive.")
        object.__setattr__(self, "center", center)

    def collides(self, point: Array, agent_radius: float = 0.0) -> bool:
        effective_radius = self.radius + agent_radius
        return bool(np.linalg.norm(np.asarray(point) - self.center) <= effective_radius)

    def clearance(self, point: Array, agent_radius: float = 0.0) -> float:
        return float(np.linalg.norm(np.asarray(point) - self.center) - self.radius - agent_radius)

    def ray_distance(
        self,
        origin: Array,
        direction: Array,
        max_distance: float,
        inflation: float = 0.0,
    ) -> float | None:
        """Return distance from origin to first ray-sphere intersection."""
        origin = np.asarray(origin, dtype=np.float64)
        direction = _unit_vector(direction)
        radius = self.radius + inflation

        offset = origin - self.center
        b = float(np.dot(offset, direction))
        c = float(np.dot(offset, offset) - radius * radius)
        discriminant = b * b - c
        if discriminant < 0:
            return None

        root = float(np.sqrt(max(discriminant, 0.0)))
        candidates = (-b - root, -b + root)
        for distance in candidates:
            if -_EPS <= distance <= max_distance + _EPS:
                return max(0.0, float(distance))
        return None


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned box obstacle represented by minimum and maximum corners."""

    minimum: Array
    maximum: Array
    name: str = "box"

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=np.float64)
        maximum = np.asarray(self.maximum, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("Box corners must have shape (3,).")
        if np.any(maximum <= minimum):
            raise ValueError("Every maximum coordinate must exceed the minimum.")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def collides(self, point: Array, agent_radius: float = 0.0) -> bool:
        point = np.asarray(point, dtype=np.float64)
        lower = self.minimum - agent_radius
        upper = self.maximum + agent_radius
        return bool(np.all(point >= lower) and np.all(point <= upper))

    def clearance(self, point: Array, agent_radius: float = 0.0) -> float:
        """Signed clearance from an agent sphere to the box surface."""
        point = np.asarray(point, dtype=np.float64)
        closest = np.clip(point, self.minimum, self.maximum)
        outside_distance = float(np.linalg.norm(point - closest))

        if outside_distance > _EPS:
            return outside_distance - agent_radius

        distances_to_faces = np.concatenate(
            (point - self.minimum, self.maximum - point)
        )
        penetration = float(np.min(distances_to_faces))
        return -penetration - agent_radius

    def ray_distance(
        self,
        origin: Array,
        direction: Array,
        max_distance: float,
        inflation: float = 0.0,
    ) -> float | None:
        """Return distance from origin to first ray-AABB intersection."""
        origin = np.asarray(origin, dtype=np.float64)
        direction = _unit_vector(direction)
        lower = self.minimum - inflation
        upper = self.maximum + inflation

        t_min = 0.0
        t_max = float(max_distance)
        for axis in range(3):
            if abs(direction[axis]) < _EPS:
                if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                    return None
                continue

            t1 = (lower[axis] - origin[axis]) / direction[axis]
            t2 = (upper[axis] - origin[axis]) / direction[axis]
            near, far = sorted((float(t1), float(t2)))
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_min > t_max:
                return None

        if 0.0 <= t_min <= max_distance + _EPS:
            return float(t_min)
        return None


Obstacle = SphereObstacle | BoxObstacle


def _unit_vector(vector: Array) -> Array:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= _EPS:
        raise ValueError("Direction vector must be non-zero.")
    return vector / norm


def create_ray_directions() -> Array:
    """Return the 26 normalized directions of a 3x3x3 neighborhood."""
    directions: list[Array] = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                if x == y == z == 0:
                    continue
                directions.append(_unit_vector(np.array([x, y, z], dtype=np.float64)))
    return np.asarray(directions, dtype=np.float64)


def create_benchmark_obstacles() -> list[Obstacle]:
    """Create a fixed urban benchmark made of ground-attached buildings.

    All building footprints start at ``z=0`` and their heights are at most
    60 simulation units. The layout leaves street-like corridors while the
    direct start-goal line intersects multiple buildings.
    """
    building_specs = [
        # Front row: a wider start corridor and moderate obstacle heights.
        ((16.0, 12.0), (29.0, 28.0), 20.0),
        ((38.0, 8.0), (51.0, 22.0), 45.0),
        ((63.0, 14.0), (77.0, 28.0), 30.0),
        # Middle row: the central building remains the main 3D detour.
        ((10.0, 43.0), (24.0, 57.0), 38.0),
        ((37.0, 37.0), (54.0, 54.0), 55.0),
        ((69.0, 40.0), (84.0, 53.0), 40.0),
        # Back row: wider corridors and a lower goal-side building.
        ((20.0, 70.0), (34.0, 84.0), 30.0),
        ((47.0, 66.0), (61.0, 80.0), 45.0),
        ((74.0, 72.0), (87.0, 87.0), 35.0),
    ]
    return [
        BoxObstacle(
            minimum=np.array([x0, y0, 0.0]),
            maximum=np.array([x1, y1, height]),
            name=f"building_{index}",
        )
        for index, ((x0, y0), (x1, y1), height) in enumerate(building_specs, start=1)
    ]


def load_config() -> dict[str, Any]:
    """Return an independent copy of the Python project configuration."""
    return deepcopy(SCENARIO_CONFIG)


class PathPlanning3DEnv(gym.Env[Array, int | Array]):
    """Shared 3D path-planning environment for discrete DQN and continuous PPO."""

    metadata = {"render_modes": []}

    DQN_DIRECTIONS = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        control_mode: ControlMode,
        obstacles: Sequence[Obstacle] | None = None,
    ) -> None:
        super().__init__()
        if control_mode not in ("dqn", "ppo"):
            raise ValueError("control_mode must be either 'dqn' or 'ppo'.")

        config = load_config()
        env_config = config["environment"]
        reward_config = env_config["reward"]

        self.control_mode = control_mode
        self.workspace_size = float(env_config["workspace_size"])
        self.ground_level = float(env_config.get("ground_level", 0.0))
        self.max_building_height = float(env_config.get("max_building_height", self.workspace_size))
        self.default_start = np.asarray(env_config["start"], dtype=np.float64)
        self.default_goal = np.asarray(env_config["goal"], dtype=np.float64)
        self.agent_radius = float(env_config["agent_radius"])
        self.safety_margin = float(env_config["safety_margin"])
        self.goal_radius = float(env_config["goal_radius"])
        self.max_steps = int(env_config["max_steps"])
        self.sensor_range = float(env_config["sensor_range"])

        self.reward_goal = float(reward_config["goal"])
        self.reward_collision = float(reward_config["collision"])
        self.progress_scale = float(reward_config["progress_scale"])
        self.reward_step = float(reward_config["step"])
        self.clearance_scale = float(reward_config["clearance_scale"])
        self.control_change_scale = float(reward_config["control_change_scale"])

        self.dqn_step_size = float(env_config["dqn"]["step_size"])
        self.dt = float(env_config["ppo"]["dt"])
        self.max_speed = float(env_config["ppo"]["max_speed"])
        self.max_acceleration = float(env_config["ppo"]["max_acceleration"])

        self._using_benchmark_obstacles = obstacles is None
        self.obstacles = list(obstacles) if obstacles is not None else create_benchmark_obstacles()
        self.ray_directions = create_ray_directions()
        expected_ray_count = int(env_config["ray_count"])
        if len(self.ray_directions) != expected_ray_count:
            raise ValueError(
                f"Configured ray_count={expected_ray_count}, but generated {len(self.ray_directions)} rays."
            )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(36,),
            dtype=np.float32,
        )
        if control_mode == "dqn":
            self.action_space = spaces.Discrete(6)
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32,
            )

        self.position = self.default_start.copy()
        self.goal = self.default_goal.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.previous_control = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.previous_goal_distance = self._goal_distance(self.position)
        self.trajectory: list[Array] = [self.position.copy()]
        self.controls: list[Array] = []
        self.outcome = "running"

        self._validate_static_setup()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Array, dict[str, Any]]:
        """Reset the environment and optionally override or randomize start/goal."""
        super().reset(seed=seed)
        options = options or {}

        start = np.asarray(options.get("start", self.default_start), dtype=np.float64)
        goal = np.asarray(options.get("goal", self.default_goal), dtype=np.float64)

        if options.get("randomize_start_goal", False):
            minimum_separation = float(options.get("minimum_separation", self.workspace_size * 0.55))
            start, goal = self._sample_valid_start_goal(minimum_separation)

        self._validate_point(start, "start")
        self._validate_point(goal, "goal")
        if np.linalg.norm(goal - start) <= self.goal_radius:
            raise ValueError("Start and goal must be separated by more than goal_radius.")

        self.position = start.copy()
        self.goal = goal.copy()
        self.velocity = np.zeros(3, dtype=np.float64)
        self.previous_control = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self.previous_goal_distance = self._goal_distance(self.position)
        self.trajectory = [self.position.copy()]
        self.controls = []
        self.outcome = "running"

        observation = self._get_observation()
        return observation, self._get_info()

    def step(self, action: int | Array) -> tuple[Array, float, bool, bool, dict[str, Any]]:
        if self.outcome != "running":
            raise RuntimeError("Episode has ended. Call reset() before step().")

        self.step_count += 1
        old_position = self.position.copy()

        if self.control_mode == "dqn":
            control, candidate_position, candidate_velocity = self._dqn_transition(action)
        else:
            control, candidate_position, candidate_velocity = self._ppo_transition(action)

        collision = self._segment_collides(old_position, candidate_position)
        out_of_bounds = not self._inside_workspace(candidate_position)

        terminated = False
        truncated = False
        reward = 0.0

        if out_of_bounds:
            self.position = candidate_position
            self.velocity = candidate_velocity
            self.outcome = "out_of_bounds"
            reward = self.reward_collision
            terminated = True
        elif collision:
            self.position = candidate_position
            self.velocity = candidate_velocity
            self.outcome = "collision"
            reward = self.reward_collision
            terminated = True
        else:
            self.position = candidate_position
            self.velocity = candidate_velocity
            current_distance = self._goal_distance(self.position)
            progress = self.previous_goal_distance - current_distance
            clearance = self.minimum_clearance(self.position)
            control_change = float(np.linalg.norm(control - self.previous_control) ** 2)

            reward = self.progress_scale * progress + self.reward_step
            clearance_threshold = self.safety_margin
            if clearance < clearance_threshold:
                reward -= self.clearance_scale * (clearance_threshold - clearance)
            reward -= self.control_change_scale * control_change

            if current_distance <= self.goal_radius:
                reward += self.reward_goal
                self.outcome = "success"
                terminated = True
            elif self.step_count >= self.max_steps:
                self.outcome = "timeout"
                truncated = True

            self.previous_goal_distance = current_distance

        self.previous_control = control.copy()
        self.trajectory.append(self.position.copy())
        self.controls.append(control.copy())

        observation = self._get_observation()
        return observation, float(reward), terminated, truncated, self._get_info()

    def _dqn_transition(self, action: int | Array) -> tuple[Array, Array, Array]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid DQN action: {action!r}")
        direction = self.DQN_DIRECTIONS[int(action)]
        displacement = direction * self.dqn_step_size
        candidate = self.position + displacement
        velocity_equivalent = displacement / max(self.dqn_step_size, _EPS)
        return direction.copy(), candidate, velocity_equivalent

    def _ppo_transition(self, action: int | Array) -> tuple[Array, Array, Array]:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (3,):
            raise ValueError("PPO action must have shape (3,).")
        control = np.clip(action_array, -1.0, 1.0)
        acceleration = control * self.max_acceleration
        velocity = self.velocity + acceleration * self.dt
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed:
            velocity = velocity * (self.max_speed / speed)
        candidate = self.position + velocity * self.dt
        return control, candidate, velocity

    def _get_observation(self) -> Array:
        workspace_diagonal = self.workspace_size * np.sqrt(3.0)
        goal_vector = (self.goal - self.position) / self.workspace_size
        goal_distance = np.array(
            [self._goal_distance(self.position) / workspace_diagonal], dtype=np.float64
        )

        if self.control_mode == "ppo":
            motion = self.velocity / max(self.max_speed, _EPS)
        else:
            motion = self.velocity.copy()

        rays = self.ray_distances(self.position) / self.sensor_range
        observation = np.concatenate(
            (goal_vector, goal_distance, motion, self.previous_control, rays)
        )
        observation = np.clip(observation, -1.0, 1.0).astype(np.float32)
        if observation.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation has unexpected shape {observation.shape}.")
        return observation

    def ray_distances(self, origin: Array) -> Array:
        """Measure obstacle and wall distance in all 26 sensor directions."""
        origin = np.asarray(origin, dtype=np.float64)
        distances = np.full(len(self.ray_directions), self.sensor_range, dtype=np.float64)

        for index, direction in enumerate(self.ray_directions):
            wall_distance = self._ray_distance_to_workspace(origin, direction)
            best = min(self.sensor_range, wall_distance)
            for obstacle in self.obstacles:
                distance = obstacle.ray_distance(
                    origin,
                    direction,
                    max_distance=best,
                    inflation=self.agent_radius,
                )
                if distance is not None:
                    best = min(best, distance)
            distances[index] = max(0.0, min(self.sensor_range, best))
        return distances

    def minimum_clearance(self, point: Array) -> float:
        """Return minimum surface-to-surface clearance to obstacles or walls."""
        point = np.asarray(point, dtype=np.float64)
        wall_clearance = float(
            min(
                np.min(point - self.agent_radius),
                np.min(self.workspace_size - self.agent_radius - point),
            )
        )
        obstacle_clearance = min(
            (obstacle.clearance(point, self.agent_radius) for obstacle in self.obstacles),
            default=float("inf"),
        )
        return float(min(wall_clearance, obstacle_clearance))

    def _segment_collides(self, start: Array, end: Array) -> bool:
        displacement = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(displacement))
        if length <= _EPS:
            return any(obstacle.collides(start, self.agent_radius) for obstacle in self.obstacles)
        direction = displacement / length
        return any(
            obstacle.ray_distance(
                start,
                direction,
                max_distance=length,
                inflation=self.agent_radius,
            )
            is not None
            for obstacle in self.obstacles
        )

    def _ray_distance_to_workspace(self, origin: Array, direction: Array) -> float:
        lower = np.full(3, self.agent_radius, dtype=np.float64)
        upper = np.full(3, self.workspace_size - self.agent_radius, dtype=np.float64)
        distances: list[float] = []
        for axis in range(3):
            if direction[axis] > _EPS:
                distances.append(float((upper[axis] - origin[axis]) / direction[axis]))
            elif direction[axis] < -_EPS:
                distances.append(float((lower[axis] - origin[axis]) / direction[axis]))
        positive = [distance for distance in distances if distance >= 0.0]
        return min(positive) if positive else self.sensor_range

    def _inside_workspace(self, point: Array) -> bool:
        point = np.asarray(point, dtype=np.float64)
        lower = self.agent_radius
        upper = self.workspace_size - self.agent_radius
        return bool(np.all(point >= lower) and np.all(point <= upper))

    def _goal_distance(self, point: Array) -> float:
        return float(np.linalg.norm(self.goal - np.asarray(point, dtype=np.float64)))

    def _get_info(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "step_count": self.step_count,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "distance_to_goal": self._goal_distance(self.position),
            "minimum_clearance": self.minimum_clearance(self.position),
            "speed": float(np.linalg.norm(self.velocity)),
        }

    def _sample_valid_start_goal(self, minimum_separation: float) -> tuple[Array, Array]:
        low = self.agent_radius + 0.5
        high = self.workspace_size - self.agent_radius - 0.5
        for _ in range(10_000):
            start = self.np_random.uniform(low, high, size=3)
            goal = self.np_random.uniform(low, high, size=3)
            if np.linalg.norm(goal - start) < minimum_separation:
                continue
            if self._point_collides(start) or self._point_collides(goal):
                continue
            return start.astype(np.float64), goal.astype(np.float64)
        raise RuntimeError("Could not sample a valid start-goal pair.")

    def _point_collides(self, point: Array) -> bool:
        return any(obstacle.collides(point, self.agent_radius) for obstacle in self.obstacles)

    def _validate_point(self, point: Array, label: str) -> None:
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"{label} must have shape (3,).")
        if not self._inside_workspace(point):
            raise ValueError(f"{label} is outside the valid workspace.")
        if self._point_collides(point):
            raise ValueError(f"{label} overlaps an obstacle.")

    def _validate_static_setup(self) -> None:
        if self.workspace_size <= 2 * self.agent_radius:
            raise ValueError("Workspace is too small for the configured agent radius.")
        if self.sensor_range <= 0 or self.max_steps <= 0:
            raise ValueError("sensor_range and max_steps must be positive.")
        self._validate_point(self.default_start, "default start")
        self._validate_point(self.default_goal, "default goal")
        for obstacle in self.obstacles:
            if isinstance(obstacle, SphereObstacle):
                minimum = obstacle.center - obstacle.radius
                maximum = obstacle.center + obstacle.radius
            else:
                minimum = obstacle.minimum
                maximum = obstacle.maximum
            if np.any(minimum < 0.0) or np.any(maximum > self.workspace_size):
                raise ValueError(f"Obstacle {obstacle.name!r} lies outside the workspace.")

        if self._using_benchmark_obstacles:
            for obstacle in self.obstacles:
                if not isinstance(obstacle, BoxObstacle):
                    raise ValueError("The urban benchmark must contain only box buildings.")
                if not np.isclose(obstacle.minimum[2], self.ground_level):
                    raise ValueError(f"Building {obstacle.name!r} must touch the ground plane.")
                height = obstacle.maximum[2] - obstacle.minimum[2]
                if height > self.max_building_height + _EPS:
                    raise ValueError(
                        f"Building {obstacle.name!r} exceeds max_building_height."
                    )


def rollout_random_policy(
    env: PathPlanning3DEnv,
    seed: int = 0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run one deterministic random-policy smoke test and return its summary."""
    observation, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    limit = max_steps or env.max_steps

    for _ in range(limit):
        if env.control_mode == "dqn":
            action: int | Array = int(rng.integers(0, 6))
        else:
            action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    return {
        "observation": observation,
        "total_reward": float(np.sum(rewards)),
        "trajectory": np.asarray(env.trajectory, dtype=np.float64),
        "info": info,
    }


__all__ = [
    "BoxObstacle",
    "PathPlanning3DEnv",
    "SphereObstacle",
    "create_benchmark_obstacles",
    "create_ray_directions",
    "load_config",
    "rollout_random_policy",
]
