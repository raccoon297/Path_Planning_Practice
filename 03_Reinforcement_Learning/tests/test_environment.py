from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from core.environment import (
    BoxObstacle,
    PathPlanning3DEnv,
    SphereObstacle,
    create_ray_directions,
    load_config,
    rollout_random_policy,
)


def test_ray_directions_are_26_unique_unit_vectors() -> None:
    directions = create_ray_directions()
    assert directions.shape == (26, 3)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert len(np.unique(np.round(directions, decimals=8), axis=0)) == 26


def test_observation_shape_and_space() -> None:
    for mode in ("dqn", "ppo"):
        env = PathPlanning3DEnv(mode)
        observation, _ = env.reset(seed=7)
        assert observation.shape == (36,)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)


def test_same_seed_reproduces_random_start_and_goal() -> None:
    env = PathPlanning3DEnv("dqn")
    options = {"randomize_start_goal": True, "minimum_separation": 40.0}

    env.reset(seed=123, options=options)
    first_start = env.position.copy()
    first_goal = env.goal.copy()

    env.reset(seed=123, options=options)
    assert np.allclose(env.position, first_start)
    assert np.allclose(env.goal, first_goal)


def test_dqn_six_actions_move_on_exact_axes() -> None:
    env = PathPlanning3DEnv("dqn", obstacles=[])
    expected = PathPlanning3DEnv.DQN_DIRECTIONS * env.dqn_step_size

    for action in range(6):
        env.reset(seed=0, options={"start": [50.0, 50.0, 70.0], "goal": [94.0, 94.0, 8.0]})
        start = env.position.copy()
        _, _, terminated, truncated, _ = env.step(action)
        assert not terminated
        assert not truncated
        assert np.allclose(env.position - start, expected[action])


def test_ppo_acceleration_and_speed_limits() -> None:
    env = PathPlanning3DEnv("ppo", obstacles=[])
    env.reset(seed=0, options={"start": [10.0, 10.0, 70.0], "goal": [94.0, 94.0, 8.0]})

    speeds = []
    for _ in range(8):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        assert not terminated
        assert not truncated
        speeds.append(info["speed"])

    assert np.all(np.diff(speeds[:4]) > 0.0)
    assert max(speeds) <= env.max_speed + 1e-9
    assert np.isclose(speeds[-1], env.max_speed)


def test_sphere_and_box_collision_geometry() -> None:
    sphere = SphereObstacle(center=np.array([5.0, 5.0, 5.0]), radius=2.0)
    assert sphere.collides(np.array([7.4, 5.0, 5.0]), agent_radius=0.5)
    assert not sphere.collides(np.array([7.6, 5.0, 5.0]), agent_radius=0.5)

    box = BoxObstacle(
        minimum=np.array([4.0, 4.0, 4.0]),
        maximum=np.array([6.0, 6.0, 6.0]),
    )
    assert box.collides(np.array([3.6, 5.0, 5.0]), agent_radius=0.5)
    assert not box.collides(np.array([3.4, 5.0, 5.0]), agent_radius=0.5)


def test_segment_collision_prevents_tunneling() -> None:
    obstacle = SphereObstacle(center=np.array([7.2, 6.0, 8.0]), radius=0.05)
    env = PathPlanning3DEnv("dqn", obstacles=[obstacle])
    env.reset(seed=0)

    _, reward, terminated, truncated, info = env.step(0)
    assert terminated
    assert not truncated
    assert reward == env.reward_collision
    assert info["outcome"] == "collision"


def test_goal_collision_boundary_and_timeout_outcomes() -> None:
    goal_env = PathPlanning3DEnv("dqn", obstacles=[])
    goal_env.reset(seed=0, options={"goal": [10.0, 6.0, 8.0]})
    _, _, terminated, truncated, info = goal_env.step(0)
    assert terminated and not truncated
    assert info["outcome"] == "success"

    boundary_env = PathPlanning3DEnv("dqn", obstacles=[])
    boundary_env.reset(
        seed=0,
        options={"start": [98.5, 50.0, 50.0], "goal": [6.0, 6.0, 8.0]},
    )
    _, _, terminated, truncated, info = boundary_env.step(0)
    assert terminated and not truncated
    assert info["outcome"] == "out_of_bounds"

    timeout_env = PathPlanning3DEnv("dqn", obstacles=[])
    timeout_env.max_steps = 1
    timeout_env.reset(seed=0, options={"start": [50.0, 50.0, 70.0], "goal": [94.0, 94.0, 8.0]})
    _, _, terminated, truncated, info = timeout_env.step(0)
    assert not terminated and truncated
    assert info["outcome"] == "timeout"


def test_invalid_actions_are_rejected() -> None:
    dqn_env = PathPlanning3DEnv("dqn", obstacles=[])
    dqn_env.reset()
    with pytest.raises(ValueError):
        dqn_env.step(6)

    ppo_env = PathPlanning3DEnv("ppo", obstacles=[])
    ppo_env.reset()
    with pytest.raises(ValueError):
        ppo_env.step(np.array([0.0, 0.0]))


def test_random_rollout_is_reproducible() -> None:
    first = rollout_random_policy(PathPlanning3DEnv("ppo", obstacles=[]), seed=9, max_steps=12)
    second = rollout_random_policy(PathPlanning3DEnv("ppo", obstacles=[]), seed=9, max_steps=12)
    assert np.allclose(first["trajectory"], second["trajectory"])
    assert first["total_reward"] == pytest.approx(second["total_reward"])



def test_urban_benchmark_uses_grounded_buildings() -> None:
    env = PathPlanning3DEnv("dqn")
    assert env.workspace_size == pytest.approx(100.0)
    assert env.max_building_height == pytest.approx(60.0)
    assert len(env.obstacles) == 9

    for obstacle in env.obstacles:
        assert isinstance(obstacle, BoxObstacle)
        assert obstacle.minimum[2] == pytest.approx(env.ground_level)
        assert obstacle.maximum[2] <= env.max_building_height + 1e-9


def test_direct_start_goal_segment_is_blocked_by_city() -> None:
    env = PathPlanning3DEnv("dqn")
    env.reset(seed=0)
    assert env._segment_collides(env.position, env.goal)

def test_benchmark_map_has_a_valid_six_direction_path() -> None:
    env = PathPlanning3DEnv("dqn")
    env.reset(seed=0)

    start = tuple(np.rint(env.position).astype(int))
    queue = deque([start])
    visited = {start}
    found = False

    while queue:
        current = np.asarray(queue.popleft(), dtype=np.float64)
        if np.linalg.norm(current - env.goal) <= env.goal_radius:
            found = True
            break

        for direction in env.DQN_DIRECTIONS:
            candidate = current + direction * env.dqn_step_size
            key = tuple(np.rint(candidate).astype(int))
            if key in visited:
                continue
            if not env._inside_workspace(candidate):
                continue
            if env._segment_collides(current, candidate):
                continue
            visited.add(key)
            queue.append(key)

    assert found, "The fixed benchmark map must be solvable by the six DQN actions."


def test_python_config_returns_independent_copies() -> None:
    first = load_config()
    second = load_config()

    first["environment"]["start"][0] = -999.0
    assert second["environment"]["start"][0] == pytest.approx(6.0)
