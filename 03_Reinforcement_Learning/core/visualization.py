"""Visualization utilities for the shared 3D path-planning environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from core.environment import BoxObstacle, PathPlanning3DEnv, SphereObstacle

Array = np.ndarray
DISPLAY_Z_MAX = 80.0


def _draw_sphere(ax, obstacle: SphereObstacle, alpha: float = 0.25) -> None:
    """Draw a spherical obstacle."""
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 18)

    x = obstacle.center[0] + obstacle.radius * np.outer(np.cos(u), np.sin(v))
    y = obstacle.center[1] + obstacle.radius * np.outer(np.sin(u), np.sin(v))
    z = obstacle.center[2] + obstacle.radius * np.outer(
        np.ones_like(u), np.cos(v)
    )

    ax.plot_surface(
        x,
        y,
        z,
        alpha=alpha,
        linewidth=0.0,
    )


def _box_faces(
    obstacle: BoxObstacle,
) -> list[list[tuple[float, float, float]]]:
    """Return the six faces of an axis-aligned box obstacle."""
    x0, y0, z0 = obstacle.minimum
    x1, y1, z1 = obstacle.maximum

    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]

    return [
        [vertices[index] for index in (0, 1, 2, 3)],
        [vertices[index] for index in (4, 5, 6, 7)],
        [vertices[index] for index in (0, 1, 5, 4)],
        [vertices[index] for index in (2, 3, 7, 6)],
        [vertices[index] for index in (1, 2, 6, 5)],
        [vertices[index] for index in (0, 3, 7, 4)],
    ]


def _draw_box(
    ax,
    obstacle: BoxObstacle,
    alpha: float = 0.22,
) -> None:
    """Draw a box obstacle using one consistent Matplotlib color."""
    collection = Poly3DCollection(
        _box_faces(obstacle),
        alpha=alpha,
    )
    collection.set_edgecolor("black")
    collection.set_linewidth(0.5)
    ax.add_collection3d(collection)


def configure_axes(
    ax,
    env: PathPlanning3DEnv,
    title: str | None = None,
) -> None:
    """Apply the shared fixed camera and axis configuration."""
    ax.set_xlim(0.0, env.workspace_size)
    ax.set_ylim(0.0, env.workspace_size)
    ax.set_zlim(0.0, DISPLAY_Z_MAX)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Preserve the original perspective view while reflecting the 100:100:80
    # displayed coordinate range.
    ax.set_box_aspect(
        (
            env.workspace_size,
            env.workspace_size,
            DISPLAY_Z_MAX,
        )
    )
    ax.view_init(elev=24, azim=-58)

    if title:
        ax.set_title(title)


def draw_environment(ax, env: PathPlanning3DEnv) -> None:
    """Draw the ground, obstacles, start point, and goal point."""
    ground = np.array(
        [
            [0.0, 0.0, env.ground_level],
            [env.workspace_size, 0.0, env.ground_level],
            [env.workspace_size, env.workspace_size, env.ground_level],
            [0.0, env.workspace_size, env.ground_level],
        ]
    )

    ground_collection = Poly3DCollection([ground], alpha=0.08)
    ground_collection.set_edgecolor("gray")
    ground_collection.set_linewidth(0.6)
    ax.add_collection3d(ground_collection)

    for obstacle in env.obstacles:
        if isinstance(obstacle, SphereObstacle):
            _draw_sphere(ax, obstacle)
        elif isinstance(obstacle, BoxObstacle):
            _draw_box(ax, obstacle)

    ax.scatter(
        *env.default_start,
        marker="o",
        s=55,
        label="Start",
    )
    ax.scatter(
        *env.default_goal,
        marker="*",
        s=110,
        label="Goal",
    )


def plot_environment(
    env: PathPlanning3DEnv,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot the fixed urban environment without an agent trajectory."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    draw_environment(ax, env)
    configure_axes(
        ax,
        env,
        "100 x 100 x 100 Urban Path-Planning Benchmark",
    )
    ax.legend(loc="upper left")

    fig.tight_layout()
    _finish_figure(fig, save_path, show)
    return fig


def plot_trajectory(
    env: PathPlanning3DEnv,
    trajectory: Sequence[Sequence[float]],
    label: str,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot one trajectory in the shared urban environment."""
    path = np.asarray(trajectory, dtype=np.float64)
    _validate_trajectory(path)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    draw_environment(ax, env)
    ax.plot(
        path[:, 0],
        path[:, 1],
        path[:, 2],
        linewidth=2.0,
        label=label,
    )
    configure_axes(ax, env, f"{label} Trajectory")
    ax.legend(loc="upper left")

    fig.tight_layout()
    _finish_figure(fig, save_path, show)
    return fig


def plot_trajectory_comparison(
    env: PathPlanning3DEnv,
    trajectories: Mapping[str, Sequence[Sequence[float]]],
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot DQN and PPO trajectories in one shared environment."""
    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    draw_environment(ax, env)

    for label, trajectory in trajectories.items():
        path = np.asarray(trajectory, dtype=np.float64)
        _validate_trajectory(path)
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            linewidth=2.0,
            label=label,
        )

    configure_axes(ax, env, "DQN and PPO Trajectory Comparison")
    ax.legend(loc="upper left")

    fig.tight_layout()
    _finish_figure(fig, save_path, show)
    return fig


def animate_trajectory(
    env: PathPlanning3DEnv,
    trajectory: Sequence[Sequence[float]],
    label: str,
    save_path: str | Path | None = None,
    interval_ms: int = 60,
    show: bool = False,
):
    """Animate one trajectory using the same fixed camera as the PNG output."""
    path = np.asarray(trajectory, dtype=np.float64)
    _validate_trajectory(path)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    draw_environment(ax, env)
    configure_axes(ax, env, f"{label} Navigation")

    (line,) = ax.plot([], [], [], linewidth=2.0, label=label)
    (marker,) = ax.plot(
        [],
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=6,
    )
    ax.legend(loc="upper left")

    def update(frame: int):
        current = path[: frame + 1]
        line.set_data(current[:, 0], current[:, 1])
        line.set_3d_properties(current[:, 2])

        marker.set_data([current[-1, 0]], [current[-1, 1]])
        marker.set_3d_properties([current[-1, 2]])
        return line, marker

    animation = FuncAnimation(
        fig,
        update,
        frames=len(path),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(
            output,
            writer=PillowWriter(fps=max(1, 1000 // interval_ms)),
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


def animate_trajectory_comparison(
    env: PathPlanning3DEnv,
    trajectories: Mapping[str, Sequence[Sequence[float]]],
    save_path: str | Path | None = None,
    interval_ms: int = 60,
    show: bool = False,
):
    """Animate multiple trajectories in the same fixed 3D environment."""
    if not trajectories:
        raise ValueError("At least one trajectory is required.")

    paths: dict[str, Array] = {}
    for label, trajectory in trajectories.items():
        path = np.asarray(trajectory, dtype=np.float64)
        _validate_trajectory(path)
        paths[label] = path

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    draw_environment(ax, env)
    configure_axes(ax, env, "DQN and PPO Navigation Comparison")

    artists: dict[str, tuple[Any, Any]] = {}
    for label in paths:
        (line,) = ax.plot([], [], [], linewidth=2.0, label=label)
        (marker,) = ax.plot(
            [],
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=6,
        )
        artists[label] = (line, marker)

    ax.legend(loc="upper left")
    total_frames = max(len(path) for path in paths.values())

    def update(frame: int):
        updated: list[Any] = []

        for label, path in paths.items():
            end = min(frame + 1, len(path))
            current = path[:end]
            line, marker = artists[label]

            line.set_data(current[:, 0], current[:, 1])
            line.set_3d_properties(current[:, 2])

            marker.set_data([current[-1, 0]], [current[-1, 1]])
            marker.set_3d_properties([current[-1, 2]])
            updated.extend((line, marker))

        return tuple(updated)

    animation = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        animation.save(
            output,
            writer=PillowWriter(fps=max(1, 1000 // interval_ms)),
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


def _validate_trajectory(path: Array) -> None:
    if path.ndim != 2 or path.shape[1] != 3 or len(path) == 0:
        raise ValueError("Trajectory must have shape (N, 3) with N >= 1.")


def _finish_figure(
    fig,
    save_path: str | Path | None,
    show: bool,
) -> None:
    if save_path is not None:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_history(
    episode_rewards: Sequence[float],
    episode_losses: Sequence[float],
    save_path: str | Path | None = None,
    show: bool = False,
    moving_average_window: int = 50,
    evaluation_returns: Sequence[float] | None = None,
):
    """Plot DQN episode returns and update losses."""
    rewards = np.asarray(episode_rewards, dtype=np.float64)
    losses = np.asarray(episode_losses, dtype=np.float64)

    if rewards.ndim != 1 or losses.ndim != 1 or len(rewards) != len(losses):
        raise ValueError(
            "episode_rewards and episode_losses must be equal-length 1D data."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        rewards,
        alpha=0.35,
        label="Episode reward",
    )

    window = min(max(1, moving_average_window), len(rewards))
    if len(rewards) >= window:
        kernel = np.ones(window, dtype=np.float64) / window
        average = np.convolve(rewards, kernel, mode="valid")
        axes[0].plot(
            np.arange(window - 1, len(rewards)),
            average,
            linewidth=2.0,
            label=f"Moving average ({window})",
        )

    if evaluation_returns is not None:
        evaluation = np.asarray(evaluation_returns, dtype=np.float64)
        if evaluation.shape != rewards.shape:
            raise ValueError("evaluation_returns must match episode_rewards length.")

        indices = np.flatnonzero(np.isfinite(evaluation))
        if len(indices) > 0:
            axes[0].scatter(
                indices,
                evaluation[indices],
                marker="x",
                s=34,
                label="Greedy evaluation",
            )

    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    valid_loss = np.where(np.isfinite(losses), losses, np.nan)
    axes[1].plot(valid_loss, linewidth=1.2)
    axes[1].set_title("Mean Update Loss")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Huber loss")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    _finish_figure(fig, save_path, show)
    return fig


def plot_ppo_training_history(
    history: Sequence[Mapping[str, Any]],
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Plot PPO returns, losses, entropy, and approximate KL."""
    if not history:
        raise ValueError("PPO training history is empty.")

    steps = np.asarray(
        [row["global_step"] for row in history],
        dtype=np.float64,
    )
    recent_returns = np.asarray(
        [row["recent_mean_return_20"] for row in history],
        dtype=np.float64,
    )
    evaluation_returns = np.asarray(
        [row["evaluation_return"] for row in history],
        dtype=np.float64,
    )
    policy_losses = np.asarray(
        [row["policy_loss"] for row in history],
        dtype=np.float64,
    )
    value_losses = np.asarray(
        [row["value_loss"] for row in history],
        dtype=np.float64,
    )
    entropies = np.asarray(
        [row["entropy"] for row in history],
        dtype=np.float64,
    )
    approximate_kls = np.asarray(
        [row["approximate_kl"] for row in history],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(
        steps,
        recent_returns,
        linewidth=1.8,
        label="Mean return (20 episodes)",
    )

    evaluation_indices = np.flatnonzero(np.isfinite(evaluation_returns))
    if len(evaluation_indices) > 0:
        axes[0].scatter(
            steps[evaluation_indices],
            evaluation_returns[evaluation_indices],
            marker="x",
            s=34,
            label="Deterministic evaluation",
        )

    axes[0].set_title("PPO Training Return")
    axes[0].set_xlabel("Environment steps")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        steps,
        policy_losses,
        linewidth=1.2,
        label="Policy loss",
    )
    axes[1].plot(
        steps,
        value_losses,
        linewidth=1.2,
        label="Value loss",
    )
    axes[1].set_title("PPO Update Losses")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    axes[2].plot(
        steps,
        entropies,
        linewidth=1.2,
        label="Gaussian entropy",
    )
    axes[2].plot(
        steps,
        approximate_kls,
        linewidth=1.2,
        label="Approximate KL",
    )
    axes[2].set_title("Policy Diagnostics")
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Value")
    axes[2].grid(alpha=0.25)
    axes[2].legend()

    fig.tight_layout()
    _finish_figure(fig, save_path, show)
    return fig


__all__ = [
    "animate_trajectory",
    "animate_trajectory_comparison",
    "configure_axes",
    "draw_environment",
    "plot_environment",
    "plot_ppo_training_history",
    "plot_trajectory",
    "plot_trajectory_comparison",
    "plot_training_history",
]
