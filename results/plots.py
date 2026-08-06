from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def moving_average(
    values: Sequence[float],
    window: int = 100,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        raise ValueError("Cannot compute a moving average on empty data.")

    if array.size < window:
        return array.copy()

    weights = np.ones(window, dtype=float) / window
    return np.convolve(array, weights, mode="valid")


def _plot_metric(
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    filename: str,
    save_dir: Path,
    window: int = 100,
    raw_label: str = "Episode values",
    average_label: str | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:

    values_array = np.asarray(values, dtype=float)
    average = moving_average(values_array, window)

    save_dir.mkdir(parents=True, exist_ok=True)

    raw_episodes = np.arange(1, len(values_array) + 1)

    if len(values_array) >= window:
        average_episodes = np.arange(window, len(values_array) + 1)
    else:
        average_episodes = raw_episodes

    if average_label is None:
        average_label = f"Moving average ({window} episodes)"

    plt.figure(figsize=(11, 6))

    plt.plot(
        raw_episodes,
        values_array,
        linewidth=0.6,
        alpha=0.12,
        label=raw_label,
    )

    plt.plot(
        average_episodes,
        average,
        linewidth=2.5,
        label=average_label,
    )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    if y_limits is not None:
        plt.ylim(*y_limits)

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = save_dir / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_rewards(
    episode_rewards: Sequence[float],
    save_dir: Path,
    window: int = 100,
):
    _plot_metric(
        values=episode_rewards,
        title="Q-learning reward during training",
        ylabel="Total reward",
        filename="q_learning_rewards.png",
        save_dir=save_dir,
        window=window,
    )


def plot_steps(
    episode_steps: Sequence[int],
    save_dir: Path,
    window: int = 100,
):
    _plot_metric(
        values=episode_steps,
        title="Number of steps per episode",
        ylabel="Steps",
        filename="q_learning_steps.png",
        save_dir=save_dir,
        window=window,
    )


def plot_epsilons(
    episode_epsilons: Sequence[float],
    save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))

    plt.plot(
        np.arange(1, len(episode_epsilons) + 1),
        episode_epsilons,
        linewidth=2.5,
        color="tab:green",
    )

    plt.title("Exploration rate during training")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")

    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = save_dir / "q_learning_epsilons.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_traps(
    episode_traps: Sequence[int],
    save_dir: Path,
    window: int = 100,
):
    _plot_metric(
        values=episode_traps,
        title="Traps triggered per episode",
        ylabel="Triggered traps",
        filename="q_learning_traps.png",
        save_dir=save_dir,
        window=window,
    )


def plot_collisions(
    episode_collisions: Sequence[int],
    save_dir: Path,
    window: int = 100,
):
    _plot_metric(
        values=episode_collisions,
        title="Number of collisions per episode",
        ylabel="Collisions",
        filename="q_learning_collisions.png",
        save_dir=save_dir,
        window=window,
    )


def plot_training_results(
    episode_rewards: Sequence[float],
    episode_steps: Sequence[int],
    episode_epsilons: Sequence[float],
    episode_collisions: Sequence[int],
    episode_traps: Sequence[int],
    save_dir: Path,
    window: int = 100,
):

    plot_rewards(
        episode_rewards,
        save_dir=save_dir,
        window=window,
    )

    plot_steps(
        episode_steps,
        save_dir=save_dir,
        window=window,
    )

    plot_epsilons(
        episode_epsilons,
        save_dir=save_dir,
    )

    plot_collisions(
        episode_collisions,
        save_dir=save_dir,
        window=window,
    )

    plot_traps(
        episode_traps,
        save_dir=save_dir,
        window=window,
    )
