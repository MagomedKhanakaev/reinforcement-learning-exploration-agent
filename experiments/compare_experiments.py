import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_DIR = Path("results") / "environment_comparison"

OUTPUT_DIR = EXPERIMENTS_DIR / "comparisons"


def moving_average(
    values: Sequence[float],
    window: int = 100,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)

    if array.size == 0:
        raise ValueError("Cannot average empty data.")

    if array.size < window:
        return array.copy()

    weights = np.ones(window, dtype=float) / window
    return np.convolve(array, weights, mode="valid")


def load_metrics(experiment_name: str) -> dict:
    metrics_path = EXPERIMENTS_DIR / experiment_name / "metrics.json"

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_metric(
    *,
    experiment_names: Sequence[str],
    labels: Sequence[str],
    metric_name: str,
    title: str,
    ylabel: str,
    filename: str,
    window: int = 100,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
) -> None:
    if len(experiment_names) != len(labels):
        raise ValueError("Experiment names and labels must have the same length.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(11, 6))

    for experiment_name, label in zip(
        experiment_names,
        labels,
    ):
        metrics = load_metrics(experiment_name)
        values = metrics[metric_name]

        average = moving_average(
            values,
            window=window,
        )

        if len(values) >= window:
            episodes = np.arange(
                window,
                len(values) + 1,
            )
        else:
            episodes = np.arange(
                1,
                len(values) + 1,
            )

        plt.plot(
            episodes,
            average,
            linewidth=2.2,
            label=label,
        )

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    if x_limits is not None:
        plt.xlim(*x_limits)

    if y_limits is not None:
        plt.ylim(*y_limits)

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved comparison: {output_path}")


def main() -> None:
    experiment_names = [
        "simple",
        "advanced",
    ]

    labels = [
        "Simple",
        "Advanced",
    ]

    compare_metric(
        experiment_names=experiment_names,
        labels=labels,
        metric_name="rewards",
        title="Reward comparison",
        ylabel="Moving-average reward",
        filename="reward_full.png",
    )

    compare_metric(
        experiment_names=experiment_names,
        labels=labels,
        metric_name="rewards",
        title="Reward comparison during early training",
        ylabel="Moving-average reward",
        filename="reward_zoom.png",
        x_limits=(300, 1600),
        y_limits=(-1200, 100),
    )

    compare_metric(
        experiment_names=experiment_names,
        labels=labels,
        metric_name="steps",
        title="Episode length comparison",
        ylabel="Moving-average steps",
        filename="steps_full.png",
    )

    compare_metric(
        experiment_names=experiment_names,
        labels=labels,
        metric_name="steps",
        title="Episode length during early training",
        ylabel="Moving-average steps",
        filename="steps_zoom.png",
        x_limits=(300, 1600),
        y_limits=(0, 300),
    )

    compare_metric(
        experiment_names=experiment_names,
        labels=labels,
        metric_name="collisions",
        title="Collision comparison",
        ylabel="Moving-average collisions",
        filename="collisions_full.png",
    )

    compare_metric(
        experiment_names=experiment_names,
        labels=labels,
        metric_name="collisions",
        title="Collision comparison during early training",
        ylabel="Moving-average collisions",
        filename="collisions_zoom.png",
        x_limits=(300, 1600),
        y_limits=(0, 80),
    )

    compare_metric(
        experiment_names=experiment_names,
        labels=labels,
        metric_name="traps",
        title="Triggered traps in the advanced environment",
        ylabel="Moving-average traps",
        filename="traps_advanced.png",
    )


if __name__ == "__main__":
    main()
