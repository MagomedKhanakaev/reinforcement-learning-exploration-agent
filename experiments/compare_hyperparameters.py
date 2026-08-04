import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = (
    Path("results")
    / "hyperparameter_analysis"
)

ALPHA_DIR = RESULTS_DIR / "alpha"
GAMMA_DIR = RESULTS_DIR / "gamma"


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


def load_metrics(
    base_dir: Path,
    experiment_name: str,
) -> dict:
    metrics_path = (
        base_dir
        / experiment_name
        / "metrics.json"
    )

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def compare_experiments(
    *,
    base_dir: Path,
    output_dir: Path,
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
        raise ValueError(
            "Experiment names and labels must have the same length."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(11, 6))

    for experiment_name, label in zip(
        experiment_names,
        labels,
    ):
        metrics = load_metrics(
            base_dir=base_dir,
            experiment_name=experiment_name,
        )

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

    output_path = output_dir / filename

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved comparison: {output_path}")


def compare_alpha() -> None:
    experiment_names = [
        "alpha_0.05",
        "alpha_0.10",
        "alpha_0.30",
        "alpha_0.50",
    ]

    labels = [
        "Alpha = 0.05",
        "Alpha = 0.10",
        "Alpha = 0.30",
        "Alpha = 0.50",
    ]

    output_dir = ALPHA_DIR / "comparisons"

    compare_experiments(
        base_dir=ALPHA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="rewards",
        title="Influence of alpha on Q-learning",
        ylabel="Moving-average reward",
        filename="reward_full.png",
    )

    compare_experiments(
        base_dir=ALPHA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="rewards",
        title="Influence of alpha during early training",
        ylabel="Moving-average reward",
        filename="reward_zoom.png",
        x_limits=(300, 1600),
        y_limits=(-1200, 100),
    )

    compare_experiments(
        base_dir=ALPHA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="steps",
        title="Influence of alpha on episode length",
        ylabel="Moving-average steps",
        filename="steps_full.png",
    )

    compare_experiments(
        base_dir=ALPHA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="steps",
        title="Influence of alpha during early training",
        ylabel="Moving-average steps",
        filename="steps_zoom.png",
        x_limits=(300, 1600),
        y_limits=(0, 300),
    )


def compare_gamma() -> None:
    experiment_names = [
        "gamma_0.80",
        "gamma_0.90",
        "gamma_0.95",
        "gamma_0.99",
    ]

    labels = [
        "Gamma = 0.80",
        "Gamma = 0.90",
        "Gamma = 0.95",
        "Gamma = 0.99",
    ]

    output_dir = GAMMA_DIR / "comparisons"

    compare_experiments(
        base_dir=GAMMA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="rewards",
        title="Influence of gamma on Q-learning",
        ylabel="Moving-average reward",
        filename="reward_full.png",
    )

    compare_experiments(
        base_dir=GAMMA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="rewards",
        title="Influence of gamma during early training",
        ylabel="Moving-average reward",
        filename="reward_zoom.png",
        x_limits=(300, 1600),
        y_limits=(-1200, 100),
    )

    compare_experiments(
        base_dir=GAMMA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="steps",
        title="Influence of gamma on episode length",
        ylabel="Moving-average steps",
        filename="steps_full.png",
    )

    compare_experiments(
        base_dir=GAMMA_DIR,
        output_dir=output_dir,
        experiment_names=experiment_names,
        labels=labels,
        metric_name="steps",
        title="Influence of gamma during early training",
        ylabel="Moving-average steps",
        filename="steps_zoom.png",
        x_limits=(300, 1600),
        y_limits=(0, 300),
    )


def main() -> None:
    compare_alpha()
    compare_gamma()


if __name__ == "__main__":
    main()