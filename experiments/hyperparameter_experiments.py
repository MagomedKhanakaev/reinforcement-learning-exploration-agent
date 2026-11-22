import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.q_learning_training import train
from results.plots import moving_average

MAP_SEEDS = [4, 10, 20, 30]
ALPHA_VALUES = [0.05, 0.10, 0.30, 0.50]
GAMMA_VALUES = [0.80, 0.90, 0.95, 0.99]
EPISODES = 5000
WINDOW = 100

SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10


def run_one_training(map_seed, alpha, gamma):
    agent, env, metrics = train(
        episodes=EPISODES,
        size=SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        trap_density=TRAP_DENSITY,
        mud_density=MUD_DENSITY,
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        seed=map_seed,
    )
    return metrics


def collect_curves(parameter_name, parameter_values, fixed_alpha, fixed_gamma):
    curves_by_value = {}

    for value in parameter_values:
        reward_curves = []
        step_curves = []

        for map_seed in MAP_SEEDS:
            if parameter_name == "alpha":
                alpha = value
                gamma = fixed_gamma
            else:
                alpha = fixed_alpha
                gamma = value

            print(f"Entrainement : carte seed={map_seed} alpha={alpha} gamma={gamma}")
            metrics = run_one_training(map_seed, alpha, gamma)

            smoothed_reward = moving_average(metrics["rewards"], WINDOW)
            smoothed_steps = moving_average(metrics["steps"], WINDOW)

            reward_curves.append(smoothed_reward)
            step_curves.append(smoothed_steps)

        reward_curves = np.array(reward_curves)
        step_curves = np.array(step_curves)

        curves_by_value[value] = {
            "mean_reward": reward_curves.mean(axis=0).tolist(),
            "std_reward": reward_curves.std(axis=0).tolist(),
            "mean_steps": step_curves.mean(axis=0).tolist(),
            "std_steps": step_curves.std(axis=0).tolist(),
        }

    return curves_by_value


def plot_curves(curves_by_value, parameter_name, metric_name, ylabel, save_dir):
    plt.figure(figsize=(11, 6))

    for value in curves_by_value:
        mean_values = np.array(curves_by_value[value][f"mean_{metric_name}"])
        std_values = np.array(curves_by_value[value][f"std_{metric_name}"])
        episodes = np.arange(WINDOW, WINDOW + len(mean_values))

        plt.plot(episodes, mean_values, linewidth=2, label=f"{parameter_name}={value}")
        plt.fill_between(
            episodes,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
        )

    plt.title(f"Q-learning {metric_name} across {len(MAP_SEEDS)} maps, varying {parameter_name}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"{parameter_name}_{metric_name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Figure enregistree : {output_path}")


def run_and_save(parameter_name, parameter_values, fixed_alpha, fixed_gamma):
    save_dir = Path("results") / "hyperparameter_analysis" / parameter_name

    curves_by_value = collect_curves(
        parameter_name=parameter_name,
        parameter_values=parameter_values,
        fixed_alpha=fixed_alpha,
        fixed_gamma=fixed_gamma,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    data_path = save_dir / "curves.json"
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(curves_by_value, file, indent=4)
    print(f"Donnees brutes enregistrees : {data_path}")

    plot_curves(curves_by_value, parameter_name, "reward", "Total reward", save_dir)
    plot_curves(curves_by_value, parameter_name, "steps", "Steps", save_dir)


def main():
    run_and_save(
        parameter_name="alpha",
        parameter_values=ALPHA_VALUES,
        fixed_alpha=None,
        fixed_gamma=0.95,
    )

    run_and_save(
        parameter_name="gamma",
        parameter_values=GAMMA_VALUES,
        fixed_alpha=0.10,
        fixed_gamma=None,
    )


if __name__ == "__main__":
    main()
