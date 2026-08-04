from pathlib import Path

from experiments.q_learning_training import (
    train,
    evaluate,
    save_experiment,
)
from results.plots import plot_training_results


BASE_CONFIG = {
    "episodes": 5000,
    "size": 15,
    "obstacle_density": 0.12,
    "trap_density": 0.04,
    "mud_density": 0.10,
    "alpha": 0.10,
    "gamma": 0.95,
    "epsilon": 1.0,
    "seed": 4,
    "moving_average_window": 100,
}


def run_experiment(
    experiment_name,
    parameter_name,
    config,
):
    experiment_dir = (
        Path("results")
        / "hyperparameter_analysis"
        / parameter_name
        / experiment_name
    )

    print(f"\nRunning experiment: {experiment_name}")
    print(f"Alpha: {config['alpha']}")
    print(f"Gamma: {config['gamma']}")

    agent, env, metrics = train(
        episodes=config["episodes"],
        size=config["size"],
        obstacle_density=config["obstacle_density"],
        trap_density=config["trap_density"],
        mud_density=config["mud_density"],
        alpha=config["alpha"],
        gamma=config["gamma"],
        epsilon=config["epsilon"],
        seed=config["seed"],
    )

    plot_training_results(
        episode_rewards=metrics["rewards"],
        episode_steps=metrics["steps"],
        episode_epsilons=metrics["epsilons"],
        episode_collisions=metrics["collisions"],
        episode_traps=metrics["traps"],
        save_dir=experiment_dir,
        window=config["moving_average_window"],
    )

    evaluation_results = evaluate(
        agent=agent,
        env=env,
    )

    save_experiment(
        experiment_dir=experiment_dir,
        agent=agent,
        env=env,
        config=config,
        evaluation_results=evaluation_results,
        metrics=metrics,
    )


def run_alpha_experiments():
    alpha_values = [0.05, 0.10, 0.30, 0.50]

    for alpha in alpha_values:
        config = BASE_CONFIG.copy()
        config["alpha"] = alpha
        config["gamma"] = 0.95

        experiment_name = f"alpha_{alpha:.2f}"

        run_experiment(
            experiment_name=experiment_name,
            parameter_name="alpha",
            config=config,
        )


def run_gamma_experiments():
    gamma_values = [0.80, 0.90, 0.95, 0.99]

    for gamma in gamma_values:
        config = BASE_CONFIG.copy()
        config["alpha"] = 0.10
        config["gamma"] = gamma

        experiment_name = f"gamma_{gamma:.2f}"

        run_experiment(
            experiment_name=experiment_name,
            parameter_name="gamma",
            config=config,
        )


def main():
    run_alpha_experiments()
    run_gamma_experiments()


if __name__ == "__main__":
    main()