import json
import pickle
from pathlib import Path

import numpy as np

from agents.q_learning_agent import QLearningAgent
from environment.advanced_environment import AdvancedEnvironment
from results.plots import plot_training_results


def train(
    episodes=5000,
    size=15,
    obstacle_density=0.12,
    trap_density=0.04,
    mud_density=0.10,
    alpha=0.1,
    gamma=0.95,
    epsilon=1.0,
    seed=4,
):
    env = AdvancedEnvironment(
        size=size,
        obstacle_density=obstacle_density,
        trap_density=trap_density,
        mud_density=mud_density,
        seed=seed,
    )

    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
    )

    metrics = {
        "rewards": [],
        "steps": [],
        "epsilons": [],
        "collisions": [],
        "traps": [],
    }

    for _ in range(episodes):
        state = env.reset_episode()
        done = False

        total_reward = 0
        collision_count = 0
        trap_count = 0

        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            agent.update_q_table(
                state,
                action,
                reward,
                next_state,
                done,
            )

            if info["collision"]:
                collision_count += 1

            if info["trap"]:
                trap_count += 1

            state = next_state
            total_reward += reward

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(env.count_steps)
        metrics["epsilons"].append(agent.epsilon)
        metrics["collisions"].append(collision_count)
        metrics["traps"].append(trap_count)

        agent.decay_epsilon()

    return agent, env, metrics


def evaluate(agent, env):
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset_episode()
    done = False

    total_reward = 0
    collision_count = 0
    trap_count = 0

    while not done:
        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)

        if info["collision"]:
            collision_count += 1

        if info["trap"]:
            trap_count += 1

        state = next_state
        total_reward += reward

    agent.epsilon = previous_epsilon

    results = {
        "reward": total_reward,
        "steps": env.count_steps,
        "collisions": collision_count,
        "traps": trap_count,
    }

    print("\nEvaluation")
    print("----------")
    print(f"Reward: {results['reward']}")
    print(f"Steps: {results['steps']}")
    print(f"Collisions: {results['collisions']}")
    print(f"Traps: {results['traps']}")

    return results


def save_experiment(
    experiment_dir,
    agent,
    env,
    config,
    evaluation_results,
    metrics,
):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(
        experiment_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=4)

    with open(
        experiment_dir / "evaluation.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(evaluation_results, file, indent=4)

    with open(
        experiment_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(experiment_dir / "q_table.pkl", "wb") as file:
        pickle.dump(agent.q_table, file)

    np.save(
        experiment_dir / "grid.npy",
        env.grid,
    )

    print(f"\nExperiment saved in: {experiment_dir}")


def main():
    experiment_name = "advanced"

    experiment_dir = (
        Path("results")
        / "environment_comparison"
        / experiment_name
    )

    config = {
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


if __name__ == "__main__":
    main()