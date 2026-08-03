import json
import pickle
from pathlib import Path

import numpy as np

from environment.advanced_environment import AdvancedEnvironment
from agents.q_learning_agent import QLearningAgent
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
):
    env = AdvancedEnvironment(
        size=size,
        obstacle_density=obstacle_density,
        trap_density=trap_density,
        mud_density=mud_density,
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

    for episode in range(episodes):
        state = env.reset_episode()
        done = False

        total_reward = 0
        collisions = 0
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
                collisions += 1

            if info["trap"]:
                trap_count += 1

            state = next_state
            total_reward += reward

        metrics["rewards"].append(total_reward)
        metrics["steps"].append(env.count_steps)
        metrics["epsilons"].append(agent.epsilon)
        metrics["collisions"].append(collisions)
        metrics["traps"].append(trap_count)

        agent.decay_epsilon()

    return agent, env, metrics


def evaluate(agent, env):
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset_episode()
    done = False

    total_reward = 0
    collisions = 0
    trap_count = 0

    while not done:
        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)

        if info["collision"]:
            collisions += 1

        if info["trap"]:
            trap_count += 1

        state = next_state
        total_reward += reward

    agent.epsilon = previous_epsilon

    results = {
        "reward": total_reward,
        "steps": env.count_steps,
        "collisions": collisions,
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
    experiment_name,
    agent,
    env,
    config,
    evaluation_results,
):
    experiment_dir = (
        Path("results")
        / "experiments"
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_path = experiment_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            indent=4,
        )

    evaluation_path = experiment_dir / "evaluation.json"

    with open(evaluation_path, "w", encoding="utf-8") as file:
        json.dump(
            evaluation_results,
            file,
            indent=4,
        )

    q_table_path = experiment_dir / "q_table.pkl"

    with open(q_table_path, "wb") as file:
        pickle.dump(
            agent.q_table,
            file,
        )

    grid_path = experiment_dir / "grid.npy"

    np.save(
        grid_path,
        env.grid,
    )

    print(f"\nExperiment saved in: {experiment_dir}")


def main():
    experiment_name = "medium"
    experiment_dir = (
        Path("results")
        / "experiments"
        / experiment_name
    )

    config = {
        "episodes": 5000,
        "size": 15,
        "obstacle_density": 0.12,
        "trap_density": 0.04,
        "mud_density": 0.10,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 1.0,
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
        experiment_name=experiment_name,
        agent=agent,
        env=env,
        config=config,
        evaluation_results=evaluation_results,
    )


if __name__ == "__main__":
    main()