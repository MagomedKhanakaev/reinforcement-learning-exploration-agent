import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.DQN_agent import DQN_agent
from experiments.DQN_training import train, evaluate

WINDOW_SIZES = [3, 5, 7]
TRAINING_SEEDS = [4, 104, 204]
EPISODES = 2000
STACK_SIZE = 4
POOL_SIZE = 50

EVAL_SEEDS = tuple(range(101, 116))
SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10

RESULTS_DIR = Path("results") / "window_size_experiment"
RESULTS_PATH = RESULTS_DIR / "results.json"


def load_existing_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return []


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def already_done(results, window_size, training_seed):
    for result in results:
        if result["window_size"] == window_size and result["training_seed"] == training_seed:
            return True
    return False


def run_one_config(window_size, training_seed):
    checkpoint_dir = RESULTS_DIR / "checkpoints" / f"window_{window_size}_seed_{training_seed}"

    print(f"\n=== window_size={window_size} training_seed={training_seed} ===")
    train(
        episodes=EPISODES,
        target_update_freq=1000,
        stack_size=STACK_SIZE,
        pool_size=POOL_SIZE,
        seed=training_seed,
        window_size=window_size,
        checkpoint_dir=str(checkpoint_dir),
    )

    frame_dim = 4 * window_size * window_size + 2
    agent = DQN_agent(obs_dim=frame_dim * STACK_SIZE, gamma=0.99, epsilon=0.0)
    best_checkpoint_path = checkpoint_dir / "policy_net_best.pth"
    agent.policy_net.load_state_dict(torch.load(best_checkpoint_path, weights_only=True))
    agent.policy_net.eval()

    eval_results = evaluate(
        agent,
        eval_seeds=EVAL_SEEDS,
        size=SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        trap_density=TRAP_DENSITY,
        mud_density=MUD_DENSITY,
        stack_size=STACK_SIZE,
        window_size=window_size,
    )

    solved_count = 0
    for steps in eval_results["steps"]:
        if steps < SIZE * SIZE * 4:
            solved_count += 1

    result = {
        "window_size": window_size,
        "training_seed": training_seed,
        "solved_count": solved_count,
        "total_maps": len(EVAL_SEEDS),
        "mean_reward": eval_results["mean_reward"],
        "mean_steps": eval_results["mean_steps"],
    }

    print(
        f"Resultat : {solved_count}/{len(EVAL_SEEDS)} resolues, "
        f"recompense moyenne={eval_results['mean_reward']:.1f}, "
        f"pas moyens={eval_results['mean_steps']:.1f}"
    )

    return result


def run_all_experiments():
    results = load_existing_results()

    for window_size in WINDOW_SIZES:
        for training_seed in TRAINING_SEEDS:
            if already_done(results, window_size, training_seed):
                print(f"Deja fait : window_size={window_size} training_seed={training_seed}, on saute")
                continue

            result = run_one_config(window_size, training_seed)
            results.append(result)
            save_results(results)

    return results


def plot_results(results):
    solved_by_window_size = {}
    reward_by_window_size = {}
    for window_size in WINDOW_SIZES:
        solved_by_window_size[window_size] = []
        reward_by_window_size[window_size] = []

    for result in results:
        window_size = result["window_size"]
        solved_by_window_size[window_size].append(result["solved_count"])
        reward_by_window_size[window_size].append(result["mean_reward"])

    mean_solved = []
    std_solved = []
    mean_reward = []
    std_reward = []
    for window_size in WINDOW_SIZES:
        solved_values = np.array(solved_by_window_size[window_size])
        reward_values = np.array(reward_by_window_size[window_size])
        mean_solved.append(solved_values.mean())
        std_solved.append(solved_values.std())
        mean_reward.append(reward_values.mean())
        std_reward.append(reward_values.std())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].errorbar(
        WINDOW_SIZES,
        mean_solved,
        yerr=std_solved,
        marker="o",
        markersize=6,
        capsize=5,
        linewidth=2,
    )
    axes[0].set_title("Solve rate vs local observation window size")
    axes[0].set_xlabel("Window size")
    axes[0].set_ylabel("Maps solved out of 15 (mean ± std over 3 seeds)")
    axes[0].set_xticks(WINDOW_SIZES)
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        WINDOW_SIZES,
        mean_reward,
        yerr=std_reward,
        marker="o",
        markersize=6,
        capsize=5,
        linewidth=2,
        color="tab:orange",
    )
    axes[1].set_title("Evaluation reward vs local observation window size")
    axes[1].set_xlabel("Window size")
    axes[1].set_ylabel("Mean evaluation reward (mean ± std over 3 seeds)")
    axes[1].set_xticks(WINDOW_SIZES)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()

    output_path = RESULTS_DIR / "window_size_vs_generalization.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nFigure enregistree : {output_path}")


def main():
    results = run_all_experiments()
    plot_results(results)


if __name__ == "__main__":
    main()
