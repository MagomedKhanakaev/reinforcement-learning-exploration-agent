import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.DQN_agent import DQN_agent
from experiments.dqn.DQN_training import train, evaluate

STACK_SIZES = [1, 2, 4]
TRAINING_SEEDS = [4, 104, 204]
EPISODES = 2000
WINDOW_SIZE = 5
POOL_SIZE = 50

EVAL_SEEDS = tuple(range(101, 116))
SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10

RESULTS_DIR = Path("results") / "dqn" / "stack_size_experiment"
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


def already_done(results, stack_size, training_seed):
    for result in results:
        if result["stack_size"] == stack_size and result["training_seed"] == training_seed:
            return True
    return False


def run_one_config(stack_size, training_seed):
    checkpoint_dir = RESULTS_DIR / "checkpoints" / f"stack_{stack_size}_seed_{training_seed}"

    print(f"\n=== stack_size={stack_size} training_seed={training_seed} ===")
    train(
        episodes=EPISODES,
        target_update_freq=1000,
        stack_size=stack_size,
        pool_size=POOL_SIZE,
        seed=training_seed,
        window_size=WINDOW_SIZE,
        checkpoint_dir=str(checkpoint_dir),
    )

    frame_dim = 4 * WINDOW_SIZE * WINDOW_SIZE + 2
    agent = DQN_agent(obs_dim=frame_dim * stack_size, gamma=0.99, epsilon=0.0)
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
        stack_size=stack_size,
        window_size=WINDOW_SIZE,
    )

    solved_count = 0
    for steps in eval_results["steps"]:
        if steps < SIZE * SIZE * 4:
            solved_count += 1

    result = {
        "stack_size": stack_size,
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

    for stack_size in STACK_SIZES:
        for training_seed in TRAINING_SEEDS:
            if already_done(results, stack_size, training_seed):
                print(f"Deja fait : stack_size={stack_size} training_seed={training_seed}, on saute")
                continue

            result = run_one_config(stack_size, training_seed)
            results.append(result)
            save_results(results)

    return results


def plot_results(results):
    solved_by_stack_size = {}
    reward_by_stack_size = {}
    for stack_size in STACK_SIZES:
        solved_by_stack_size[stack_size] = []
        reward_by_stack_size[stack_size] = []

    for result in results:
        stack_size = result["stack_size"]
        solved_by_stack_size[stack_size].append(result["solved_count"])
        reward_by_stack_size[stack_size].append(result["mean_reward"])

    mean_solved = []
    std_solved = []
    mean_reward = []
    std_reward = []
    for stack_size in STACK_SIZES:
        solved_values = np.array(solved_by_stack_size[stack_size])
        reward_values = np.array(reward_by_stack_size[stack_size])
        mean_solved.append(solved_values.mean())
        std_solved.append(solved_values.std())
        mean_reward.append(reward_values.mean())
        std_reward.append(reward_values.std())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].errorbar(
        STACK_SIZES,
        mean_solved,
        yerr=std_solved,
        marker="o",
        markersize=6,
        capsize=5,
        linewidth=2,
    )
    axes[0].set_title("Solve rate vs number of stacked frames")
    axes[0].set_xlabel("Stack size")
    axes[0].set_ylabel("Maps solved out of 15 (mean ± std over 3 seeds)")
    axes[0].set_xticks(STACK_SIZES)
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        STACK_SIZES,
        mean_reward,
        yerr=std_reward,
        marker="o",
        markersize=6,
        capsize=5,
        linewidth=2,
        color="tab:orange",
    )
    axes[1].set_title("Evaluation reward vs number of stacked frames")
    axes[1].set_xlabel("Stack size")
    axes[1].set_ylabel("Mean evaluation reward (mean ± std over 3 seeds)")
    axes[1].set_xticks(STACK_SIZES)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()

    output_path = RESULTS_DIR / "stack_size_vs_generalization.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nFigure enregistree : {output_path}")


def main():
    results = run_all_experiments()
    plot_results(results)


if __name__ == "__main__":
    main()
