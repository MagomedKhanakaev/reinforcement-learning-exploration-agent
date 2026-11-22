import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.DQN_agent import DQN_agent
from experiments.dqn.DQN_training import train, evaluate, FRAME_DIM

POOL_SIZES = [1, 5, 10, 25, 50]
TRAINING_SEEDS = [4, 104, 204]
EPISODES = 2000
STACK_SIZE = 4

EVAL_SEEDS = tuple(range(101, 116))
SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10

RESULTS_DIR = Path("results") / "dqn" / "pool_size_experiment"
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


def already_done(results, pool_size, training_seed):
    for result in results:
        if result["pool_size"] == pool_size and result["training_seed"] == training_seed:
            return True
    return False


def run_one_config(pool_size, training_seed):
    checkpoint_dir = RESULTS_DIR / "checkpoints" / f"pool_{pool_size}_seed_{training_seed}"

    print(f"\n=== pool_size={pool_size} training_seed={training_seed} ===")
    train(
        episodes=EPISODES,
        target_update_freq=1000,
        stack_size=STACK_SIZE,
        pool_size=pool_size,
        seed=training_seed,
        checkpoint_dir=str(checkpoint_dir),
    )

    agent = DQN_agent(obs_dim=FRAME_DIM * STACK_SIZE, gamma=0.99, epsilon=0.0)
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
    )

    solved_count = 0
    for steps in eval_results["steps"]:
        if steps < SIZE * SIZE * 4:
            solved_count += 1

    result = {
        "pool_size": pool_size,
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

    for pool_size in POOL_SIZES:
        for training_seed in TRAINING_SEEDS:
            if already_done(results, pool_size, training_seed):
                print(f"Deja fait : pool_size={pool_size} training_seed={training_seed}, on saute")
                continue

            result = run_one_config(pool_size, training_seed)
            results.append(result)
            save_results(results)

    return results


def plot_results(results):
    solved_by_pool_size = {}
    for pool_size in POOL_SIZES:
        solved_by_pool_size[pool_size] = []

    for result in results:
        pool_size = result["pool_size"]
        solved_by_pool_size[pool_size].append(result["solved_count"])

    mean_solved = []
    std_solved = []
    for pool_size in POOL_SIZES:
        values = np.array(solved_by_pool_size[pool_size])
        mean_solved.append(values.mean())
        std_solved.append(values.std())

    plt.figure(figsize=(9, 6))
    plt.errorbar(
        POOL_SIZES,
        mean_solved,
        yerr=std_solved,
        marker="o",
        markersize=6,
        capsize=5,
        linewidth=2,
    )

    plt.title("DQN generalization vs training pool size")
    plt.xlabel("Number of maps in the training pool")
    plt.ylabel("Maps solved out of 15 (mean ± std over 3 seeds)")
    plt.xscale("log")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = RESULTS_DIR / "pool_size_vs_generalization.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nFigure enregistree : {output_path}")


def main():
    results = run_all_experiments()
    plot_results(results)


if __name__ == "__main__":
    main()
