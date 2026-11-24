import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.DQN_agent import DQN_agent
from experiments.dqn.DQN_training import evaluate
from experiments.generalization_comparison import evaluate_q_learning

# The 15 frozen maps were used to pick the best checkpoint of every run, so the
# score on them is optimistic. The new maps were never used for anything.
FROZEN_SEEDS = tuple(range(101, 116))
NEW_SEEDS = tuple(range(1001, 1201))

TRAINING_SEEDS = [4, 104, 204]
POOL_SIZES = [1, 5, 10, 25, 50]
WINDOW_SIZES = [3, 5, 7]
STACK_SIZES = [1, 2, 4]

SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10

DQN_DIR = Path("results") / "dqn"
RESULTS_DIR = DQN_DIR / "unbiased_evaluation"
RESULTS_PATH = RESULTS_DIR / "results.json"
Q_TABLE_PATH = "results/q_learning/environment_comparison/advanced/q_table.pkl"


def build_configs():
    configs = []

    for pool_size in POOL_SIZES:
        for training_seed in TRAINING_SEEDS:
            folder = f"pool_{pool_size}_seed_{training_seed}"
            configs.append({
                "experiment": "pool_size",
                "setting": pool_size,
                "training_seed": training_seed,
                "stack_size": 4,
                "window_size": 5,
                "checkpoint": str(DQN_DIR / "pool_size_experiment" / "checkpoints" / folder / "policy_net_best.pth"),
            })

    for window_size in WINDOW_SIZES:
        for training_seed in TRAINING_SEEDS:
            folder = f"window_{window_size}_seed_{training_seed}"
            configs.append({
                "experiment": "window_size",
                "setting": window_size,
                "training_seed": training_seed,
                "stack_size": 4,
                "window_size": window_size,
                "checkpoint": str(DQN_DIR / "window_size_experiment" / "checkpoints" / folder / "policy_net_best.pth"),
            })

    for stack_size in STACK_SIZES:
        for training_seed in TRAINING_SEEDS:
            folder = f"stack_{stack_size}_seed_{training_seed}"
            configs.append({
                "experiment": "stack_size",
                "setting": stack_size,
                "training_seed": training_seed,
                "stack_size": stack_size,
                "window_size": 5,
                "checkpoint": str(DQN_DIR / "stack_size_experiment" / "checkpoints" / folder / "policy_net_best.pth"),
            })

    configs.append({
        "experiment": "reference",
        "setting": "best checkpoint",
        "training_seed": 4,
        "stack_size": 4,
        "window_size": 5,
        "checkpoint": str(DQN_DIR / "dqn_checkpoints" / "policy_net_best.pth"),
    })

    return configs


def load_existing_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"dqn": [], "q_learning": None}


def save_results(results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)


def already_done(results, config):
    for result in results["dqn"]:
        same_experiment = result["experiment"] == config["experiment"]
        same_setting = result["setting"] == config["setting"]
        same_seed = result["training_seed"] == config["training_seed"]
        if same_experiment and same_setting and same_seed:
            return True
    return False


def count_solved(steps_list):
    solved_count = 0
    for steps in steps_list:
        if steps < SIZE * SIZE * 4:
            solved_count += 1
    return solved_count


def evaluate_one_checkpoint(config):
    frame_dim = 4 * config["window_size"] * config["window_size"] + 2
    agent = DQN_agent(obs_dim=frame_dim * config["stack_size"], gamma=0.99, epsilon=0.0)
    agent.policy_net.load_state_dict(torch.load(config["checkpoint"], weights_only=True))
    agent.policy_net.eval()

    frozen = evaluate(
        agent,
        eval_seeds=FROZEN_SEEDS,
        size=SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        trap_density=TRAP_DENSITY,
        mud_density=MUD_DENSITY,
        stack_size=config["stack_size"],
        window_size=config["window_size"],
    )
    new = evaluate(
        agent,
        eval_seeds=NEW_SEEDS,
        size=SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        trap_density=TRAP_DENSITY,
        mud_density=MUD_DENSITY,
        stack_size=config["stack_size"],
        window_size=config["window_size"],
    )

    result = {
        "experiment": config["experiment"],
        "setting": config["setting"],
        "training_seed": config["training_seed"],
        "frozen_solved": count_solved(frozen["steps"]),
        "frozen_total": len(FROZEN_SEEDS),
        "new_solved": count_solved(new["steps"]),
        "new_total": len(NEW_SEEDS),
        "new_mean_reward": new["mean_reward"],
        "new_mean_steps": new["mean_steps"],
    }
    return result


def evaluate_q_learning_on_new_maps():
    with open(Q_TABLE_PATH, "rb") as file:
        q_table = pickle.load(file)

    per_seed_results, new_state_ratio = evaluate_q_learning(q_table, eval_seeds=NEW_SEEDS)

    solved_count = 0
    total_reward = 0
    total_steps = 0
    for result in per_seed_results:
        if result["resolved"]:
            solved_count += 1
        total_reward += result["reward"]
        total_steps += result["steps"]

    count = len(per_seed_results)
    return {
        "new_solved": solved_count,
        "new_total": count,
        "new_mean_reward": total_reward / count,
        "new_mean_steps": total_steps / count,
        "unseen_state_ratio": new_state_ratio,
    }


def run_all_evaluations():
    results = load_existing_results()
    configs = build_configs()

    for index, config in enumerate(configs):
        if already_done(results, config):
            continue

        print(f"[{index + 1}/{len(configs)}] {config['experiment']} {config['setting']} seed={config['training_seed']}")
        result = evaluate_one_checkpoint(config)
        results["dqn"].append(result)
        save_results(results)

        print(
            f"  gelees {result['frozen_solved']}/{result['frozen_total']}, "
            f"neuves {result['new_solved']}/{result['new_total']}, "
            f"recompense neuves={result['new_mean_reward']:.1f}"
        )

    if results["q_learning"] is None:
        print("Q-learning sur les cartes neuves")
        results["q_learning"] = evaluate_q_learning_on_new_maps()
        save_results(results)
        print(
            f"  neuves {results['q_learning']['new_solved']}/{results['q_learning']['new_total']}, "
            f"etats inconnus={results['q_learning']['unseen_state_ratio'] * 100:.1f}%"
        )

    return results


def check_against_previous_runs(results):
    """The frozen-map score recomputed here must match the one saved by the
    original experiments, otherwise the checkpoints or the protocol changed."""
    previous_files = {
        "pool_size": DQN_DIR / "pool_size_experiment" / "results.json",
        "window_size": DQN_DIR / "window_size_experiment" / "results.json",
        "stack_size": DQN_DIR / "stack_size_experiment" / "results.json",
    }

    mismatches = 0
    for experiment in previous_files:
        with open(previous_files[experiment], "r", encoding="utf-8") as file:
            previous_results = json.load(file)

        for previous in previous_results:
            for result in results["dqn"]:
                if result["experiment"] != experiment:
                    continue
                if result["setting"] != previous[experiment]:
                    continue
                if result["training_seed"] != previous["training_seed"]:
                    continue
                if result["frozen_solved"] != previous["solved_count"]:
                    mismatches += 1
                    print(
                        f"Ecart : {experiment} {result['setting']} seed={result['training_seed']} "
                        f"avant={previous['solved_count']} maintenant={result['frozen_solved']}"
                    )

    print(f"Ecarts avec les experiences d'origine sur les cartes gelees : {mismatches}")


def summarize_experiment(results, experiment, settings):
    summary = {}

    for setting in settings:
        frozen_percent = []
        new_percent = []
        new_rewards = []

        for result in results["dqn"]:
            if result["experiment"] != experiment or result["setting"] != setting:
                continue
            frozen_percent.append(100 * result["frozen_solved"] / result["frozen_total"])
            new_percent.append(100 * result["new_solved"] / result["new_total"])
            new_rewards.append(result["new_mean_reward"])

        summary[str(setting)] = {
            "runs": len(new_percent),
            "frozen_percent_mean": float(np.mean(frozen_percent)),
            "frozen_percent_std": float(np.std(frozen_percent)),
            "new_percent_mean": float(np.mean(new_percent)),
            "new_percent_std": float(np.std(new_percent)),
            "new_reward_mean": float(np.mean(new_rewards)),
            "new_reward_std": float(np.std(new_rewards)),
        }

    return summary


def draw_two_curves(ax, settings, summary, xlabel, log_scale=False):
    frozen_mean = []
    frozen_std = []
    new_mean = []
    new_std = []
    for setting in settings:
        values = summary[str(setting)]
        frozen_mean.append(values["frozen_percent_mean"])
        frozen_std.append(values["frozen_percent_std"])
        new_mean.append(values["new_percent_mean"])
        new_std.append(values["new_percent_std"])

    ax.errorbar(
        settings, frozen_mean, yerr=frozen_std,
        marker="o", markersize=6, capsize=5, linewidth=2,
        linestyle="--", color="tab:gray",
        label="15 frozen maps (used to pick the checkpoint)",
    )
    ax.errorbar(
        settings, new_mean, yerr=new_std,
        marker="o", markersize=6, capsize=5, linewidth=2,
        color="tab:blue",
        label="200 unseen maps",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Maps solved (%), mean ± std over 3 seeds")
    ax.set_xticks(settings)
    ax.set_xticklabels([str(setting) for setting in settings])
    if log_scale:
        ax.set_xscale("log")
        ax.set_xticks(settings)
        ax.set_xticklabels([str(setting) for setting in settings])
        ax.minorticks_off()
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)


def plot_results(results):
    pool_summary = summarize_experiment(results, "pool_size", POOL_SIZES)
    window_summary = summarize_experiment(results, "window_size", WINDOW_SIZES)
    stack_summary = summarize_experiment(results, "stack_size", STACK_SIZES)

    fig, ax = plt.subplots(figsize=(9, 6))
    draw_two_curves(ax, POOL_SIZES, pool_summary, "Number of maps in the training pool", log_scale=True)
    ax.set_title("DQN generalization vs training pool size")
    ax.legend(loc="lower right")
    plt.tight_layout()
    pool_path = RESULTS_DIR / "pool_size_unbiased.png"
    plt.savefig(pool_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure enregistree : {pool_path}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    draw_two_curves(axes[0], WINDOW_SIZES, window_summary, "Local observation window size")
    axes[0].set_title("Window size")
    axes[0].legend(loc="lower right", fontsize=8)
    draw_two_curves(axes[1], STACK_SIZES, stack_summary, "Number of stacked frames")
    axes[1].set_title("Stacked frames")
    plt.tight_layout()
    ablation_path = RESULTS_DIR / "ablations_unbiased.png"
    plt.savefig(ablation_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Figure enregistree : {ablation_path}")

    summary = {
        "pool_size": pool_summary,
        "window_size": window_summary,
        "stack_size": stack_summary,
        "q_learning": results["q_learning"],
    }
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
    print(f"Resume enregistre : {summary_path}")


def main():
    results = run_all_evaluations()
    check_against_previous_runs(results)
    plot_results(results)


if __name__ == "__main__":
    main()
