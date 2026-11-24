import json
import random
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patheffects

from agents.DQN_agent import DQN_agent
from environment.advanced_environment import AdvancedEnvironment
from experiments.dqn.DQN_training import FRAME_DIM
from utils.frame_stack import FrameStack
from utils.render import draw_agent, draw_map, trail_points

CHECKPOINT_PATH = "results/dqn/dqn_checkpoints/policy_net_best.pth"
OUTPUT_DIR = Path("results") / "dqn" / "failure_analysis"
RESULTS_PATH = OUTPUT_DIR / "results.json"

STACK_SIZE = 4
SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10

FROZEN_SEEDS = tuple(range(101, 116))
NEW_SEEDS = tuple(range(1001, 1201))


def load_agent():
    agent = DQN_agent(obs_dim=FRAME_DIM * STACK_SIZE, gamma=0.99, epsilon=0.0)
    agent.policy_net.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    agent.policy_net.eval()
    return agent


def shortest_path_length(env):
    # Same blocking rules as path_exists: obstacles and traps block, mud does not.
    queue = deque([(env.start, 0)])
    visited = {env.start}

    while queue:
        position, distance = queue.popleft()
        if position == env.goal:
            return distance

        for dx, dy in env.actions.values():
            next_position = (position[0] + dx, position[1] + dy)
            row, col = next_position

            if not (0 <= row < env.size and 0 <= col < env.size):
                continue
            if env.grid[row][col] == 1 or env.grid[row][col] == 2:
                continue
            if next_position in visited:
                continue

            visited.add(next_position)
            queue.append((next_position, distance + 1))

    return None


def run_episode(agent, seed, memory_window=None):
    if memory_window is None:
        memory_window = 2 * SIZE

    env = AdvancedEnvironment(
        size=SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        trap_density=TRAP_DENSITY,
        mud_density=MUD_DENSITY,
        seed=seed,
    )

    manhattan = abs(env.start[0] - env.goal[0]) + abs(env.start[1] - env.goal[1])
    shortest = shortest_path_length(env)

    frame_stack = FrameStack(STACK_SIZE)
    state = frame_stack.reset(env.local_vision())

    recent_positions = deque(maxlen=memory_window)
    recent_positions.append(env.current_position)
    stuck_streak = 0

    positions = [env.current_position]
    trap_hits = 0
    trap_hits_by_rank = [0, 0, 0, 0]
    trap_hits_random = 0
    collisions = 0
    random_fallback_steps = 0
    guardrail_steps = 0
    total_reward = 0
    done = False

    while not done:
        ranked = agent.ranked_actions(state)
        chosen_rank = None
        if stuck_streak >= len(ranked):
            action = random.randrange(agent.n_actions)
            random_fallback_steps += 1
        else:
            action = ranked[stuck_streak]
            chosen_rank = stuck_streak
            if stuck_streak > 0:
                guardrail_steps += 1

        next_frame, reward, done = env.step_DQN(agent.actions[action])
        state = frame_stack.push(next_frame)
        total_reward += reward

        if reward == -25:
            trap_hits += 1
            # Rank 0 is the network's favourite action, rank 3 is the one it likes the least.
            if chosen_rank is None:
                trap_hits_random += 1
            else:
                trap_hits_by_rank[chosen_rank] += 1
        if reward == -6:
            collisions += 1

        if env.current_position in recent_positions:
            stuck_streak += 1
        else:
            stuck_streak = 0
        recent_positions.append(env.current_position)
        positions.append(env.current_position)

    solved = env.current_position == env.goal

    result = {
        "seed": seed,
        "solved": solved,
        "steps": env.count_steps,
        "reward": total_reward,
        "manhattan_distance": manhattan,
        "shortest_path": shortest,
        "detour": shortest - manhattan,
        "trap_hits": trap_hits,
        "trap_hits_by_action_rank": trap_hits_by_rank,
        "trap_hits_random_action": trap_hits_random,
        "collisions": collisions,
        "guardrail_steps": guardrail_steps,
        "random_fallback_steps": random_fallback_steps,
        "unique_cells": len(set(positions)),
        "start": list(env.start),
        "goal": list(env.goal),
    }

    # The full trajectory and grid are only stored for failures, to keep the file small.
    if not solved:
        result["positions"] = [list(position) for position in positions]
        result["grid"] = env.grid.tolist()

    return result


def collect_results():
    agent = load_agent()

    frozen_results = []
    for seed in FROZEN_SEEDS:
        frozen_results.append(run_episode(agent, seed))

    new_results = []
    for index, seed in enumerate(NEW_SEEDS):
        new_results.append(run_episode(agent, seed))
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(NEW_SEEDS)} cartes evaluees")

    frozen_solved = sum(1 for result in frozen_results if result["solved"])
    new_solved = sum(1 for result in new_results if result["solved"])
    print(f"Cartes gelees : {frozen_solved}/{len(frozen_results)} resolues")
    print(f"Cartes neuves : {new_solved}/{len(new_results)} resolues")

    return {
        "checkpoint": CHECKPOINT_PATH,
        "frozen": frozen_results,
        "new": new_results,
    }


TRAP_LOOP = "trap loop"
LOST_WITH_DETOUR = "lost, detour needed"
LOST_WITHOUT_DETOUR = "lost, no detour needed"

# Thresholds used to sort the failures into categories.
MANY_TRAP_HITS = 15
MANY_RANDOM_STEPS = 150


def classify_failure(result):
    if result["trap_hits"] >= MANY_TRAP_HITS and result["random_fallback_steps"] < MANY_RANDOM_STEPS:
        return TRAP_LOOP
    if result["detour"] > 0:
        return LOST_WITH_DETOUR
    return LOST_WITHOUT_DETOUR


def average(results, key):
    total = 0
    for result in results:
        total += result[key]
    return total / len(results)


def summarize(results):
    new_results = results["new"]

    failures = []
    successes = []
    for result in new_results:
        if result["solved"]:
            successes.append(result)
        else:
            failures.append(result)

    categories = {}
    for result in failures:
        category = classify_failure(result)
        if category not in categories:
            categories[category] = []
        categories[category].append(result["seed"])

    detour_stats = {"no_detour": {"maps": 0, "failures": 0}, "detour": {"maps": 0, "failures": 0}}
    for result in new_results:
        if result["detour"] == 0:
            key = "no_detour"
        else:
            key = "detour"
        detour_stats[key]["maps"] += 1
        if not result["solved"]:
            detour_stats[key]["failures"] += 1

    compared_keys = [
        "trap_hits",
        "collisions",
        "guardrail_steps",
        "random_fallback_steps",
        "unique_cells",
        "detour",
    ]
    means = {}
    for key in compared_keys:
        means[key] = {
            "failures": average(failures, key),
            "successes": average(successes, key),
        }

    frozen_solved = 0
    for result in results["frozen"]:
        if result["solved"]:
            frozen_solved += 1

    # How the traps were entered: by which action rank of the network (0 = favourite, 3 = least liked).
    trap_entries = {}
    for name, group in [("failures", failures), ("successes", successes)]:
        by_rank = [0, 0, 0, 0]
        random_action = 0
        for result in group:
            for rank in range(4):
                by_rank[rank] += result["trap_hits_by_action_rank"][rank]
            random_action += result["trap_hits_random_action"]
        trap_entries[name] = {"by_action_rank": by_rank, "random_action": random_action}

    summary = {
        "frozen_maps_solved": frozen_solved,
        "frozen_maps_total": len(results["frozen"]),
        "new_maps_solved": len(successes),
        "new_maps_total": len(new_results),
        "failure_categories": categories,
        "failure_rate_by_detour": detour_stats,
        "trap_entries": trap_entries,
        "mean_per_map": means,
    }
    return summary


def pick_typical_failures(results):
    failures = []
    for result in results["new"]:
        if not result["solved"]:
            failures.append(result)

    by_category = {TRAP_LOOP: [], LOST_WITH_DETOUR: [], LOST_WITHOUT_DETOUR: []}
    for result in failures:
        by_category[classify_failure(result)].append(result)

    typical = []

    # Trap loop: the cleanest loop is the one with the fewest random steps.
    best = None
    for result in by_category[TRAP_LOOP]:
        if best is None or result["random_fallback_steps"] < best["random_fallback_steps"]:
            best = result
    if best is not None:
        typical.append((TRAP_LOOP, best))

    # Detour: the map where the agent has to move away from the goal the most.
    best = None
    for result in by_category[LOST_WITH_DETOUR]:
        if best is None or result["detour"] > best["detour"]:
            best = result
    if best is not None:
        typical.append((LOST_WITH_DETOUR, best))

    # No detour: the map where the policy relies the most on random fallback.
    best = None
    for result in by_category[LOST_WITHOUT_DETOUR]:
        if best is None or result["random_fallback_steps"] > best["random_fallback_steps"]:
            best = result
    if best is not None:
        typical.append((LOST_WITHOUT_DETOUR, best))

    return typical


def plot_trajectory(result, category, output_path):
    grid = np.array(result["grid"])
    positions = result["positions"]

    visits = np.zeros(grid.shape, dtype=int)
    for x, y in positions:
        visits[x, y] += 1

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    draw_map(ax, grid, result["start"], result["goal"])

    # A trap sends the agent back to the start, so the line is cut there.
    xs, ys = trail_points(positions, len(positions) - 1)
    ax.plot(xs, ys, color=(0.95, 0.55, 0.10), linewidth=1.6, alpha=0.55, zorder=6)

    # Number of visits on the cells the agent came back to many times.
    outline = [patheffects.withStroke(linewidth=2.5, foreground="black")]
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if visits[x, y] >= 5:
                ax.text(y + 0.5, x + 0.5, str(visits[x, y]), ha="center", va="center", fontsize=8, color="white", path_effects=outline, zorder=12)

    draw_agent(ax, positions[-1])

    ax.set_title(
        f"Map seed {result['seed']}: {category}\n"
        f"{result['trap_hits']} trap hits, {result['collisions']} collisions, "
        f"{result['random_fallback_steps']} random steps, "
        f"{result['unique_cells']} cells visited (numbers = visits)",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if RESULTS_PATH.exists():
        print(f"Resultats deja presents, on les recharge : {RESULTS_PATH}")
        with open(RESULTS_PATH, "r", encoding="utf-8") as file:
            results = json.load(file)
    else:
        results = collect_results()
        with open(RESULTS_PATH, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"Resultats enregistres : {RESULTS_PATH}")

    summary = summarize(results)
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
    print(f"Resume enregistre : {summary_path}")

    for category, result in pick_typical_failures(results):
        file_name = "failure_" + category.replace(", ", "_").replace(" ", "_") + f"_seed_{result['seed']}.png"
        output_path = OUTPUT_DIR / file_name
        plot_trajectory(result, category, output_path)
        print(f"Trajectoire enregistree : {output_path}")


if __name__ == "__main__":
    main()
