import json
import pickle
import random as rd
from collections import deque
from pathlib import Path

import torch

from agents.q_learning_agent import QLearningAgent
from agents.DQN_agent import DQN_agent
from environment.advanced_environment import AdvancedEnvironment
from experiments.DQN_training import FRAME_DIM
from utils.frame_stack import FrameStack

EVAL_SEEDS = list(range(101, 116))
SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10
STACK_SIZE = 4

Q_TABLE_PATH = "results/environment_comparison/advanced/q_table.pkl"
DQN_CHECKPOINT_PATH = "results/dqn_checkpoints/policy_net_best.pth"
OUTPUT_DIR = Path("results/generalization_comparison")


def evaluate_q_learning(q_table):
    """Evalue une Q-table deja entrainee, en politique gloutonne, sur les
    15 cartes gelees. Meme garde-fou anti-cycle que celui utilise pour le
    DQN (positions recentes, escalade des actions, repli aleatoire)."""
    agent = QLearningAgent(epsilon=0.0)
    agent.q_table = q_table

    revisit_window = 2 * SIZE

    per_seed_results = []
    new_state_count = 0
    total_state_count = 0

    for seed in EVAL_SEEDS:
        env = AdvancedEnvironment(
            size=SIZE,
            obstacle_density=OBSTACLE_DENSITY,
            trap_density=TRAP_DENSITY,
            mud_density=MUD_DENSITY,
            seed=seed,
        )

        state = env.current_position

        recent_positions = deque(maxlen=revisit_window)
        recent_positions.append(state)
        stuck_streak = 0

        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < env.max_steps:
            if state not in q_table:
                new_state_count += 1
            total_state_count += 1
            agent.verify_state_exists(state)

            ranked_actions = sorted(
                agent.q_table[state],
                key=lambda action: agent.q_table[state][action],
                reverse=True,
            )

            if stuck_streak >= len(ranked_actions):
                action = rd.choice(agent.actions)
            else:
                action = ranked_actions[stuck_streak]

            next_state, reward, done, info = env.step(action)
            step_count += 1
            episode_reward += reward

            if next_state in recent_positions:
                stuck_streak += 1
            else:
                stuck_streak = 0
            recent_positions.append(next_state)

            state = next_state

        resolved = state == env.goal
        per_seed_results.append({
            "seed": seed,
            "resolved": resolved,
            "steps": step_count,
            "reward": episode_reward,
        })

    new_state_ratio = new_state_count / total_state_count

    return per_seed_results, new_state_ratio


def evaluate_dqn(checkpoint_path):
    """Reevalue le checkpoint DQN retenu sur les 15 cartes gelees, avec le
    meme garde-fou que celui deja utilise dans experiments/DQN_training.py."""
    agent = DQN_agent(obs_dim=FRAME_DIM * STACK_SIZE, gamma=0.99, epsilon=0.0)
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    agent.policy_net.eval()

    revisit_window = 2 * SIZE

    per_seed_results = []

    for seed in EVAL_SEEDS:
        env = AdvancedEnvironment(
            size=SIZE,
            obstacle_density=OBSTACLE_DENSITY,
            trap_density=TRAP_DENSITY,
            mud_density=MUD_DENSITY,
            seed=seed,
        )

        frame_stack = FrameStack(STACK_SIZE)
        state = frame_stack.reset(env.local_vision())

        recent_positions = deque(maxlen=revisit_window)
        recent_positions.append(env.current_position)
        stuck_streak = 0

        done = False
        episode_reward = 0
        step_count = 0

        while not done and step_count < env.max_steps:
            ranked_actions = agent.ranked_actions(state)

            if stuck_streak >= len(ranked_actions):
                action = rd.randrange(agent.n_actions)
            else:
                action = ranked_actions[stuck_streak]

            next_frame, reward, done = env.step_DQN(agent.actions[action])
            state = frame_stack.push(next_frame)
            step_count += 1
            episode_reward += reward

            if env.current_position in recent_positions:
                stuck_streak += 1
            else:
                stuck_streak = 0
            recent_positions.append(env.current_position)

        resolved = env.current_position == env.goal
        per_seed_results.append({
            "seed": seed,
            "resolved": resolved,
            "steps": step_count,
            "reward": episode_reward,
        })

    return per_seed_results


def summarize(per_seed_results):
    solved_count = 0
    total_reward = 0
    total_steps = 0

    for result in per_seed_results:
        if result["resolved"]:
            solved_count += 1
        total_reward += result["reward"]
        total_steps += result["steps"]

    count = len(per_seed_results)
    mean_reward = total_reward / count
    mean_steps = total_steps / count

    return solved_count, mean_reward, mean_steps


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(Q_TABLE_PATH, "rb") as file:
        q_table = pickle.load(file)

    q_learning_results, new_state_ratio = evaluate_q_learning(q_table)
    dqn_results = evaluate_dqn(DQN_CHECKPOINT_PATH)

    q_solved, q_reward, q_steps = summarize(q_learning_results)
    dqn_solved, dqn_reward, dqn_steps = summarize(dqn_results)

    print("Comparaison generalisation sur 15 cartes gelees")
    print("-" * 55)
    print(f"{'':20s}{'Q-learning':>15s}{'DQN':>15s}")
    print(f"{'Cartes resolues':20s}{q_solved:>12d}/15{dqn_solved:>12d}/15")
    print(f"{'Recompense moyenne':20s}{q_reward:>15.1f}{dqn_reward:>15.1f}")
    print(f"{'Pas moyens':20s}{q_steps:>15.1f}{dqn_steps:>15.1f}")
    print()
    print(f"Etats jamais vus pendant l'entrainement Q-learning : {new_state_ratio * 100:.1f}%")

    output = {
        "q_learning": {
            "per_seed": q_learning_results,
            "solved": q_solved,
            "mean_reward": q_reward,
            "mean_steps": q_steps,
            "new_state_ratio": new_state_ratio,
        },
        "dqn": {
            "per_seed": dqn_results,
            "solved": dqn_solved,
            "mean_reward": dqn_reward,
            "mean_steps": dqn_steps,
        },
    }

    output_path = OUTPUT_DIR / "evaluation.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print(f"\nResultats sauvegardes dans {output_path}")


if __name__ == "__main__":
    main()
