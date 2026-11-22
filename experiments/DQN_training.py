import random
from collections import deque
from pathlib import Path

import torch

from agents.DQN_agent import DQN_agent
from environment.advanced_environment import AdvancedEnvironment
from utils.frame_stack import FrameStack

FRAME_DIM = 4 * 5 * 5 + 2 


def evaluate(agent, eval_seeds, size, obstacle_density, trap_density, mud_density, stack_size, window_size=5):
    rng_state = random.getstate()
    previous_epsilon = agent.epsilon
    agent.epsilon = 0.0

    revisit_window = 2 * size

    rewards = []
    steps = []

    for seed in eval_seeds:
        env = AdvancedEnvironment(
            size=size,
            obstacle_density=obstacle_density,
            trap_density=trap_density,
            mud_density=mud_density,
            seed=seed,
            window_size=window_size,
        )
        frame_stack = FrameStack(stack_size)
        state = frame_stack.reset(env.local_vision(window_size=window_size))
        recent_positions = deque(maxlen=revisit_window)
        recent_positions.append(env.current_position)
        stuck_streak = 0
        done = False
        episode_reward = 0

        while not done:
            ranked = agent.ranked_actions(state)
            if stuck_streak >= len(ranked):
                action = random.randrange(agent.n_actions)
            else:
                action = ranked[stuck_streak]

            next_frame, reward, done = env.step_DQN(agent.actions[action])
            next_state = frame_stack.push(next_frame)
            episode_reward += reward
            state = next_state

            if env.current_position in recent_positions:
                stuck_streak += 1
            else:
                stuck_streak = 0
            recent_positions.append(env.current_position)

        rewards.append(episode_reward)
        steps.append(env.count_steps)

    agent.epsilon = previous_epsilon
    random.setstate(rng_state)

    return {
        "mean_reward": sum(rewards) / len(rewards),
        "mean_steps": sum(steps) / len(steps),
        "rewards": rewards,
        "steps": steps,
    }


def train(
    episodes=2000,
    size=15,
    obstacle_density=0.12,
    trap_density=0.04,
    mud_density=0.10,
    gamma=0.99,
    epsilon=1.0,
    seed=4,
    target_update_freq=500,
    eval_every=100,
    eval_seeds=tuple(range(101, 116)),
    checkpoint_dir="results/dqn_checkpoints",
    stack_size=4,
    pool_size=50,
    window_size=5,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pool = [
        AdvancedEnvironment(
            size=size,
            obstacle_density=obstacle_density,
            trap_density=trap_density,
            mud_density=mud_density,
            seed=seed + i,
            window_size=window_size,
        )
        for i in range(pool_size)
    ]

    frame_dim = 4 * window_size * window_size + 2
    obs_dim = frame_dim * stack_size
    agent = DQN_agent(obs_dim=obs_dim, gamma=gamma, epsilon=epsilon)

    global_steps = 0
    best_eval_reward = float("-inf")
    best_eval_episode = None

    recent_rewards = deque(maxlen=100)

    for episode in range(episodes):
        env = random.choice(pool)
        env.reset_episode()
        frame_stack = FrameStack(stack_size)
        state = frame_stack.reset(env.local_vision(window_size=window_size))

        done = False
        episode_reward = 0
        loss = None
        while not done:
            x, y = env.current_position
            dist_before = abs(x - env.goal[0]) + abs(y - env.goal[1])

            action = agent.choose_action(state)
            next_frame, reward, done = env.step_DQN(agent.actions[action])
            next_state = frame_stack.push(next_frame)

            if done:
                dist_after = 0
            else:
                xx, yy = env.current_position
                dist_after = abs(xx - env.goal[0]) + abs(yy - env.goal[1])
            reward += agent.gamma * (-dist_after) - (-dist_before)

            global_steps += 1

            agent.remember(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state

            if global_steps % 4 == 0:
                loss = agent.learn()

            if global_steps % target_update_freq == 0:
                agent.sync_target()

        recent_rewards.append(episode_reward)
        average_reward = sum(recent_rewards) / len(recent_rewards)

        if episode % 10 == 0:
            print(
                f"Épisode {episode} | "
                f"récompense={episode_reward:.1f} | "
                f"moyenne_100={average_reward:.1f} | "
                f"étapes={env.count_steps} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"loss={loss if loss is not None else 'N/A'}"
            )

        if eval_every and episode % eval_every == 0:
            eval_results = evaluate(
                agent,
                eval_seeds=eval_seeds,
                size=size,
                obstacle_density=obstacle_density,
                trap_density=trap_density,
                mud_density=mud_density,
                stack_size=stack_size,
                window_size=window_size,
            )
            print(
                f"  -- Évaluation (gloutonne, cartes fixes) -- "
                f"récompense_moyenne={eval_results['mean_reward']:.1f} | "
                f"pas_moyens={eval_results['mean_steps']:.1f}"
            )

            torch.save(
                agent.policy_net.state_dict(),
                checkpoint_dir / f"policy_net_ep{episode:05d}.pth",
            )

            if eval_results["mean_reward"] > best_eval_reward:
                best_eval_reward = eval_results["mean_reward"]
                best_eval_episode = episode
                torch.save(
                    agent.policy_net.state_dict(),
                    checkpoint_dir / "policy_net_best.pth",
                )

        agent.decay_epsilon()

    torch.save(
        agent.policy_net.state_dict(),
        checkpoint_dir / "policy_net_final.pth",
    )

    print(
        f"Meilleur checkpoint : épisode {best_eval_episode} "
        f"(récompense_moyenne={best_eval_reward:.1f}) -> policy_net_best.pth"
    )

if __name__ == "__main__":
    train(
        episodes=2000,
        target_update_freq=1000,
    )
