import copy
import pickle
import random
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from agents.DQN_agent import DQN_agent
from agents.q_learning_agent import QLearningAgent
from environment.advanced_environment import AdvancedEnvironment
from experiments.dqn.DQN_training import FRAME_DIM, evaluate
from utils.frame_stack import FrameStack
from utils.render import draw_agent, draw_legend, draw_map, remove_artists, trail_points

SIZE = 15
OBSTACLE_DENSITY = 0.12
TRAP_DENSITY = 0.04
MUD_DENSITY = 0.10
STACK_SIZE = 4

UNSEEN_SEEDS = tuple(range(1001, 1201))

Q_TABLE_PATH = "results/q_learning/environment_comparison/advanced/q_table.pkl"
Q_GRID_PATH = "results/q_learning/environment_comparison/advanced/grid.npy"
Q_LEARNING_START = (2, 0)
Q_LEARNING_GOAL = (8, 12)
CHECKPOINT_DIR = Path("results") / "dqn" / "dqn_checkpoints"
OUTPUT_DIR = Path("results") / "animations"

# Checkpoints shown in the training progress animation.
PROGRESS_CHECKPOINTS = [
    ("Episode 0", "policy_net_ep00000.pth"),
    ("Episode 100", "policy_net_ep00100.pth"),
    ("Episode 500", "policy_net_ep00500.pth"),
    ("Episode 1000", "policy_net_ep01000.pth"),
    ("Best checkpoint", "policy_net_best.pth"),
]

TRAIL_COLOR = (0.95, 0.55, 0.10)
GOOD_COLOR = (0.10, 0.50, 0.15)
BAD_COLOR = (0.75, 0.10, 0.10)


def make_env(seed):
    return AdvancedEnvironment(
        size=SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        trap_density=TRAP_DENSITY,
        mud_density=MUD_DENSITY,
        seed=seed,
    )


def load_dqn_agent(checkpoint_path):
    agent = DQN_agent(obs_dim=FRAME_DIM * STACK_SIZE, gamma=0.99, epsilon=0.0)
    agent.policy_net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    agent.policy_net.eval()
    return agent


def play_dqn(agent, seed):
    """Play one map with the greedy policy and the anti-cycle guardrail."""
    env = make_env(seed)
    frame_stack = FrameStack(STACK_SIZE)
    state = frame_stack.reset(env.local_vision())

    recent_positions = deque(maxlen=2 * SIZE)
    recent_positions.append(env.current_position)
    stuck_streak = 0

    positions = [env.current_position]
    trap_hits = 0
    done = False

    while not done:
        ranked = agent.ranked_actions(state)
        if stuck_streak >= len(ranked):
            action = random.randrange(agent.n_actions)
        else:
            action = ranked[stuck_streak]

        next_frame, reward, done = env.step_DQN(agent.actions[action])
        state = frame_stack.push(next_frame)

        if reward == -25:
            trap_hits += 1

        if env.current_position in recent_positions:
            stuck_streak += 1
        else:
            stuck_streak = 0
        recent_positions.append(env.current_position)
        positions.append(env.current_position)

    return {
        "seed": seed,
        "positions": positions,
        "solved": env.current_position == env.goal,
        "steps": env.count_steps,
        "trap_hits": trap_hits,
        "grid": env.grid.copy(),
        "start": env.start,
        "goal": env.goal,
    }


def play_q_learning(q_table, seed):
    """Play one map with a trained Q-table, same guardrail as the DQN."""
    agent = QLearningAgent(epsilon=0.0)
    agent.q_table = copy.deepcopy(q_table)

    env = make_env(seed)
    state = env.current_position

    recent_positions = deque(maxlen=2 * SIZE)
    recent_positions.append(state)
    stuck_streak = 0

    positions = [state]
    trap_hits = 0
    done = False
    step_count = 0

    while not done and step_count < env.max_steps:
        agent.verify_state_exists(state)
        ranked_actions = sorted(
            agent.q_table[state],
            key=lambda action: agent.q_table[state][action],
            reverse=True,
        )

        if stuck_streak >= len(ranked_actions):
            action = random.choice(agent.actions)
        else:
            action = ranked_actions[stuck_streak]

        next_state, reward, done, info = env.step(action)
        step_count += 1
        if info["trap"]:
            trap_hits += 1

        if next_state in recent_positions:
            stuck_streak += 1
        else:
            stuck_streak = 0
        recent_positions.append(next_state)
        positions.append(next_state)
        state = next_state

    return {
        "seed": seed,
        "positions": positions,
        "solved": state == env.goal,
        "steps": step_count,
        "trap_hits": trap_hits,
        "grid": env.grid.copy(),
        "start": env.start,
        "goal": env.goal,
    }


def choose_demo_seed(q_table, dqn_agent):
    """A map where the Q-table fails and the DQN solves quickly without
    touching a trap. Among them, the one with the longest trip."""
    best_seed = None
    best_distance = -1

    for seed in UNSEEN_SEEDS:
        dqn_play = play_dqn(dqn_agent, seed)
        if not dqn_play["solved"] or dqn_play["trap_hits"] > 0 or dqn_play["steps"] > 60:
            continue

        q_play = play_q_learning(q_table, seed)
        if q_play["solved"]:
            continue

        start = dqn_play["start"]
        goal = dqn_play["goal"]
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        if distance > best_distance:
            best_distance = distance
            best_seed = seed

    return best_seed


def count_trap_hits_until(positions, last_index):
    count = 0
    for index in range(1, last_index + 1):
        row, col = positions[index]
        previous_row, previous_col = positions[index - 1]
        if abs(row - previous_row) + abs(col - previous_col) > 1:
            count += 1
    return count


def frame_to_image(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[:, :, :3].copy())


def save_gif(images, output_path, frame_ms):
    # One palette for all the frames, so that only the moving parts change
    # from one frame to the next and the file stays small.
    first = images[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [first]
    for image in images[1:]:
        frames.append(image.quantize(palette=first, dither=Image.Dither.NONE))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"GIF enregistre : {output_path} ({len(frames)} images, {size_mb:.1f} Mo)")


def make_comparison_gif(q_play, dqn_play, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.8), dpi=70)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.02, right=0.98, wspace=0.05)

    fig.suptitle("Same unseen map, same guardrail", fontsize=14, y=0.97)

    plays = [q_play, dqn_play]
    titles = ["Q-learning (trained on one map)", "DQN (trained on 50 maps)"]
    trail_lines = []
    counters = []
    for ax, play, title in zip(axes, plays, titles):
        draw_map(ax, play["grid"], play["start"], play["goal"])
        ax.set_title(title, fontsize=12)
        line, = ax.plot([], [], color=TRAIL_COLOR, linewidth=2.2, alpha=0.8, zorder=6)
        trail_lines.append(line)
        counter = ax.text(0.5, -0.03, "", transform=ax.transAxes, ha="center", va="top", fontsize=12)
        counters.append(counter)

    # The Q-learning run is much longer, so it is played faster.
    q_stride = int(np.ceil((len(q_play["positions"]) - 1) / 110))
    total_frames = 130
    hold_frames = 25

    agent_artists = [None, None]
    images = []

    for frame in range(total_frames + hold_frames):
        indexes = [
            min(len(q_play["positions"]) - 1, frame * q_stride),
            min(len(dqn_play["positions"]) - 1, frame),
        ]

        for panel in range(2):
            play = plays[panel]
            index = indexes[panel]

            if agent_artists[panel] is not None:
                remove_artists(agent_artists[panel])
            agent_artists[panel] = draw_agent(axes[panel], play["positions"][index])

            xs, ys = trail_points(play["positions"], index)
            trail_lines[panel].set_data(xs, ys)

            traps = count_trap_hits_until(play["positions"], index)
            finished = index == len(play["positions"]) - 1
            if finished and play["solved"]:
                text = f"Goal reached in {index} steps"
                counters[panel].set_color(GOOD_COLOR)
            elif finished:
                text = f"Not solved after {index} steps, {traps} trap hits"
                counters[panel].set_color(BAD_COLOR)
            else:
                text = f"step {index}, trap hits {traps}"
                counters[panel].set_color("black")
            counters[panel].set_text(text)

        images.append(frame_to_image(fig))

    plt.close(fig)
    save_gif(images, output_path, frame_ms=110)


def make_progress_gif(segments, output_path):
    first_play = segments[0]["play"]

    fig, ax = plt.subplots(figsize=(5.6, 6.2), dpi=64)
    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.03, right=0.97)
    draw_map(ax, first_play["grid"], first_play["start"], first_play["goal"])

    title = fig.suptitle("", fontsize=12, y=0.975)
    trail_line, = ax.plot([], [], color=TRAIL_COLOR, linewidth=2.2, alpha=0.8, zorder=6)
    counter = ax.text(0.5, -0.025, "", transform=ax.transAxes, ha="center", va="top", fontsize=12)

    agent_artists = None
    images = []

    for segment in segments:
        play = segment["play"]
        title.set_text(f"{segment['label']}\n{segment['solved_percent']:.0f}% of 200 unseen maps solved")

        stride = int(np.ceil((len(play["positions"]) - 1) / 60))
        indexes = list(range(0, len(play["positions"]), stride))
        if indexes[-1] != len(play["positions"]) - 1:
            indexes.append(len(play["positions"]) - 1)
        indexes = indexes + [indexes[-1]] * 10

        for index in indexes:
            if agent_artists is not None:
                remove_artists(agent_artists)
            agent_artists = draw_agent(ax, play["positions"][index])

            xs, ys = trail_points(play["positions"], index)
            trail_line.set_data(xs, ys)

            traps = count_trap_hits_until(play["positions"], index)
            finished = index == len(play["positions"]) - 1
            if finished and play["solved"]:
                counter.set_text(f"Goal reached in {index} steps")
                counter.set_color(GOOD_COLOR)
            elif finished:
                counter.set_text(f"Not solved after {index} steps, {traps} trap hits")
                counter.set_color(BAD_COLOR)
            else:
                counter.set_text(f"step {index}, trap hits {traps}")
                counter.set_color("black")

            images.append(frame_to_image(fig))

    plt.close(fig)
    save_gif(images, output_path, frame_ms=110)


def solved_percent_on_unseen_maps(agent):
    results = evaluate(
        agent,
        eval_seeds=UNSEEN_SEEDS,
        size=SIZE,
        obstacle_density=OBSTACLE_DENSITY,
        trap_density=TRAP_DENSITY,
        mud_density=MUD_DENSITY,
        stack_size=STACK_SIZE,
    )
    solved = 0
    for steps in results["steps"]:
        if steps < SIZE * SIZE * 4:
            solved += 1
    return 100 * solved / len(UNSEEN_SEEDS)


def make_environment_image(output_path):
    """The map used to train the Q-learning agent, drawn with the same tiles
    as the animations. The environment code has changed since that training,
    so the map is read from the saved grid and not generated again. Start and
    goal come from the original figure (distance 18, like the 18 steps of the
    greedy evaluation)."""
    grid = np.load(Q_GRID_PATH)

    fig = plt.figure(figsize=(6.4, 7.2), dpi=120)
    map_ax = fig.add_axes([0.03, 0.15, 0.94, 0.78])
    legend_ax = fig.add_axes([0.03, 0.01, 0.94, 0.12])

    draw_map(map_ax, grid, Q_LEARNING_START, Q_LEARNING_GOAL)
    draw_agent(map_ax, Q_LEARNING_START)
    map_ax.set_title("Advanced environment", fontsize=13)
    draw_legend(legend_ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Image enregistree : {output_path}")


def count_inversions(steps_list):
    """Number of pairs where a later checkpoint needs more steps than an earlier one."""
    inversions = 0
    for first in range(len(steps_list)):
        for second in range(first + 1, len(steps_list)):
            if steps_list[first] < steps_list[second]:
                inversions += 1
    return inversions


def choose_progress_seed(agents):
    """A map that every checkpoint from the second one on solves, with a
    first checkpoint that fails, and trips that get shorter as the training
    goes on. The percentages shown in the animation still come from all the
    unseen maps, this map is only used to show how the routes change."""
    best_seed = None
    best_score = None

    # Cheapest checks first: a failed run lasts 900 steps.
    check_order = list(range(len(agents) - 1, -1, -1))

    for seed in UNSEEN_SEEDS:
        plays = {}
        accepted = True
        for index in check_order:
            play = play_dqn(agents[index], seed)
            plays[index] = play
            if index == 0:
                if play["solved"]:
                    accepted = False
            elif not play["solved"] or play["steps"] > 400:
                accepted = False
            if not accepted:
                break

        if not accepted:
            continue

        steps_list = []
        for index in range(1, len(agents)):
            steps_list.append(plays[index]["steps"])

        inversions = count_inversions(steps_list)
        improvement = steps_list[0] - steps_list[-1]
        if improvement <= 0:
            continue

        score = (inversions, -improvement)
        if best_score is None or score < best_score:
            best_score = score
            best_seed = seed

    return best_seed


def main():
    make_environment_image(OUTPUT_DIR / "environment.png")

    with open(Q_TABLE_PATH, "rb") as file:
        q_table = pickle.load(file)

    reference_agent = load_dqn_agent(CHECKPOINT_DIR / "policy_net_best.pth")

    demo_seed = choose_demo_seed(q_table, reference_agent)
    print(f"Carte de demonstration : seed {demo_seed}")

    q_play = play_q_learning(q_table, demo_seed)
    dqn_play = play_dqn(reference_agent, demo_seed)
    print(f"Q-learning : resolu={q_play['solved']}, pas={q_play['steps']}")
    print(f"DQN : resolu={dqn_play['solved']}, pas={dqn_play['steps']}")

    make_comparison_gif(q_play, dqn_play, OUTPUT_DIR / "q_learning_vs_dqn.gif")

    agents = []
    for label, file_name in PROGRESS_CHECKPOINTS:
        agents.append(load_dqn_agent(CHECKPOINT_DIR / file_name))

    progress_seed = choose_progress_seed(agents)
    print(f"Carte pour l'animation des progres : seed {progress_seed}")

    segments = []
    for (label, file_name), agent in zip(PROGRESS_CHECKPOINTS, agents):
        percent = solved_percent_on_unseen_maps(agent)
        play = play_dqn(agent, progress_seed)
        print(f"{label} : {percent:.1f}% des cartes neuves, carte choisie resolue={play['solved']} en {play['steps']} pas")
        segments.append({"label": label, "solved_percent": percent, "play": play})

    make_progress_gif(segments, OUTPUT_DIR / "dqn_training_progress.gif")


if __name__ == "__main__":
    main()
