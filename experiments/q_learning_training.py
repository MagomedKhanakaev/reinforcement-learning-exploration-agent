from environment.advanced_environment import AdvancedEnvironment
from agents.q_learning_agent import QLearningAgent
from results.plots import plot_training_results

episodes = 5000
episode_rewards = []
episode_steps = []
episode_epsilons = []
episode_collisions = []
episode_traps = []

"""
# Simple
obstacle_density = 0.12
trap_density = 0.00
mud_density = 0.00

# Medium
obstacle_density = 0.12
trap_density = 0.04
mud_density = 0.10

# Hard
obstacle_density = 0.18
trap_density = 0.07
mud_density = 0.15
"""

size = 15
obstacle_density = 0.12
trap_density = 0.04
mud_density = 0.10

env = AdvancedEnvironment(size=size, obstacle_density=obstacle_density, trap_density=trap_density, mud_density=mud_density)

alpha = 0.1
gamma = 0.95
epsilon = 1.0

agent = QLearningAgent(alpha, gamma, epsilon)


for episode in range (episodes):
    state = env.reset_episode()
    done = False
    total_reward = 0
    collisions = 0
    count_traps = 0

    while not done:

        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)
        agent.update_q_table(state, action, reward, next_state, done)

        if info["collision"]:
            collisions +=1

        if info["trap"]:
            count_traps += 1

        state = next_state
        total_reward += reward

    episode_epsilons.append(agent.epsilon)
    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    episode_steps.append(env.count_steps)
    episode_collisions.append(collisions)
    episode_traps.append(count_traps)


"""
agent.epsilon = 0.0
state = env.reset_episode()
done = False

while not done:
    env.render()
    print()

    action = agent.choose_action(state)
    state, reward, done, info = env.step(action)

env.render()
"""

plot_training_results(
    episode_rewards=episode_rewards,
    episode_steps=episode_steps,
    episode_epsilons=episode_epsilons,
    episode_collisions=episode_collisions,
    episode_traps=episode_traps,
    window=100,
)