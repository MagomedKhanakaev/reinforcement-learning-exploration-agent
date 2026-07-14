from environment.environment import Environment
from agents.q_learning_agent import QLearningAgent

episodes = 100
episode_rewards = []
episode_steps = []
episode_successes = []
episode_collisions = [0 for _ in range(episodes)]


env = Environment()

alpha = 0.1
gamma = 0.95
epsilon = 1.0

agent = QLearningAgent(alpha, gamma, epsilon)

env.render()

for episode in range (episodes):
    state = env.reset_episode()
    done = False
    total_reward = 0
    collisions = 0

    while not done:

        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state, done)

        if state == next_state:
            collisions +=1

        state = next_state
        total_reward += reward

    agent.decay_epsilon()
    episode_rewards.append(total_reward)
    episode_steps.append(env.count_steps)
    episode_successes.append(env.current_position == env.goal)
    episode_collisions.append(collisions)



agent.epsilon = 0.0
state = env.reset_episode()
done = False

while not done:
    env.render()
    print()

    action = agent.choose_action(state)
    state, reward, done = env.step(action)

env.render()