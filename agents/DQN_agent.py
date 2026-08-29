import torch
import torch.nn as nn
import torch.optim as optim
import random as rd
import numpy as np


class DQN(nn.Module):

    def __init__(self, obs_dim=4*5*5+2, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        batch = rd.sample(self.buffer, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = np.stack(states)
        next_states = np.stack(next_states)

        states_tensor = torch.from_numpy(states)
        actions_tensor = torch.as_tensor(actions, dtype=torch.long)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        next_states_tensor = torch.from_numpy(next_states)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

class DQN_agent:
    def __init__(
        self,
        obs_dim=4*5*5+2,
        n_actions=4,
        gamma=0.99,
        batch_size=64,
        epsilon = 1.0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.actions = {0 : "UP", 1 : "DOWN", 2 : "LEFT", 3 : "RIGHT"}
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon

        self.policy_net = DQN(self.obs_dim, self.n_actions)
        self.target_net = DQN(obs_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=1e-4,
            foreach=True,
        )
        self.replay_buffer = ReplayBuffer()

    def state_to_tensor(self, state):
        state_tensor = torch.from_numpy(state)
        state_tensor = state_tensor.unsqueeze(0)
        return state_tensor

    def choose_action(self, state):
        if rd.random() < self.epsilon:
            return rd.randrange(self.n_actions)

        state_tensor = self.state_to_tensor(state)

        with torch.inference_mode():
            q_values = self.policy_net(state_tensor)

        best_action = q_values.argmax(dim=1)
        return best_action.item()

    def ranked_actions(self, state):
        state_tensor = self.state_to_tensor(state)

        with torch.inference_mode():
            q_values = self.policy_net(state_tensor)

        sorted_actions = torch.argsort(q_values, dim=1, descending=True)
        sorted_actions = sorted_actions.squeeze(0)
        return sorted_actions.tolist()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = (
            self.replay_buffer.sample(self.batch_size)
        )

        q_values = self.policy_net(state)
        q_sa_values = []
        for i in range(len(action)):
            q_sa_values.append(q_values[i][action[i]])
        q_sa = torch.stack(q_sa_values)

        with torch.no_grad():
            next_q_values = self.policy_net(next_state)
            next_actions = next_q_values.argmax(dim=1)

            target_q_values = self.target_net(next_state)
            q_next_values = []
            for i in range(len(next_actions)):
                q_next_values.append(target_q_values[i][next_actions[i]])
            q_next = torch.stack(q_next_values)

            target = reward + self.gamma * q_next * (1.0 - done)

        loss = nn.functional.smooth_l1_loss(q_sa, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0,
            foreach=True,
        )

        self.optimizer.step()

        return loss.item()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, epsilon_min=0.05, epsilon_decay=0.999):
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
