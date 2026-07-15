import random as rd

class QLearningAgent:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.actions = ( "UP", "DOWN", "RIGHT", "LEFT" )

    def choose_action(self, state):
        self.verify_state_exists(state)

        if rd.random() < self.epsilon:
            best_action = rd.choice(self.actions)
        else:
            best_value = max(self.q_table[state].values())
            best_action = rd.choice([ action 
                                     for action, value in self.q_table[state].items() 
                                     if value == best_value ])

        return best_action
    
    def update_q_table(self, state, action, reward, next_state, done):
        
        old_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            self.verify_state_exists(next_state)
            target = reward + self.gamma * max(self.q_table[next_state].values())

        self.q_table[state][action] = (
            (1 - self.alpha) * old_q 
            + self.alpha * target 
        )

    def verify_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = { 
                action: 0.0 
                for action in self.actions
            }
    
    def decay_epsilon(self, epsilon_min=0.05, epsilon_decay=0.999):
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)
