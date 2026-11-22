from collections import deque

import numpy as np


class FrameStack:
    """Empile les `stack_size` dernières observations pour donner à un
    agent memoryless une courte fenetre d'historique (convention DQN Atari :
    au reset, la premiere frame est repetee pour remplir la pile).
    """

    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    def reset(self, frame):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return self._get_state()

    def push(self, frame):
        self.frames.append(frame)
        return self._get_state()

    def _get_state(self):
        return np.concatenate(list(self.frames))
