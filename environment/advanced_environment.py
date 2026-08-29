import random
import numpy as np
from environment.environment import Environment
from collections import deque

class AdvancedEnvironment(Environment):
    def __init__(
        self,
        size=10,
        obstacle_density=0.2,
        trap_density=0.05,
        mud_density=0.10,
        seed=None,
    ):
        super().__init__(size, obstacle_density, seed)
        if obstacle_density + trap_density + mud_density > 1:
            raise ValueError("Total density is > 1")
        self.trap_density = trap_density
        self.mud_density = mud_density

        while True:
            self.create_grid()
            self.start_and_goal()
            self.add_obstacles()
            self.add_mud()
            self.add_traps()


            if self.path_exists():
                break
        self.current_position = self.start

    def reset(self):
        super().reset()
        while True:
            self.create_grid()
            self.start_and_goal()
            self.add_obstacles()
            self.add_mud()
            self.add_traps()

            if self.path_exists():
                break
        self.current_position = self.start
        return self.current_position
    
    def reset_DQN(self):
        self.count_steps = 0
        while True:
            self.create_grid()
            self.start_and_goal()
            self.add_obstacles()
            self.add_mud()
            self.add_traps()

            if self.path_exists():
                break
        self.current_position = self.start
        return self.local_vision()

    def step(self, action):
        if action not in self.actions:
            raise ValueError("Action must be RIGHT, LEFT, UP or DOWN")

        x, y = self.current_position
        dx, dy = self.actions[action]
        new_position = (x + dx, y + dy)
        x1, y1 = new_position
        reward = 0
        done = False
        info = {"trap": False, "collision": False, "mud": False}

        if 0 <= x1 < self.size and 0 <= y1 < self.size:
            if self.grid[x1, y1] == 0:
                self.current_position = new_position
                reward = -1

            elif self.grid[x1, y1] == 2:
                self.current_position = self.start
                reward = -25
                info["trap"] = True

            elif self.grid[x1, y1] == 3:
                self.current_position = new_position
                reward = -4
                info["mud"] = True

            else:
                reward = -6
                info["collision"] = True
        else:
            reward = -6
            info["collision"] = True

        self.count_steps += 1

        if self.current_position == self.goal:
            reward = 100
            done = True
        elif self.count_steps >= self.max_steps:
            done = True
        return self.current_position, reward, done, info


    def step_DQN(self, action):
        if action not in self.actions:
            raise ValueError("Action must be RIGHT, LEFT, UP or DOWN")

        x, y = self.current_position
        dx, dy = self.actions[action]
        new_position = (x + dx, y + dy)
        x1, y1 = new_position
        reward = 0
        done = False

        if 0 <= x1 < self.size and 0 <= y1 < self.size:
            if self.grid[x1, y1] == 0:
                self.current_position = new_position
                reward = -1

            elif self.grid[x1, y1] == 2:
                self.current_position = self.start
                reward = -25

            elif self.grid[x1, y1] == 3:
                self.current_position = new_position
                reward = -4

            else:
                reward = -6
        else:
            reward = -6

        self.count_steps += 1

        if self.current_position == self.goal:
            reward = 100
            done = True
        elif self.count_steps >= self.max_steps:
            done = True

        return self.local_vision(), reward, done

    def render(self):
        display = np.full(self.grid.shape, ".", dtype="<U1")

        display[self.grid == 1] = "#"
        display[self.grid == 2] = "T"
        display[self.grid == 3] = "M"

        display[self.start] = "S"
        display[self.goal] = "G"
        display[self.current_position] = "A"

        for row in display:
            print(" ".join(row))

    def path_exists(self):
        queue = deque([self.start])
        visited = {self.start}

        while queue:
            current = queue.popleft()

            if current == self.goal:
                return True

            for dr, dc in self.actions.values():
                next_position = (
                    current[0] + dr,
                    current[1] + dc
                )

                row, col = next_position

                if not (0 <= row < self.size and 0 <= col < self.size):
                    continue

                if self.grid[row][col] == 1 or self.grid[row][col] == 2:
                    continue

                if next_position not in visited:
                    visited.add(next_position)
                    queue.append(next_position)

        return False


    def add_traps(self):

        free_cells = [
            (i, j)
            for i in range(self.grid.shape[0])
            for j in range(self.grid.shape[1])
            if self.grid[i, j] == 0 and (i, j) not in (self.start, self.goal)
        ]

        num_traps = int(self.trap_density * self.grid.shape[0] * self.grid.shape[1])
        if num_traps > len(free_cells):
            raise ValueError("Not enough free cells to place traps")

        traps = random.sample(free_cells, num_traps)

        for x, y in traps:
            self.grid[x, y] = 2
        return self.grid


    def add_mud(self):

        free_cells = [
            (i, j)
            for i in range(self.grid.shape[0])
            for j in range(self.grid.shape[1])
            if self.grid[i, j] == 0 and (i, j) not in (self.start, self.goal)
        ]

        num_mud = int(self.mud_density * self.grid.shape[0] * self.grid.shape[1])
        if num_mud > len(free_cells):
            raise ValueError("Not enough free cells to place mud")

        mud = random.sample(free_cells, num_mud)

        for x, y in mud:
            self.grid[x, y] = 3
        return self.grid


    def local_vision(self, nb_features=4, window_size=5):
        window = np.zeros(
            (nb_features, window_size, window_size),
            dtype=np.float32,
        )

        x, y = self.current_position
        half = window_size // 2

        for i, dx in enumerate(range(-half, half + 1)):
            for j, dy in enumerate(range(-half, half + 1)):
                gx, gy = x + dx, y + dy

                if not (0 <= gx < self.size and 0 <= gy < self.size):
                    window[0, i, j] = 1
                elif self.grid[gx, gy] != 0:
                    window[self.grid[gx, gy], i, j] = 1

        gx, gy = self.goal
        goal_vector = np.array(
            [(gx - x) / self.size, (gy - y) / self.size],
            dtype=np.float32,
        )

        return np.concatenate([window.flatten(), goal_vector])