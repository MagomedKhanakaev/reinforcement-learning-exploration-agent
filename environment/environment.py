import numpy as np
import random


class Environment:
    def __init__(
        self,
        size=10,
        obstacle_density=0.2,
        seed=None,
    ):
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("Size must be an integer")
        if not isinstance(obstacle_density, (int, float)) or isinstance(
            obstacle_density, bool
        ):
            raise TypeError("Density must be a number")
        if size <= 0:
            raise ValueError("Size must be a positive integer")
        if not (0 <= obstacle_density <= 1):
            raise ValueError("Density must be between 0 and 1")
        if size * size - int(obstacle_density * size * size) < 2:
            raise ValueError("Not enough free space for start and goal positions")
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)

        self.size = size
        self.obstacle_density = obstacle_density
        self.grid = self.create_grid()
        self.start, self.goal = self.start_and_goal()
        self.grid = self.add_obstacles()
        self.current_position = self.start
        self.count_steps = 0
        self.max_steps = 4 * self.size * self.size
        self.actions = {"UP": (-1, 0), "DOWN": (1, 0), "RIGHT": (0, 1), "LEFT": (0, -1)}

    def reset(self):
        self.grid = self.create_grid()
        self.start, self.goal = self.start_and_goal()
        self.grid = self.add_obstacles()
        self.current_position = self.start
        self.count_steps = 0
        return self.current_position

    def reset_episode(self):
        self.count_steps = 0
        self.current_position = self.start
        return self.current_position

    def step(self, action):
        if action not in self.actions:
            raise ValueError("Action must be RIGHT, LEFT, UP or DOWN")

        x, y = self.current_position
        dx, dy = self.actions[action]
        new_position = (x + dx, y + dy)
        x1, y1 = new_position
        reward = -6
        done = False

        if 0 <= x1 < self.size and 0 <= y1 < self.size:
            if self.grid[x1, y1] == 0:
                self.current_position = new_position
                reward = -1

        self.count_steps += 1

        if self.current_position == self.goal:
            reward = 100
            done = True
        elif self.count_steps >= self.max_steps:
            done = True
        return self.current_position, reward, done

    def render(self):
        display = np.full(self.grid.shape, ".", dtype="<U1")
        display[self.grid == 1] = "#"
        display[self.current_position] = "A"
        display[self.goal] = "G"

        for row in display:
            print(" ".join(row))


    def create_grid(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        return self.grid


    def add_obstacles(self):
        coordinates = [(i, j) for i in range(self.grid.shape[0]) for j in range(self.grid.shape[1]) if (i, j) not in (self.start, self.goal)]
        num_obstacles = int(self.obstacle_density * self.grid.shape[0] * self.grid.shape[1])
        obstacles = random.sample(coordinates, num_obstacles)

        for x, y in obstacles:
           self.grid[x, y] = 1
        return self.grid


    def start_and_goal(self, min_distance=None):

        border_cells = [
            (i, j)
            for i in range(self.size) for j in range(self.size)
            if i in (0, self.grid.shape[0] - 1) or j in (0, self.grid.shape[1] - 1)
        ]


        if min_distance is None:
            min_distance = self.grid.shape[0]

        random.shuffle(border_cells)

        for start in border_cells:
            valid_goals = [
                (i, j)
                for i in range(self.size) for j in range(self.size)
                if (i, j) != start
                and abs(start[0] - i) + abs(start[1] - j) >= min_distance
            ]

            if valid_goals:
                self.start, self.goal = start, random.choice(valid_goals)
                return self.start, self.goal



