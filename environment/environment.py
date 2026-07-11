import numpy as np
import random

class Environment:
    def __init__(self, size=10, density=0.2):
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("Size must be an integer")
        if not isinstance(density, (int, float)) or isinstance(density, bool):
            raise TypeError("Density must be a number")
        if size <= 0:
            raise ValueError("Size must be a positive integer")
        if not (0 <= density <= 1):
            raise ValueError("Density must be between 0 and 1")
        if size * size - int(density * size * size) < 2:
            raise ValueError("Not enough free space for start and goal positions")
        
        self.size = size
        self.density = density
        self.grid = create_grid(size)
        self.grid = add_obstacles(self.grid, density)
        self.current_position, self.goal = start_and_goal(self.grid)
        self.count_steps = 0
        self.max_steps = 4 * self.size * self.size

    def reset(self):
        self.grid = create_grid(self.size)
        self.grid = add_obstacles(self.grid, self.density)
        self.current_position, self.goal = start_and_goal(self.grid)
        self.count_steps = 0
        return self.current_position

    def step(self, action):
        x, y = self.current_position
        n = self.size
        reward = 0
        done = False
        if action == "RIGHT":
            if y+1<n and self.grid[x, y+1] == 0:
                self.current_position = x, y+1
                reward = -1
            else:
                reward = -5
        elif action == "LEFT":
            if y-1>=0 and self.grid[x, y-1] == 0:
                self.current_position = x, y-1
                reward = -1
            else:
                reward = -5
        elif action == "DOWN":
            if x+1<n and self.grid[x+1, y] == 0:
                self.current_position = x+1, y
                reward = -1
            else:
                reward = -5
        elif action == "UP":
            if x-1>=0 and self.grid[x-1, y] == 0:
                self.current_position = x-1, y
                reward = -1
            else:
                reward = -5
        else:
            raise ValueError("Action must be RIGHT, LEFT, UP or DOWN")
        self.count_steps += 1
        if self.current_position == self.goal:
            reward = 100
            done = True
        elif self.count_steps >= self.max_steps:
            done = True
        return self.current_position, reward, done
    
    def render(self):
        display_grid(self.grid, self.current_position, self.goal)

def create_grid(size = 10):
    grid = np.zeros((size, size), dtype=int)
    return grid

def add_obstacles(grid, density=0.2):
    coordinates = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    num_obstacles = int(density * len(coordinates))
    obstacles = random.sample(coordinates, num_obstacles)
    for x, y in obstacles:
        grid[x, y] = 1
    return grid

def start_and_goal(grid):
    coordinates = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1]) if grid[i,j] == 0]
    start, goal = random.sample(coordinates, 2)
    return start, goal

def display_grid(grid, current_position, goal):
    display = np.full(grid.shape, ".", dtype="<U1")
    display[grid == 1] = "#"
    display[current_position] = "A"
    display[goal] = "G"
    for row in display:
        print(" ".join(row))
