import random
import numpy as np
from environment.environment import Environment

class AdvancedEnvironment(Environment):
    def __init__(self, size=10, obstacle_density=0.2, trap_density=0.05, mud_density=0.10):
        super().__init__(size, obstacle_density)
        if obstacle_density + trap_density + mud_density > 1:
            raise ValueError("Total density is > 1")
        self.trap_density = trap_density
        self.mud_density = mud_density

        self.grid = add_traps(self.grid, self.trap_density, self.start, self.goal)

        self.grid = add_mud(self.grid, self.mud_density, self.start, self.goal)


    def reset(self):
        super().reset()
        self.grid = add_traps(self.grid, self.trap_density, self.start, self.goal)
        self.grid = add_mud(self.grid, self.mud_density, self.start, self.goal)
        return self.current_position
        
    def step(self, action):
        if action not in self.actions:
            raise ValueError("Action must be RIGHT, LEFT, UP or DOWN")

        x, y = self.current_position
        dx, dy = self.actions[action]
        new_position = (x + dx, y + dy)
        x1, y1 = new_position
        reward = 0
        done = False
        info = {"trap" : False, "collision" : False, "mud" : False}

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

        self.count_steps += 1

        if self.current_position == self.goal:
            reward = 100
            done = True
        elif self.count_steps >= self.max_steps:
            done = True
        return self.current_position, reward, done, info
            
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


def add_traps(grid, trap_density, start, goal):

    free_cells = [
        (i, j)
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
        if grid[i, j] == 0 and (i, j) not in (start, goal)
    ]

    num_traps = int(trap_density * grid.shape[0] * grid.shape[1])
    if num_traps > len(free_cells):
        raise ValueError("Not enough free cells to place traps")
    
    traps = random.sample(free_cells, num_traps)

    for x, y in traps:
        grid[x, y] = 2
    return grid

def add_mud(grid, mud_density, start, goal):

    free_cells = [
        (i, j)
        for i in range(grid.shape[0])
        for j in range(grid.shape[1])
        if grid[i, j] == 0 and (i, j) not in (start, goal)
    ]

    num_mud = int(mud_density * grid.shape[0] * grid.shape[1])
    if num_mud > len(free_cells):
        raise ValueError("Not enough free cells to place mud")
    
    mud = random.sample(free_cells, num_mud)

    for x, y in mud:
        grid[x, y] = 3
    return grid