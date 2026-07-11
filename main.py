print("Reinforcement learning project started")

from environment.environment import *

env = Environment(size=10, density=0.2)

env.render()

print("RIGHT")

env.step("RIGHT")

env.render()

