from environment.environment import Environment
import pytest
import numpy as np


def test_negative_size():

    with pytest.raises(ValueError):

        Environment(size=-2)


def test_negative_obstacle_density():

    with pytest.raises(ValueError):

        Environment(obstacle_density=-0.1)


def test_over_1_obstacle_density():

    with pytest.raises(ValueError):

        Environment(obstacle_density=1.1)


def test_not_enough_free_cells():

    with pytest.raises(ValueError):

        Environment(obstacle_density=1.0)


def test_equal_grid_with_same_seed():

    env1 = Environment(seed=4)
    env2 = Environment(seed=4)

    assert np.array_equal(env1.grid, env2.grid)
    assert env1.start == env2.start
    assert env1.goal == env2.goal


def test_reset_episode():

    env = Environment()
    env.count_steps = 45
    env.current_position = (0, 0)

    env.reset_episode()

    assert env.count_steps == 0
    assert env.current_position == env.start


def test_reset():

    env = Environment()
    env.count_steps = 45

    env.reset()

    assert env.current_position == env.start
    assert env.count_steps == 0


def test_valid_step():

    env = Environment(size=2, obstacle_density=0.0)
    env.start = (0, 0)
    env.goal = (1, 1)
    env.current_position = (0, 0)

    current_position, reward, done = env.step("DOWN")

    assert current_position == (1, 0)
    assert reward == -1
    assert done is False
    assert env.count_steps == 1


def test_collision():

    env = Environment(size=2, obstacle_density=0.0)
    env.start = (0, 0)
    env.goal = (1, 1)
    env.current_position = (0, 0)
    env.grid[1, 0] = 1

    current_position, reward, done = env.step("DOWN")

    assert current_position == (0, 0)
    assert reward == -6
    assert done is False
    assert env.count_steps == 1


def test_off_the_grid_step():

    env = Environment(size=2, obstacle_density=0.0)
    env.start = (0, 0)
    env.goal = (1, 1)
    env.current_position = (0, 0)

    current_position, reward, done = env.step("LEFT")

    assert current_position == (0, 0)
    assert reward == -6
    assert done is False
    assert env.count_steps == 1


def test_reach_goal():

    env = Environment(size=2, obstacle_density=0.0)
    env.start = (0, 0)
    env.goal = (1, 0)
    env.current_position = (0, 0)

    current_position, reward, done = env.step("DOWN")

    assert current_position == env.goal
    assert reward == 100
    assert done is True


def test_max_steps():

    env = Environment(size=2, obstacle_density=0.0)
    env.start = (0, 0)
    env.goal = (1, 1)
    env.current_position = (0, 0)
    env.count_steps = env.max_steps - 1

    current_position, reward, done = env.step("DOWN")

    assert current_position == (1, 0)
    assert reward == -1
    assert done is True


def test_invalid_action():

    env = Environment()

    with pytest.raises(ValueError):

        env.step("JUMP")


def test_start_different_than_goal():

    env = Environment()

    assert env.start != env.goal


def test_start_and_goal_on_free_cells():

    env = Environment()

    assert env.grid[env.start] == 0
    assert env.grid[env.goal] == 0


def test_distance_start_and_goal():

    env = Environment()

    assert (
        abs(env.start[0] - env.goal[0]) + abs(env.start[1] - env.goal[1])
        >= env.grid.shape[0]
    )
