import numpy as np
import pytest

from environment.advanced_environment import AdvancedEnvironment


def test_total_density_over_one():
    with pytest.raises(ValueError):
        AdvancedEnvironment(
            obstacle_density=0.5,
            trap_density=0.3,
            mud_density=0.3,
        )


def test_same_seed_gives_same_grid():
    env1 = AdvancedEnvironment(seed=4)
    env2 = AdvancedEnvironment(seed=4)

    assert np.array_equal(env1.grid, env2.grid)
    assert env1.start == env2.start
    assert env1.goal == env2.goal


def test_start_and_goal_are_free():
    env = AdvancedEnvironment(seed=4)

    assert env.grid[env.start] == 0
    assert env.grid[env.goal] == 0


def test_traps_are_added():
    env = AdvancedEnvironment(
        size=10,
        obstacle_density=0.0,
        trap_density=0.1,
        mud_density=0.0,
        seed=4,
    )

    expected_traps = int(0.1 * 10 * 10)

    assert np.sum(env.grid == 2) == expected_traps


def test_mud_is_added():
    env = AdvancedEnvironment(
        size=10,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.1,
        seed=4,
    )

    expected_mud = int(0.1 * 10 * 10)

    assert np.sum(env.grid == 3) == expected_mud


def test_normal_step():
    env = AdvancedEnvironment(
        size=3,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.0,
    )

    env.grid[:] = 0
    env.start = (0, 0)
    env.goal = (2, 2)
    env.current_position = (1, 1)

    position, reward, done, info = env.step("RIGHT")

    assert position == (1, 2)
    assert reward == -1
    assert done is False
    assert info == {
        "trap": False,
        "collision": False,
        "mud": False,
    }
    assert env.count_steps == 1


def test_obstacle_collision():
    env = AdvancedEnvironment(
        size=3,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.0,
    )

    env.grid[:] = 0
    env.start = (0, 0)
    env.goal = (2, 2)
    env.current_position = (1, 1)

    env.grid[1, 2] = 1

    position, reward, done, info = env.step("RIGHT")

    assert position == (1, 1)
    assert reward == -6
    assert done is False
    assert info["collision"] is True
    assert info["trap"] is False
    assert info["mud"] is False


def test_off_grid_collision():
    env = AdvancedEnvironment(
        size=3,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.0,
    )

    env.grid[:] = 0
    env.start = (0, 0)
    env.goal = (2, 2)
    env.current_position = (0, 0)

    position, reward, done, info = env.step("UP")

    assert position == (0, 0)
    assert reward == -6
    assert done is False
    assert info["collision"] is True
    assert env.count_steps == 1


def test_trap_step():
    env = AdvancedEnvironment(
        size=3,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.0,
    )

    env.grid[:] = 0
    env.start = (0, 0)
    env.goal = (2, 2)
    env.current_position = (1, 1)

    env.grid[1, 2] = 2

    position, reward, done, info = env.step("RIGHT")

    assert position == env.start
    assert reward == -25
    assert done is False
    assert info["trap"] is True
    assert info["collision"] is False
    assert info["mud"] is False


def test_mud_step():
    env = AdvancedEnvironment(
        size=3,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.0,
    )

    env.grid[:] = 0
    env.start = (0, 0)
    env.goal = (2, 2)
    env.current_position = (1, 1)

    env.grid[1, 2] = 3

    position, reward, done, info = env.step("RIGHT")

    assert position == (1, 2)
    assert reward == -4
    assert done is False
    assert info["mud"] is True
    assert info["collision"] is False
    assert info["trap"] is False


def test_reach_goal():
    env = AdvancedEnvironment(
        size=3,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.0,
    )

    env.grid[:] = 0
    env.start = (0, 0)
    env.goal = (1, 2)
    env.current_position = (1, 1)

    position, reward, done, info = env.step("RIGHT")

    assert position == env.goal
    assert reward == 100
    assert done is True
    assert info["collision"] is False
    assert info["trap"] is False
    assert info["mud"] is False


def test_max_steps():
    env = AdvancedEnvironment(
        size=3,
        obstacle_density=0.0,
        trap_density=0.0,
        mud_density=0.0,
    )

    env.grid[:] = 0
    env.start = (0, 0)
    env.goal = (2, 2)
    env.current_position = (1, 1)
    env.count_steps = env.max_steps - 1

    position, reward, done, info = env.step("RIGHT")

    assert position == (1, 2)
    assert reward == -1
    assert done is True
    assert env.count_steps == env.max_steps


def test_invalid_action():
    env = AdvancedEnvironment()

    with pytest.raises(ValueError):
        env.step("JUMP")


def test_reset_episode():
    env = AdvancedEnvironment(seed=4)

    original_grid = env.grid.copy()
    original_start = env.start
    original_goal = env.goal

    env.current_position = (5, 5)
    env.count_steps = 50

    env.reset_episode()

    assert env.current_position == original_start
    assert env.count_steps == 0
    assert np.array_equal(env.grid, original_grid)
    assert env.start == original_start
    assert env.goal == original_goal
