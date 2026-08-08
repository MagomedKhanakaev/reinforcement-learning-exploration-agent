from agents.q_learning_agent import QLearningAgent


def test_default_parameters():
    agent = QLearningAgent()

    assert agent.alpha == 0.1
    assert agent.gamma == 0.95
    assert agent.epsilon == 1.0


def test_new_state_is_added_to_q_table():
    agent = QLearningAgent()

    state = (1, 2)

    agent.verify_state_exists(state)

    assert state in agent.q_table
    assert agent.q_table[state] == {
        "UP": 0.0,
        "DOWN": 0.0,
        "RIGHT": 0.0,
        "LEFT": 0.0,
    }


def test_existing_state_is_not_reset():
    agent = QLearningAgent()

    state = (1, 2)

    agent.verify_state_exists(state)
    agent.q_table[state]["RIGHT"] = 5.0

    agent.verify_state_exists(state)

    assert agent.q_table[state]["RIGHT"] == 5.0


def test_choose_action_returns_valid_action():
    agent = QLearningAgent()

    action = agent.choose_action((0, 0))

    assert action in agent.actions


def test_choose_best_action_with_zero_epsilon():
    agent = QLearningAgent(epsilon=0.0)

    state = (0, 0)

    agent.q_table[state] = {
        "UP": 1.0,
        "DOWN": 2.0,
        "RIGHT": 10.0,
        "LEFT": 3.0,
    }

    action = agent.choose_action(state)

    assert action == "RIGHT"


def test_choose_action_among_tied_best_actions():
    agent = QLearningAgent(epsilon=0.0)

    state = (0, 0)

    agent.q_table[state] = {
        "UP": 5.0,
        "DOWN": 1.0,
        "RIGHT": 5.0,
        "LEFT": 2.0,
    }

    actions = {agent.choose_action(state) for _ in range(100)}

    assert actions.issubset({"UP", "RIGHT"})
    assert len(actions) == 2


def test_full_exploration_can_choose_different_actions():
    agent = QLearningAgent()

    state = (0, 0)

    actions = {agent.choose_action(state) for _ in range(100)}

    assert len(actions) > 1


def test_q_table_update_non_terminal():
    agent = QLearningAgent(
        alpha=0.5,
        gamma=0.9,
    )

    state = (0, 0)
    next_state = (0, 1)

    agent.verify_state_exists(state)
    agent.verify_state_exists(next_state)

    agent.q_table[state]["RIGHT"] = 2.0

    agent.q_table[next_state] = {
        "UP": 4.0,
        "DOWN": 6.0,
        "RIGHT": 3.0,
        "LEFT": 1.0,
    }

    agent.update_q_table(
        state=state,
        action="RIGHT",
        reward=5,
        next_state=next_state,
        done=False,
    )

    expected_target = 5 + 0.9 * 6
    expected_q = (1 - 0.5) * 2 + 0.5 * expected_target

    assert agent.q_table[state]["RIGHT"] == expected_q


def test_q_table_update_terminal():
    agent = QLearningAgent(alpha=0.5)

    state = (0, 0)
    next_state = (0, 1)

    agent.verify_state_exists(state)
    agent.q_table[state]["RIGHT"] = 2.0

    agent.update_q_table(
        state=state,
        action="RIGHT",
        reward=100,
        next_state=next_state,
        done=True,
    )

    expected_q = (1 - 0.5) * 2 + 0.5 * 100

    assert agent.q_table[state]["RIGHT"] == expected_q


def test_terminal_update_does_not_create_next_state():
    agent = QLearningAgent()

    state = (0, 0)
    next_state = (0, 1)

    agent.verify_state_exists(state)

    agent.update_q_table(
        state=state,
        action="RIGHT",
        reward=100,
        next_state=next_state,
        done=True,
    )

    assert next_state not in agent.q_table


def test_update_only_changes_selected_action():
    agent = QLearningAgent(alpha=0.5)

    state = (0, 0)
    next_state = (0, 1)

    agent.verify_state_exists(state)
    agent.verify_state_exists(next_state)

    before = agent.q_table[state].copy()

    agent.update_q_table(
        state=state,
        action="RIGHT",
        reward=5,
        next_state=next_state,
        done=False,
    )

    assert agent.q_table[state]["UP"] == before["UP"]
    assert agent.q_table[state]["DOWN"] == before["DOWN"]
    assert agent.q_table[state]["LEFT"] == before["LEFT"]


def test_decay_epsilon():
    agent = QLearningAgent()

    agent.decay_epsilon(epsilon_min=0.05, epsilon_decay=0.9)

    assert agent.epsilon == 0.9


def test_epsilon_does_not_go_below_minimum():
    agent = QLearningAgent(epsilon=0.051)

    agent.decay_epsilon(epsilon_min=0.05, epsilon_decay=0.5)

    assert agent.epsilon == 0.05
