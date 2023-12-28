import numpy as np
from numpy.testing import assert_array_equal

from c4 import N_COLS, STARTING_POS

from mcts import mcts


def test_mcts_depth_one():
    policy = mcts(
        pos=STARTING_POS,
        n_iterations=1 + N_COLS + N_COLS,
        exploration_constant=1.0,
        eval_pos=mock_eval_pos,
    )
    desired_policy = np.ones(N_COLS) / N_COLS
    assert_array_equal(policy, desired_policy)


def test_mcts_depth_two():
    policy = mcts(
        pos=STARTING_POS,
        n_iterations=1 + N_COLS + (N_COLS * N_COLS) + (N_COLS * N_COLS),
        exploration_constant=1.0,
        eval_pos=mock_eval_pos,
    )
    desired_policy = np.ones(N_COLS) / N_COLS
    assert_array_equal(policy, desired_policy)


def test_mcts_depth_uneven():
    """
    Because we have an uneven number of iterations, child nodes will be expanded unevenly resulting
    in a non-uniform policy
    """
    policy = mcts(
        pos=STARTING_POS,
        n_iterations=47,
        exploration_constant=1.0,
        eval_pos=mock_eval_pos,
    )
    uniform_policy = np.ones(N_COLS) / N_COLS
    assert np.all(policy != uniform_policy)


def mock_eval_pos(pos):
    policy = np.ones(N_COLS) / N_COLS
    value = 0.0
    return policy, value
