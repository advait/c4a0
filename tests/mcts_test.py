from typing import Tuple
import numpy as np
from numpy.testing import assert_array_equal

import pytest

from c4 import N_COLS, STARTING_POS, Pos
from mcts import mcts
from nn import Policy, Value


@pytest.mark.asyncio
async def test_mcts_depth_one():
    policy = await mcts(
        pos=STARTING_POS,
        n_iterations=1 + N_COLS + N_COLS,
        exploration_constant=1.0,
        eval_pos=uniform_eval_pos,
    )
    desired_policy = np.ones(N_COLS) / N_COLS
    assert_array_equal(policy, desired_policy)


@pytest.mark.asyncio
async def test_mcts_depth_two():
    policy = await mcts(
        pos=STARTING_POS,
        n_iterations=1 + N_COLS + (N_COLS * N_COLS) + (N_COLS * N_COLS),
        exploration_constant=1.0,
        eval_pos=uniform_eval_pos,
    )
    desired_policy = np.ones(N_COLS) / N_COLS
    assert_array_equal(policy, desired_policy)


@pytest.mark.asyncio
async def test_mcts_depth_uneven():
    """
    Because we have an uneven number of iterations, child nodes will be expanded unevenly resulting
    in a non-uniform policy
    """
    policy = await mcts(
        pos=STARTING_POS,
        n_iterations=47,
        exploration_constant=1.0,
        eval_pos=uniform_eval_pos,
    )
    uniform_policy = np.ones(N_COLS) / N_COLS
    assert np.all(policy != uniform_policy)


async def uniform_eval_pos(pos: Pos) -> Tuple[Policy, Value]:
    """Simple position evaluator that always returns a value of zero and a uniform policy."""
    policy = np.ones(N_COLS) / N_COLS
    value = 0.0
    return policy, value
