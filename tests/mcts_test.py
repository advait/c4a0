from textwrap import dedent
from typing import Tuple
import numpy as np
from numpy.testing import assert_array_equal

import pytest

from c4 import N_COLS, STARTING_POS, Pos, pos_from_str
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
    assert np.all(policy != UNIFORM_POLICY)


@pytest.mark.asyncio
async def test_winning_position():
    """
    From a winning position, mcts should end up with a policy that prefers the winning move.
    """
    pos = pos_from_str(
        dedent(
            """
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫🔵🔵🔵⚫⚫⚫
            ⚫🔴🔴🔴⚫⚫⚫
            """
        ).strip()
    )

    policy = await mcts(
        pos=pos,
        n_iterations=100,
        exploration_constant=3.5,
        eval_pos=uniform_eval_pos,
    )
    winning_moves_p = policy[0] + policy[4]
    print(policy)
    assert winning_moves_p > 0.97


@pytest.mark.asyncio
async def test_losing_position():
    """
    From a definitively losing position, mcts should end up with a uniform policy because it's
    desperately trying to find a non-losing move.
    """
    pos = pos_from_str(
        dedent(
            """
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫⚫⚫⚫⚫⚫⚫
            ⚫🔴🔴⚫⚫⚫⚫
            ⚫🔵🔵🔵⚫⚫⚫
            """
        ).strip()
    )

    policy = await mcts(
        pos=pos,
        n_iterations=10000,
        exploration_constant=3.5,
        eval_pos=uniform_eval_pos,
    )
    print(policy)
    assert policy == pytest.approx(UNIFORM_POLICY, abs=0.01)


UNIFORM_POLICY: Policy = np.ones(N_COLS) / N_COLS


async def uniform_eval_pos(pos: Pos) -> Tuple[Policy, Value]:
    """Simple position evaluator that always returns a value of zero and a uniform policy."""
    policy = UNIFORM_POLICY
    value = 0.0
    return policy, value
