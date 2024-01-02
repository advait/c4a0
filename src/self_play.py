"""
Generating training data via self-play
"""

from typing import Iterable, List, NewType, Optional, Tuple

import numpy as np

from c4 import N_COLS, STARTING_POS, Pos, get_ply, is_game_over, make_move
from mcts import mcts
from nn import EvaluatePos, Policy, Value

NetworkID = NewType("NetworkID", int)
"""Which network generation this sample is from.""" ""

GameID = NewType("GameID", int)
"""Which game this sample is from."""

Sample = NewType("Sample", Tuple[GameID, Pos, Policy, Value])
"""A sample of training data representing a position and the desired policy and value outputs."""


def gen_samples(
    eval1: EvaluatePos,
    n_games: int,
    mcts_iterations: int,
    exploration_constant: float,
    eval2: Optional[EvaluatePos] = None,
) -> Iterable[Sample]:
    """Generates a seqence of Samples for n_games number of games."""
    if eval2 is None:
        eval2 = eval1

    for game_id in range(n_games):
        yield from _gen_game(
            eval1,
            eval2,
            game_id,
            mcts_iterations,
            exploration_constant,
        )


def _gen_game(
    p1: EvaluatePos,
    p2: EvaluatePos,
    game_id: GameID,
    mcts_iterations: int,
    exploration_constant: float,
) -> Iterable[Sample]:
    """
    Generates a single game of self-play, yielding a move at a time.
    """
    results: List[Tuple[GameID, Pos, Policy]] = []

    # Generate moves until terminal state
    pos = STARTING_POS
    while (res := is_game_over(pos)) is None:
        f = p1 if get_ply(pos) % 2 == 0 else p2
        policy = mcts(
            pos,
            eval_pos=f,
            n_iterations=mcts_iterations,
            exploration_constant=exploration_constant,
        )
        results.append((game_id, pos, policy))
        move = np.random.choice(range(len(policy)), p=policy)
        pos = make_move(pos, move)

    # Because there isn't a valid policy a terminal state, we simply use a uniform policy
    final_policy = np.ones(N_COLS) / N_COLS
    results.append((game_id, pos, final_policy))

    final_value = res.value
    if len(results) % 2 == 0:
        # If there are an even number of positions, the final value is flipped for the first move
        final_value *= -1

    for t in results:
        yield t + (final_value,)
        # Alternate the sign of the final value as the board perspective changes between each move
        final_value *= -1
