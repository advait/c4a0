from c4 import is_game_over

from self_play import gen_samples
from test_utils import mock_eval_pos


def test_gen_sample():
    poss = list(
        gen_samples(
            p1=mock_eval_pos,
            p2=mock_eval_pos,
            n_games=1,
            mcts_iterations=50,
            exploration_constant=1.4,
        )
    )

    _, pos, _, value = poss[0]

    assert len(poss) > 6
    assert value == -1 or value == 0 or value == 1
    assert is_game_over(pos) is not None
