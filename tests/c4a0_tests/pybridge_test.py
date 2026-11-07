from collections.abc import Sequence
from pathlib import Path
from typing import Never

import numpy as np
import pytest
from numpy.typing import NDArray

import c4a0_rust
from c4a0_rust import N_COLS

SOLVER = Path(__file__).resolve().parents[2] / "solver" / "c4solver"
BOOK = SOLVER.with_name("7x6.book")


def _uniform_eval(
    _model_id: int, pos: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    batch_size = pos.shape[0]
    policy_logits = np.zeros((batch_size, N_COLS), dtype=np.float32)
    q_value = np.zeros((batch_size,), dtype=np.float32)
    return policy_logits, q_value, q_value


def _game_ids(games: c4a0_rust.PlayGamesResult) -> list[int]:
    return [result.metadata.game_id for result in games.results]


def _sample_positions(samples: Sequence[c4a0_rust.Sample]) -> list[str]:
    return [sample.pos_str() for sample in samples]


def test_split_train_test_is_deterministic_and_non_mutating() -> None:
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0) for i in range(4)],
        8,
        2,
        1.4,
        0.01,
        _uniform_eval,
        greedy_after_ply=None,
    )
    original_ids = _game_ids(games)

    first_train, first_test = games.split_train_test(0.5, 1337)
    assert _game_ids(games) == original_ids

    second_train, second_test = games.split_train_test(0.5, 1337)
    assert _game_ids(games) == original_ids
    assert _sample_positions(first_train) == _sample_positions(second_train)
    assert _sample_positions(first_test) == _sample_positions(second_test)


def test_encode_positions_puts_mover_in_channel_0() -> None:
    enc = c4a0_rust.encode_positions(["", "3", "33"])
    assert enc.shape == (3, 2, 6, 7) and enc.dtype == np.float32
    assert enc[0].sum() == 0
    # After one move in column 3, that stone belongs to the opponent of the player to move.
    assert enc[1, 1, 0, 3] == 1 and enc[1].sum() == 1
    assert enc[2, 0, 0, 3] == 1 and enc[2, 1, 1, 3] == 1 and enc[2].sum() == 2


@pytest.mark.parametrize("bad", ["7", "a", "0000000", "01010102"])
def test_encode_positions_rejects_invalid_moves(bad: str) -> None:
    with pytest.raises(ValueError):
        c4a0_rust.encode_positions([bad])


def test_mcts_policies_prefers_immediate_win() -> None:
    # Player to move has stones in columns 1-3 on the bottom row: 0 and 4 win.
    policies = c4a0_rust.mcts_policies(["112233"], 256, 4.0, 0.01, _uniform_eval)
    assert policies.shape == (1, N_COLS)
    assert policies[0].argmax() in (0, 4)
    assert policies[0, 0] + policies[0, 4] > 0.8


def test_mcts_policies_is_deterministic_and_batch_independent() -> None:
    roots = ["", "3", "33", "24", "42", "3"]
    batched = c4a0_rust.mcts_policies(roots, 32, 4.0, 0.01, _uniform_eval)
    singles = np.concatenate(
        [c4a0_rust.mcts_policies([r], 32, 4.0, 0.01, _uniform_eval) for r in roots]
    )
    np.testing.assert_array_equal(batched, singles)


def test_mcts_policies_rejects_finished_games() -> None:
    with pytest.raises(ValueError):
        c4a0_rust.mcts_policies(["0101010"], 8, 4.0, 0.01, _uniform_eval)


def test_mcts_policies_raises_callback_errors() -> None:
    def failing_eval(_model_id: int, _pos: NDArray[np.float32]) -> Never:
        raise KeyError("boom")

    with pytest.raises(KeyError):
        c4a0_rust.mcts_policies(["3"], 8, 4.0, 0.01, failing_eval)


def test_mcts_policies_rejects_wrong_callback_shapes() -> None:
    def short_eval(
        model_id: int, pos: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        policy, q, _ = _uniform_eval(model_id, pos)
        return policy[:, :6].copy(), q, q

    with pytest.raises(ValueError):
        c4a0_rust.mcts_policies(["3"], 8, 4.0, 0.01, short_eval)


@pytest.mark.skipif(not SOLVER.is_file(), reason="solver not built in solver/")
def test_label_positions_rows_are_in_split_and_solved() -> None:
    rows = c4a0_rust.label_positions("dev", 10, 12, 4, 7, str(SOLVER), str(BOOK), 2)
    assert [len(m) for m, _ in rows] == [10] * 4 + [11] * 4 + [12] * 4
    for _moves, scores in rows:
        assert len(scores) == N_COLS
        assert any(s is not None for s in scores)


def test_label_positions_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        c4a0_rust.label_positions("val", 0, 1, 1, 0, "x", "y", 1)


def test_nan_value_from_callback_raises_instead_of_panicking() -> None:
    import numpy as np
    import pytest

    def nan_cb(
        _model_id: int, pos: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        n = len(pos)
        return (
            np.zeros((n, 7), dtype=np.float32),
            np.full(n, np.nan, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )

    with pytest.raises(ValueError, match="non-finite"):
        c4a0_rust.mcts_policies(["3"], 4, 1.0, 0.01, nan_cb)


def test_greedy_cutoff_plays_most_visited_move() -> None:
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0) for i in range(4)],
        8,
        32,
        2.0,
        0.01,
        _uniform_eval,
        greedy_after_ply=4,
    )
    for game in games.results:
        for ply, (sample, following) in enumerate(
            zip(game.samples, game.samples[1:], strict=False)
        ):
            if ply < 4:
                continue
            pos, policy, _, _ = sample.to_numpy()
            next_pos, _, _, _ = following.to_numpy()
            col = (next_pos.sum(axis=(0, 1)) - pos.sum(axis=(0, 1))).argmax()
            assert policy[col] == policy.max()
