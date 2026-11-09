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


@pytest.mark.parametrize("frac", [-0.1, 1.1, float("nan"), float("inf")])
def test_random_starts_reject_invalid_fraction(frac: float) -> None:
    with pytest.raises(ValueError, match="random_start_frac"):
        c4a0_rust.play_games([], 8, 4, 2.0, 0.01, _uniform_eval, random_start_frac=frac)


def test_random_starts_reject_zero_max_ply() -> None:
    with pytest.raises(ValueError, match="random_start_max_ply"):
        c4a0_rust.play_games([], 8, 4, 2.0, 0.01, _uniform_eval, random_start_max_ply=0)


def test_split_train_test_is_deterministic_and_non_mutating() -> None:
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0) for i in range(4)],
        8,
        2,
        1.4,
        0.01,
        _uniform_eval,
        root_noise_alpha=None,
        root_noise_frac=0.0,
        value_target_q_weight=0.0,
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


def test_mcts_policies_leaf_search_proves_forced_win_at_once() -> None:
    # The mover wins in 5 plies with column 1 or 4. A 5-ply leaf search proves the root before
    # any network call and splits the policy over both. Plain MCTS-Solver, with a large budget,
    # proves the same win through one of them and leaves the other unproven.
    calls: list[int] = []

    def counting_eval(
        model_id: int, pos: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        calls.append(pos.shape[0])
        return _uniform_eval(model_id, pos)

    root = ["2453645131230"]
    searched = c4a0_rust.mcts_policies(
        root, 2, 2.0, 0.01, counting_eval, leaf_minimax_depth=5
    )
    assert calls == []
    np.testing.assert_array_equal(searched[0], [0, 0.5, 0, 0, 0.5, 0, 0])
    plain = c4a0_rust.mcts_policies(root, 4096, 2.0, 0.01, _uniform_eval)
    assert set(np.flatnonzero(plain[0])) <= {1, 4}


def test_mcts_policies_default_leaf_depth_is_zero() -> None:
    roots = ["", "33", "2453645131"]
    np.testing.assert_array_equal(
        c4a0_rust.mcts_policies(roots, 32, 2.0, 0.01, _uniform_eval),
        c4a0_rust.mcts_policies(
            roots, 32, 2.0, 0.01, _uniform_eval, leaf_minimax_depth=0
        ),
    )


def test_play_games_with_leaf_search_finishes_games() -> None:
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0) for i in range(4)],
        8,
        8,
        2.0,
        0.01,
        _uniform_eval,
        leaf_minimax_depth=4,
    )
    assert sorted(_game_ids(games)) == [0, 1, 2, 3]
    for result in games.results:
        for sample in result.samples:
            assert sample.to_numpy()[1].sum() == pytest.approx(1.0)


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


@pytest.mark.parametrize(
    "alpha,frac",
    [
        (0.0, 0.25),
        (-1.0, 0.25),
        (float("nan"), 0.25),
        (None, 0.25),
        (0.3, -0.1),
        (0.3, 1.1),
        (0.3, float("nan")),
    ],
)
def test_root_noise_rejects_invalid_parameters(
    alpha: float | None, frac: float
) -> None:
    with pytest.raises(ValueError):
        c4a0_rust.play_games(
            [],
            8,
            32,
            2.0,
            0.01,
            _uniform_eval,
            root_noise_alpha=alpha,
            root_noise_frac=frac,
            value_target_q_weight=0.0,
        )


def test_root_noise_off_preserves_games_and_on_changes_them() -> None:
    def play(alpha: float | None, frac: float) -> dict[int, list[str]]:
        games = c4a0_rust.play_games(
            [c4a0_rust.GameMetadata(i, 0, 0) for i in range(4)],
            8,
            32,
            2.0,
            0.01,
            _uniform_eval,
            root_noise_alpha=alpha,
            root_noise_frac=frac,
            value_target_q_weight=0.0,
        )
        return {g.metadata.game_id: _sample_positions(g.samples) for g in games.results}

    baseline = play(None, 0.0)
    assert play(0.3, 0.0) == baseline
    assert play(0.3, 0.25) != baseline


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf")])
def test_value_target_weight_rejects_invalid_parameters(weight: float) -> None:
    with pytest.raises(ValueError, match="value_target_q_weight"):
        c4a0_rust.play_games(
            [], 8, 16, 2.0, 0.01, _uniform_eval, value_target_q_weight=weight
        )


def test_value_target_blending_preserves_moves_and_terminal_values() -> None:
    def play(weight: float) -> dict[int, c4a0_rust.GameResult]:
        result = c4a0_rust.play_games(
            [c4a0_rust.GameMetadata(i, 0, 0) for i in range(4)],
            8,
            16,
            2.0,
            0.01,
            _uniform_eval,
            value_target_q_weight=weight,
        )
        return {game.metadata.game_id: game for game in result.results}

    legacy, roots, blended = play(0.0), play(1.0), play(0.25)
    changed = False
    for game_id, game in legacy.items():
        for z, q, target in zip(
            game.samples, roots[game_id].samples, blended[game_id].samples, strict=True
        ):
            assert z.pos_str() == q.pos_str() == target.pos_str()
            z_arrays, q_arrays, target_arrays = (
                z.to_numpy(),
                q.to_numpy(),
                target.to_numpy(),
            )
            for i in [0, 1]:
                np.testing.assert_array_equal(z_arrays[i], q_arrays[i])
                np.testing.assert_array_equal(z_arrays[i], target_arrays[i])
            for i in [2, 3]:
                np.testing.assert_array_equal(
                    target_arrays[i],
                    np.float32(0.75) * z_arrays[i] + np.float32(0.25) * q_arrays[i],
                )
                changed |= bool(z_arrays[i] != q_arrays[i])
        for actual in [roots[game_id], blended[game_id]]:
            for i in [2, 3]:
                assert (
                    game.samples[-1].to_numpy()[i].tobytes()
                    == actual.samples[-1].to_numpy()[i].tobytes()
                )
    assert changed


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.8, 1.0])
def test_bulk_samples_equal_legacy_conversion_byte_for_byte(fraction: float) -> None:
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0) for i in range(8)],
        8,
        4,
        2.0,
        0.01,
        _uniform_eval,
        value_target_q_weight=0.6,
        random_start_frac=0.5,
    )
    for arrays, legacy in zip(
        games.split_train_test_numpy(fraction, 1337),
        games.split_train_test(fraction, 1337),
        strict=True,
    ):
        augmented = legacy + [s.flip_h() for s in legacy]
        if not augmented:
            assert [a.shape for a in arrays] == [(0, 2, 6, 7), (0, 7), (0,), (0,)]
            continue
        columns = zip(*(s.to_numpy() for s in augmented), strict=True)
        for array, column in zip(arrays, columns, strict=True):
            expected = np.stack(column)
            assert array.dtype == np.float32
            assert array.shape == expected.shape
            assert array.tobytes() == expected.tobytes()  # Includes signed zero.
    assert len(games) == 8
    combined = c4a0_rust.PlayGamesResult.concat([games, games])
    legacy_combined = games + games
    assert combined.__getstate__() == legacy_combined.__getstate__()


def test_bulk_empty_and_invalid_split() -> None:
    games = c4a0_rust.PlayGamesResult()
    assert games.split_train_test_numpy(0.8, 1337)[0][0].shape == (0, 2, 6, 7)
    for fraction in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError, match="train_frac"):
            games.split_train_test_numpy(fraction, 1337)
