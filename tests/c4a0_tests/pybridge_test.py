import numpy as np

import c4a0_rust
from c4a0_rust import N_COLS


def _uniform_eval(_model_id, pos):
    batch_size = pos.shape[0]
    policy_logits = np.zeros((batch_size, N_COLS), dtype=np.float32)
    q_value = np.zeros((batch_size,), dtype=np.float32)
    return policy_logits, q_value, q_value


def _game_ids(games):
    return [result.metadata.game_id for result in games.results]


def _sample_positions(samples):
    return [sample.pos_str() for sample in samples]


def test_play_games_can_start_from_opening_prefix():
    default_games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(0, 0, 0)],
        8,
        2,
        1.4,
        0.01,
        _uniform_eval,
    )
    opened_games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(0, 0, 0, [3, 3])],
        8,
        2,
        1.4,
        0.01,
        _uniform_eval,
    )

    assert opened_games.results[0].metadata.initial_moves == [3, 3]
    assert (
        opened_games.results[0].samples[0].pos_str()
        != default_games.results[0].samples[0].pos_str()
    )


def test_initial_sample_counts_score_one_root_position_per_game():
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0, [i % N_COLS]) for i in range(4)],
        8,
        2,
        1.4,
        0.01,
        _uniform_eval,
    )

    assert games.initial_nonterminal_sample_count() == 4
    assert games.initial_unique_positions() == 4
    assert games.nonterminal_sample_count() >= games.initial_nonterminal_sample_count()
    assert games.unique_positions() >= games.initial_unique_positions()


def test_split_train_test_is_deterministic_and_non_mutating():
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0, [i % N_COLS]) for i in range(4)],
        8,
        2,
        1.4,
        0.01,
        _uniform_eval,
    )
    original_ids = _game_ids(games)

    first_train, first_test = games.split_train_test(0.5, 1337)
    assert _game_ids(games) == original_ids

    second_train, second_test = games.split_train_test(0.5, 1337)
    assert _game_ids(games) == original_ids
    assert _sample_positions(first_train) == _sample_positions(second_train)
    assert _sample_positions(first_test) == _sample_positions(second_test)
