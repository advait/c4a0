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


def test_split_train_test_is_deterministic_and_non_mutating():
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0) for i in range(4)],
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
