from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

N_COLS: int
N_ROWS: int
BUF_N_CHANNELS: int

class GameMetadata:
    game_id: int
    player0_id: int
    player1_id: int

    def __init__(
        self, game_id: int = 0, player0_id: int = 0, player1_id: int = 0
    ) -> None: ...

class Sample:
    def flip_h(self) -> Sample: ...
    def to_numpy(
        self,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]: ...
    def pos_str(self) -> str: ...

class GameResult:
    metadata: GameMetadata
    samples: list[Sample]

    def player0_score(self) -> float: ...

SampleArrays = tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]

class PlayGamesResult:
    results: list[GameResult]

    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    @staticmethod
    def concat(games: Sequence[PlayGamesResult]) -> PlayGamesResult: ...
    def __add__(self, other: PlayGamesResult) -> PlayGamesResult: ...
    def split_train_test(
        self, train_frac: float, seed: int
    ) -> tuple[list[Sample], list[Sample]]: ...
    def split_train_test_numpy(
        self, train_frac: float, seed: int
    ) -> tuple[SampleArrays, SampleArrays]:
        """Stacked arrays, original samples followed by mirrors, with the legacy split order."""
        ...
    def unique_positions(self) -> int: ...

EvalPosCb = Callable[
    [int, NDArray[np.float32]],
    tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
]
"""cb(model_id, pos[B,2,6,7]) -> (policy_logprobs[B,7], q_penalty[B], q_no_penalty[B])"""

def play_games(
    reqs: Sequence[GameMetadata],
    max_nn_batch_size: int,
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    py_eval_pos_cb: EvalPosCb,
    root_noise_alpha: float | None = None,
    root_noise_frac: float = 0.0,
    value_target_q_weight: float = 0.0,
    random_start_frac: float = 0.0,
    random_start_max_ply: int = 12,
    leaf_minimax_depth: int = 0,
) -> PlayGamesResult: ...
def encode_positions(moves: Sequence[str]) -> NDArray[np.float32]:
    """Network input (N,2,6,7) for 0-indexed move strings. Raises ValueError on invalid moves."""
    ...

def mcts_policies(
    moves: Sequence[str],
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    py_eval_pos_cb: EvalPosCb,
    leaf_minimax_depth: int = 0,
) -> NDArray[np.float32]:
    """Root visit policies (N,7) after single-threaded lockstep MCTS from non-terminal roots.
    `leaf_minimax_depth` > 0 proves leaves by exact search to that depth before evaluation."""
    ...

def label_positions(
    split: str,
    first_ply: int,
    last_ply: int,
    per_ply: int,
    seed: int,
    book_path: str | None = None,
    threads: int | None = None,
) -> list[tuple[str, list[int | None]]]:
    """Samples positions in `split` and labels them with the native solver (None = full column).
    `book_path` is an opening book from `mise run solver:book`; it only speeds early plies.
    `threads` defaults to every core."""
    ...

def run_tui(
    py_eval_pos_cb: EvalPosCb,
    max_mcts_iters: int,
    c_exploration: float,
    c_ply_penalty: float,
) -> None: ...
