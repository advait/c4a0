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

    def __init__(self, game_id: int = 0, player0_id: int = 0, player1_id: int = 0) -> None: ...


class Sample:
    def flip_h(self) -> Sample: ...
    def to_numpy(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...
    def pos_str(self) -> str: ...


class GameResult:
    metadata: GameMetadata
    samples: list[Sample]

    def player0_score(self) -> float: ...


class PlayGamesResult:
    results: list[GameResult]

    def __init__(self) -> None: ...
    def __add__(self, other: PlayGamesResult) -> PlayGamesResult: ...
    def split_train_test(self, train_frac: float, seed: int) -> tuple[list[Sample], list[Sample]]: ...
    def score_policies(
        self,
        solver_path: str,
        solver_book_path: str,
        solution_cache_path: str,
    ) -> float: ...
    def score_top_moves(
        self,
        solver_path: str,
        solver_book_path: str,
        solution_cache_path: str,
    ) -> float: ...
    def sample_count(self) -> int: ...
    def nonterminal_sample_count(self) -> int: ...
    def unique_positions(self) -> int: ...


def play_games(
    reqs: Sequence[GameMetadata],
    max_nn_batch_size: int,
    n_mcts_iterations: int,
    c_exploration: float,
    c_ply_penalty: float,
    py_eval_pos_cb: Callable[[int, NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]],
) -> PlayGamesResult: ...


def run_tui(
    py_eval_pos_cb: Callable[[int, NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]],
    max_mcts_iters: int,
    c_exploration: float,
    c_ply_penalty: float,
) -> None: ...
