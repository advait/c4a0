import pytest
import numpy as np
from c4 import (
    make_move,
    is_game_over,
    N_ROWS,
    N_COLS,
    STARTING_POS,
    CellValue,
    TerminalState,
    IllegalMove,
)


def test_initial_position():
    assert np.array_equal(STARTING_POS, np.zeros((N_ROWS, N_COLS), dtype=np.int8))


def test_legal_move():
    pos = make_move(STARTING_POS, 0)
    # Because the board is inverted, the last move should be the opponent's
    assert pos[-1][0] == CellValue.OPPONENT_MOVE.value
    assert pos[-1][1] == CellValue.EMPTY.value


def test_illegal_move():
    pos = STARTING_POS
    for _ in range(N_ROWS):
        pos = make_move(pos, 0)

    with pytest.raises(IllegalMove):
        make_move(pos, 0)


@pytest.mark.parametrize("col", range(N_COLS))
def test_alternating_players(col):
    pos = make_move(STARTING_POS, col)
    assert pos[-1][col] == CellValue.OPPONENT_MOVE.value
    pos = make_move(pos, col)
    assert pos[-1][col] == CellValue.PLAYER_MOVE.value
    assert pos[-2][col] == CellValue.OPPONENT_MOVE.value


def test_player_win_horizontal():
    pos = STARTING_POS
    for i in range(3):
        pos = make_move(pos, i)  # Player
        pos = make_move(pos, i + 4)  # Opponent

    pos = make_move(pos, 3)  # Winning move for Player
    # Because the board is inverted, the last move results in the opponent winning
    assert is_game_over(pos) == TerminalState.OPPONENT_WIN


def test_opponent_win_vertical():
    pos = STARTING_POS
    for _ in range(3):
        pos = make_move(pos, 0)  # Player
        pos = make_move(pos, 1)  # Opponent

    # Last move for the win
    pos = make_move(pos, 3)  # Player
    pos = make_move(pos, 1)  # Opponent
    assert is_game_over(pos) == TerminalState.OPPONENT_WIN


def test_draw():
    pos = STARTING_POS

    # For the first three rows, alternate moves in 0-5 columns

    # First row
    pos = make_move(pos, 0)  # Player
    pos = make_move(pos, 1)  # Opponent
    pos = make_move(pos, 2)  # Player
    pos = make_move(pos, 3)  # Opponent
    pos = make_move(pos, 4)  # Player
    pos = make_move(pos, 5)  # Opponent

    # Second row
    pos = make_move(pos, 0)  # Player
    pos = make_move(pos, 1)  # Opponent
    pos = make_move(pos, 2)  # Player
    pos = make_move(pos, 3)  # Opponent
    pos = make_move(pos, 4)  # Player
    pos = make_move(pos, 5)  # Opponent

    # Third row
    pos = make_move(pos, 0)  # Player
    pos = make_move(pos, 1)  # Opponent
    pos = make_move(pos, 2)  # Player
    pos = make_move(pos, 3)  # Opponent
    pos = make_move(pos, 4)  # Player
    pos = make_move(pos, 5)  # Opponent

    # For the remaining rows, reverse the pattern
    # Fourth row
    pos = make_move(pos, 5)  # Player
    pos = make_move(pos, 4)  # Opponent
    pos = make_move(pos, 3)  # Player
    pos = make_move(pos, 2)  # Opponent
    pos = make_move(pos, 1)  # Player
    pos = make_move(pos, 0)  # Opponent

    # Fifth row
    pos = make_move(pos, 5)  # Player
    pos = make_move(pos, 4)  # Opponent
    pos = make_move(pos, 3)  # Player
    pos = make_move(pos, 2)  # Opponent
    pos = make_move(pos, 1)  # Player
    pos = make_move(pos, 0)  # Opponent

    # Sixth row
    pos = make_move(pos, 5)  # Player
    pos = make_move(pos, 4)  # Opponent
    pos = make_move(pos, 3)  # Player
    pos = make_move(pos, 2)  # Opponent
    pos = make_move(pos, 1)  # Player
    pos = make_move(pos, 0)  # Opponent

    # Now the first 0-5 columns are full. Simply fill up the final column with alternates.
    pos = make_move(pos, 6)  # Player
    pos = make_move(pos, 6)  # Opponent
    pos = make_move(pos, 6)  # Player
    pos = make_move(pos, 6)  # Opponent
    pos = make_move(pos, 6)  # Player
    pos = make_move(pos, 6)  # Opponent

    assert is_game_over(pos) == TerminalState.DRAW

    # The next move should raise an IllegalMove exception
    with pytest.raises(IllegalMove):
        make_move(pos, 6)
