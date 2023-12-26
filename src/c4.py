"""
Connect Four game logic.
"""

from enum import Enum
from typing import NewType

import numpy as np

N_ROWS = 6
N_COLS = 7

class CellValue(Enum):
    PLAYER_MOVE = 1
    OPPONENT_MOVE = -1
    EMPTY = 0

Pos = NewType('Pos', np.ndarray)
"""Represents a connect four position."""

ColIndex = NewType('ColIndex', int)
"""Represents playing a move in the given column."""

STARTING_POS: Pos = np.zeros((N_ROWS, N_COLS), dtype=np.int8)

def play_move(pos: Pos, col: ColIndex) -> Pos:
    """
    Plays a move in the given column from the perspective of the 1 player.
    Returns a new position where the 1 and -1 values are flipped.
    """
    pos = np.copy(pos)
    for row in reversed(range(N_ROWS)):
        if pos[row][col] != CellValue.EMPTY:
            continue
        pos[row][col] = CellValue.PLAYER_MOVE
        pos = -pos  # Invert the board
        return pos
    raise IllegalMove(pos, col)
    
class IllegalMove(Exception):
    """Raised when a move is played in a full column."""

    def __init__(self, pos: Pos, col: int):
        super().__init__(f"Illegal move in column {col} of position {pos}")
