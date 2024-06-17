"""
Connect Four game logic.
"""

from enum import Enum
from typing import Optional, NewType

import numpy as np

N_ROWS = 6
N_COLS = 7


class CellValue(Enum):
    """Represents the value of a cell in a connect four position."""

    PLAYER_MOVE = 1
    OPPONENT_MOVE = -1
    EMPTY = 0


class TerminalState(Enum):
    """Represents the possible terminal states of a connect four game."""

    PLAYER_WIN = 1
    OPPONENT_WIN = -1
    DRAW = 0


Pos = NewType("Pos", np.ndarray)
"""Represents a connect four position."""

ColIndex = NewType("ColIndex", int)
"""Represents playing a move in the given column."""

STARTING_POS: Pos = Pos(np.zeros((N_ROWS, N_COLS), dtype=np.float32))
"""The starting position of a connect four game."""

Ply = NewType("Ply", int)
"""The number of moves that have been played. A ply of 0 is the starting position."""


def make_move(pos: Pos, col: ColIndex | int) -> Pos:
    """
    Plays a move in the given column from the perspective of the 1 player.
    Returns a new position where the 1 and -1 values are flipped.
    """
    pos = Pos(np.copy(pos))
    for row in reversed(range(N_ROWS)):
        if pos[row][col] != CellValue.EMPTY.value:
            continue
        pos[row][col] = CellValue.PLAYER_MOVE.value
        pos = Pos(-pos)  # Invert the board
        return pos
    raise IllegalMove(pos, col)


class IllegalMove(Exception):
    """Raised when a move is played in a full column."""

    def __init__(self, pos: Pos, col: int):
        super().__init__(f"Illegal move in column {col} of position {pos}")


def _is_consecutive_four(cells):
    """Check if four cells are the same and not empty."""
    return len(set(cells)) == 1 and cells[0] != CellValue.EMPTY.value


def is_game_over(pos: Pos) -> Optional[TerminalState]:
    """Determines if the game is over, and if so, who won. If the game is not over, returns None."""
    # Check rows, columns, and diagonals for winner
    for row in range(N_ROWS):
        for col in range(N_COLS):
            if col <= N_COLS - 4 and _is_consecutive_four(
                [pos[row][c] for c in range(col, col + 4)]
            ):
                return (
                    TerminalState.PLAYER_WIN
                    if pos[row][col] == CellValue.PLAYER_MOVE.value
                    else TerminalState.OPPONENT_WIN
                )

            if row <= N_ROWS - 4 and _is_consecutive_four(
                [pos[r][col] for r in range(row, row + 4)]
            ):
                return (
                    TerminalState.PLAYER_WIN
                    if pos[row][col] == CellValue.PLAYER_MOVE.value
                    else TerminalState.OPPONENT_WIN
                )

            if (
                row <= N_ROWS - 4
                and col <= N_COLS - 4
                and _is_consecutive_four([pos[row + i][col + i] for i in range(4)])
            ):
                return (
                    TerminalState.PLAYER_WIN
                    if pos[row][col] == CellValue.PLAYER_MOVE.value
                    else TerminalState.OPPONENT_WIN
                )

            if (
                row >= 3
                and col <= N_COLS - 4
                and _is_consecutive_four([pos[row - i][col + i] for i in range(4)])
            ):
                return (
                    TerminalState.PLAYER_WIN
                    if pos[row][col] == CellValue.PLAYER_MOVE.value
                    else TerminalState.OPPONENT_WIN
                )

    # Check for draw
    if np.all(pos != CellValue.EMPTY.value):
        return TerminalState.DRAW

    # Game is not over
    return None


def get_legal_moves(pos: Pos) -> np.ndarray:
    """
    Returns an array indicating which columns are legal to play in (1 is legal, 0 is not).
    This array can be used to mask the policy output of the neural network.
    """
    return (pos[0] == CellValue.EMPTY.value).astype(np.float64)


def get_ply(pos: Pos) -> Ply:
    """
    Returns the ply of the position or the number of moves that have been played.
    Ply of 0 is the starting position.
    """
    return np.sum(pos != CellValue.EMPTY.value)


cell_colors = {
    CellValue.PLAYER_MOVE.value: "🔴",
    CellValue.OPPONENT_MOVE.value: "🔵",
    CellValue.EMPTY.value: "⚫",
}

cell_colors_flipped = {v: k for k, v in cell_colors.items()}


def pos_to_str(pos: Pos, from_p0_perspective: bool = False) -> str:
    """
    Returns a string representation of the position.

    If from_p0_perspective is True, the position colors are such that the first player is red.
    If from_p0_perspective is False, the position colors are such that the current player is red.
    """
    if from_p0_perspective and get_ply(pos) % 2 == 1:
        pos = Pos(-pos)

    return "\n".join(["".join([cell_colors[cell] for cell in row]) for row in pos])


def pos_from_str(s: str) -> Pos:
    """Returns a Pos from a string representation generated by pos_to_str."""
    pos = Pos(np.zeros((N_ROWS, N_COLS), dtype=np.float32))
    for row, line in enumerate(s.split("\n")):
        for col, char in enumerate(line):
            pos[row][col] = cell_colors_flipped[char]
    return pos


def pos_to_bytes(pos: Pos) -> bytes:
    """Serializes a position into bytes. Useful for using positions as dict keys (hashable)."""
    return pos.tobytes()


def pos_from_bytes(b: bytes) -> Pos:
    """Returns a position that was serialized with pos_to_bytes."""
    return Pos(np.frombuffer(b, dtype=np.float32).reshape((N_ROWS, N_COLS)))