"""
The Neural Network is used to evaluate the position of the game.
"""

from typing import Callable, NewType, Tuple

import numpy as np

from c4 import Pos


Policy = NewType("Pos", np.ndarray)
"""Represents a N_COLS-dimensional vector of probabilities."""

Value = NewType("Value", float)
"""
Represents the value of a position, a continuous number in [-1, 1].

1 is a win for the 1 player.
-1 is a win for the -1 player.
0 is a draw.
"""

EvaluatePos = Callable[[Pos], Tuple[Value, Policy]]
"""Function that evaluates a position and returns its value and a policy."""
