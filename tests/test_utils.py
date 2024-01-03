from typing import Tuple
import numpy as np

from c4 import N_COLS, Pos
from nn import Policy, Value


async def uniform_eval_pos(pos: Pos) -> Tuple[Policy, Value]:
    """Simple position evaluator that always returns a value of zero and a uniform policy."""
    policy = np.ones(N_COLS) / N_COLS
    value = 0.0
    return policy, value
