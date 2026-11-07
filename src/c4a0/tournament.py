"""Baseline players for the interactive TUI."""

from typing import NewType

import numpy as np
import torch
from numpy.typing import NDArray

from c4a0_rust import N_COLS

ModelID = NewType("ModelID", int)


class RandomPlayer:
    """Player that provides a random policy and value."""

    def __init__(self, model_id: ModelID) -> None:
        self.model_id = model_id

    def forward_numpy(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        batch_size = x.shape[0]
        policy_logits = torch.rand(batch_size, N_COLS).numpy()
        q_value = (torch.rand(batch_size) * 2 - 1).numpy()  # [-1, 1]
        return policy_logits, q_value, q_value


class UniformPlayer:
    """Player that provides a uniform policy and 0 value."""

    def __init__(self, model_id: ModelID) -> None:
        self.model_id = model_id

    def forward_numpy(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        batch_size = x.shape[0]
        policy_logits = torch.ones(batch_size, N_COLS).numpy()
        q_value = torch.zeros(batch_size).numpy()
        return policy_logits, q_value, q_value
