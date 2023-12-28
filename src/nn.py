"""
The Neural Network is used to evaluate the position of the game.
"""

from typing import Callable, NewType, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange

from c4 import N_COLS, N_ROWS, Pos


Policy = NewType("Pos", np.ndarray)
"""Represents a N_COLS-dimensional vector of probabilities."""

Value = NewType("Value", float)
"""
Represents the value of a position, a continuous number in [-1, 1].

1 is a win for the 1 player.
-1 is a win for the -1 player.
0 is a draw.
"""

EvaluatePos = Callable[[Pos], Tuple[Policy, Value]]
"""Function that evaluates a position and returns its value and a policy."""


class ConnectFourNet(pl.LightningModule):
    def __init__(self):
        super(ConnectFourNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        conv_output_size = self._calculate_conv_output_size()

        self.fc_policy = nn.Linear(conv_output_size, N_COLS)  # Policy head
        self.fc_value = nn.Linear(conv_output_size, 1)  # Value head

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = rearrange(x, "b h w -> b 1 h w")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        x = rearrange(x, "b c h w -> b (c h w)")

        policy = self.softmax(self.fc_policy(x))
        value = self.tanh(self.fc_value(x))

        return policy, value

    def _calculate_conv_output_size(self):
        """Helper function to calculate the output size of the last convolutional layer."""
        # Apply the convolutional layers to a dummy input
        dummy_input = torch.zeros(1, 1, N_ROWS, N_COLS)
        with torch.no_grad():
            dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))
        return int(torch.numel(dummy_output) / dummy_output.shape[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def single(self, pos: Pos) -> Tuple[Policy, Value]:
        """Helper function to evaluate a single position (unbatched)."""
        pos = torch.from_numpy(pos).float()
        x = rearrange(pos, "h w -> 1 h w")
        policy, value = self(x)
        policy = rearrange(policy, "1 c -> c")
        value = rearrange(value, "1 v -> v")
        return policy, value
