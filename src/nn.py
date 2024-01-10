"""
The Neural Network is used to evaluate the position of the game.
"""

from typing import NewType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange

from c4 import N_COLS, N_ROWS


Policy = NewType("Policy", np.ndarray)
"""Represents a N_COLS-dimensional vector of probabilities."""

Value = NewType("Value", float)
"""
Represents the value of a position, a continuous number in [-1, 1].

1 is a win for the 1 player.
-1 is a win for the -1 player.
0 is a draw.
"""


class ConnectFourNet(pl.LightningModule):
    def __init__(self):
        super(ConnectFourNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        conv_output_size = self._calculate_conv_output_size()

        self.fc_policy = nn.Linear(conv_output_size, N_COLS)  # Policy head
        self.fc_value = nn.Linear(conv_output_size, 1)  # Value head

        # Metrics
        self.policy_kl_div = torchmetrics.KLDivergence(log_prob=True)
        self.value_mse = torchmetrics.MeanSquaredError()

    def forward(self, x):
        x = rearrange(x, "b h w -> b 1 h w")
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = rearrange(x, "b c h w -> b (c h w)")

        policy_logprobs = F.log_softmax(self.fc_policy(x), dim=1)
        value = F.tanh(self.fc_value(x))

        return policy_logprobs, value

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

    def training_step(self, batch, batch_idx):
        # Forward pass
        game_ids, inputs, policy_logprob_targets, value_targets = batch
        policy_logprob, value_pred = self.forward(inputs)
        value_pred = rearrange(value_pred, "b 1 -> b")
        policy_logprob_targets = torch.log(policy_logprob_targets)

        # Losses
        policy_loss = self.policy_kl_div(policy_logprob_targets, policy_logprob)
        value_loss = self.value_mse(value_pred, value_targets)
        loss = policy_loss + value_loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Forward pass
        game_ids, inputs, policy_logprob_targets, value_targets = batch
        policy_logprob, value_pred = self.forward(inputs)
        value_pred = rearrange(value_pred, "b 1 -> b")
        policy_logprob_targets = torch.log(policy_logprob_targets)

        # Losses
        policy_loss = self.policy_kl_div(policy_logprob_targets, policy_logprob)
        value_loss = self.value_mse(value_pred, value_targets)
        loss = policy_loss + value_loss

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_policy_kl_div", policy_loss)
        self.log("val_value_mse", value_loss)
        return loss
