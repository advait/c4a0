from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange

from c4a0_rust import N_COLS, N_ROWS  # type: ignore


class ConnectFourNet(pl.LightningModule):
    EPS = 1e-8  # Epsilon small constant to avoid log(0)

    def __init__(self):
        super(ConnectFourNet, self).__init__()

        # Shared conv blocks
        self.conv1 = nn.Conv2d(2, 16, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        fc_size = self._calculate_conv_output_size()

        # Policy head
        self.fc_policy1 = nn.Linear(fc_size, fc_size)
        self.bn_policy1 = nn.BatchNorm1d(fc_size)
        self.fc_policy2 = nn.Linear(fc_size, fc_size)
        self.bn_policy2 = nn.BatchNorm1d(fc_size)
        self.fc_policy3 = nn.Linear(fc_size, N_COLS)

        # Value head
        self.fc_value1 = nn.Linear(fc_size, fc_size)
        self.bn_value1 = nn.BatchNorm1d(fc_size)
        self.fc_value2 = nn.Linear(fc_size, fc_size)
        self.bn_value2 = nn.BatchNorm1d(fc_size)
        self.fc_value3 = nn.Linear(fc_size, 1)

        # Metrics
        self.policy_kl_div = torchmetrics.KLDivergence(log_prob=True)
        self.value_mse = torchmetrics.MeanSquaredError()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = rearrange(x, "b c h w -> b (c h w)")

        x_p = F.relu(self.bn_policy1(self.fc_policy1(x)))
        x_p = F.relu(self.bn_policy2(self.fc_policy2(x_p)))
        x_p = self.fc_policy3(x_p)
        policy_logprobs = F.log_softmax(x_p, dim=1)

        x_v = F.relu(self.bn_value1(self.fc_value1(x)))
        x_v = F.relu(self.bn_value2(self.fc_value2(x_v)))
        x_v = self.fc_value3(x_v)
        x_v = F.tanh(x_v)
        value = x_v.squeeze(1)

        return policy_logprobs, value

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with input/output as numpy. Model is run in inference mode. Used for self play.
        """
        self.eval()
        pos = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            policy, value = self.forward(pos)
        policy = policy.cpu().numpy()
        value = value.cpu().numpy()
        return policy, value

    def _calculate_conv_output_size(self):
        """Helper function to calculate the output size of the last convolutional layer."""
        # Apply the convolutional layers to a dummy input
        dummy_input = torch.zeros(1, 2, N_ROWS, N_COLS)
        with torch.no_grad():
            dummy_output = self.conv3(self.conv2(self.conv1(dummy_input)))
        return int(torch.numel(dummy_output) / dummy_output.shape[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, log_prefix="val")

    def step(self, batch, log_prefix):
        # Forward pass
        game_ids, inputs, policy_targets, value_targets = batch
        policy_logprob, value_pred = self.forward(inputs)
        value_pred = rearrange(value_pred, "b 1 -> b")
        policy_logprob_targets = torch.log(policy_targets + self.EPS)

        # Losses
        policy_loss = self.policy_kl_div(policy_logprob_targets, policy_logprob)
        value_loss = self.value_mse(value_pred, value_targets)
        loss = 6 * policy_loss + value_loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        self.log(f"{log_prefix}_policy_kl_div", policy_loss)
        self.log(f"{log_prefix}_value_mse", value_loss)
        return loss
