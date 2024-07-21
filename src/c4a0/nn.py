from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange

from c4a0_rust import N_COLS, N_ROWS  # type: ignore


class ConnectFourNet(pl.LightningModule):
    EPS = 1e-8  # Epsilon small constant to avoid log(0)

    def __init__(
        self,
        n_conv_layers: int = 3,
        conv_filter_size: int = 64,
        n_policy_layers: int = 3,
        n_value_layers: int = 3,
        learning_rate: float = 0.001,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        2 if i == 0 else conv_filter_size,
                        conv_filter_size,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(conv_filter_size),
                    nn.ReLU(),
                )
                for i in range(n_conv_layers)
            ]
        )

        fc_size = self._calculate_conv_output_size()

        self.fc_policy = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(fc_size, fc_size),
                    nn.BatchNorm1d(fc_size),
                    nn.ReLU(),
                )
                for _ in range(n_policy_layers - 1)
            ],
            nn.Linear(fc_size, N_COLS),
            nn.LogSoftmax(dim=1),
        )

        self.fc_value = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(fc_size, fc_size),
                    nn.BatchNorm1d(fc_size),
                    nn.ReLU(),
                )
                for _ in range(n_value_layers - 1)
            ],
            nn.Linear(fc_size, 1),
            nn.Tanh(),
        )

        # Metrics
        self.policy_kl_div = torchmetrics.KLDivergence(log_prob=True)
        self.value_mse = torchmetrics.MeanSquaredError()

        self.save_hyperparameters()

    def forward(self, x):
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        policy_logprobs = self.fc_policy(x)
        value = self.fc_value(x).squeeze(1)
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
        """Helper function to calculate the output size of the convolutional block."""
        # Apply the convolutional layers to a dummy input
        dummy_input = torch.zeros(1, 2, N_ROWS, N_COLS)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
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
        pos, policy_targets, value_targets = batch
        policy_logprob, value_pred = self.forward(pos)
        policy_logprob_targets = torch.log(policy_targets + self.EPS)

        # Losses
        policy_loss = self.policy_kl_div(policy_logprob_targets, policy_logprob)
        value_loss = self.value_mse(value_pred, value_targets)
        loss = 6 * policy_loss + value_loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        self.log(f"{log_prefix}_policy_kl_div", policy_loss)
        self.log(f"{log_prefix}_value_mse", value_loss)
        return loss
