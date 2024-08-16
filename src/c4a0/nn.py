from typing import Dict, Tuple

from loguru import logger
import numpy as np
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange

from c4a0_rust import N_COLS, N_ROWS  # type: ignore


class ModelConfig(BaseModel):
    """Configuration for ConnectFourNet."""

    n_residual_blocks: int
    """The number of residual blocks."""

    conv_filter_size: int
    """The number of filters for the conv layers in the residual blocks."""

    n_policy_layers: int
    """Number of fully connected layers in the policy head."""

    n_value_layers: int
    """Number of fully connected layers in the value head."""

    lr_schedule: Dict[int, float]
    """
    Learning rate schedule. The first item in the tuple indicates which gen_n the learning rate
    begins to be effective for. The second item in the tuple is the learning rate.
    """

    l2_reg: float
    """L2 weight decay regularization for the optimizer"""


class ConnectFourNet(pl.LightningModule):
    """
    A CNN that takes in as input connect four positions and outputs a policy (in logprob space)
    and two Q Values. The policy is a (log) probability distribution over moves. q_penalty
    represents the predicted lucrativeness of the position between [-1, +1] where -1 is a
    definitively losing position and +1 is a definitively winning position where there is a
    penalty applied based on the number of plys (to encourage faster wins). q_no_penalty is the
    lucrativeness without the ply penalty.

    The outputs of this network are used to guide MCTS.
    The outputs of MCTS are used to train the next network.

    The network consists of a sequence of ResidualBlocks (CNN + CNN + BatchNormalization + Relu)
    followed by separate fully connected policy and value heads.
    """

    EPS = 1e-8  # Epsilon small constant to avoid log(0)

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lr_schedule = config.lr_schedule
        self.l2_reg = config.l2_reg

        self.conv = nn.Sequential(
            nn.Conv2d(2, config.conv_filter_size, kernel_size=3, padding=1),
            *[
                ResidualBlock(config.conv_filter_size)
                for i in range(config.n_residual_blocks)
            ],
        )

        fc_size = self._calculate_conv_output_size()

        # Policy head
        self.fc_policy = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(fc_size, fc_size),
                    nn.BatchNorm1d(fc_size),
                    nn.ReLU(),
                )
                for _ in range(config.n_policy_layers - 1)
            ],
            nn.Linear(fc_size, N_COLS),
            nn.LogSoftmax(dim=1),
        )

        # Q Value head, one output dim for q_penalty and another for q_no_penalty
        self.fc_value = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(fc_size, fc_size),
                    nn.BatchNorm1d(fc_size),
                    nn.ReLU(),
                )
                for _ in range(config.n_value_layers - 1)
            ],
            nn.Linear(fc_size, 2),
            nn.Tanh(),
        )

        # Metrics
        self.policy_kl_div = torchmetrics.KLDivergence(log_prob=True)
        self.q_penalty_mse = torchmetrics.MeanSquaredError()
        self.q_no_penalty_mse = torchmetrics.MeanSquaredError()

        self.save_hyperparameters(config.model_dump())

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        policy_logprobs = self.fc_policy(x)
        q_values = self.fc_value(x)  # b 2
        q_penalty, q_no_penalty = q_values.split(1, dim=1)  # both: b 1
        q_penalty = q_penalty.squeeze(1)  # b
        q_no_penalty = q_no_penalty.squeeze(1)  # b
        return policy_logprobs, q_penalty, q_no_penalty

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass with input/output as numpy. Model is run in inference mode. Used for self play.
        """
        self.eval()
        pos = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            policy, q_penalty, q_no_penalty = self.forward(pos)
        policy = policy.cpu().numpy()
        q_penalty = q_penalty.cpu().numpy()
        q_no_penalty = q_no_penalty.cpu().numpy()
        return policy, q_penalty, q_no_penalty

    def _calculate_conv_output_size(self):
        """Helper function to calculate the output size of the convolutional block."""
        # Apply the convolutional layers to a dummy input
        dummy_input = torch.zeros(1, 2, N_ROWS, N_COLS)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
        return int(torch.numel(dummy_output))

    def configure_optimizers(self):
        gen_n: int = self.trainer.gen_n  # type: ignore
        assert gen_n is not None, "please pass gen_n to trainer"
        schedule = sorted(list(self.lr_schedule.items()))
        _, lr = schedule.pop(0)
        for gen_threshold, gen_rate in schedule:
            if gen_n < gen_threshold:
                break
            lr = gen_rate

        logger.info("using lr {} for gen_n {}", lr, gen_n)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(self, batch, batch_idx):
        return self.step(batch, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, log_prefix="val")

    def step(self, batch, log_prefix):
        # Forward pass
        pos, policy_target, q_penalty_target, q_no_penalty_target = batch
        policy_logprob, q_penalty_pred, q_no_penalty_pred = self.forward(pos)
        policy_logprob_targets = torch.log(policy_target + self.EPS)

        # Losses
        policy_loss = self.policy_kl_div(policy_logprob_targets, policy_logprob)
        q_penalty_loss = self.q_penalty_mse(q_penalty_pred, q_penalty_target)
        q_no_penalty_loss = self.q_no_penalty_mse(
            q_no_penalty_pred, q_no_penalty_target
        )
        loss = policy_loss + q_penalty_loss + q_no_penalty_loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        self.log(f"{log_prefix}_policy_kl_div", policy_loss)
        self.log(f"{log_prefix}_value_mse", q_penalty_loss)
        return loss


class ResidualBlock(pl.LightningModule):
    def __init__(self, n_channels: int, kernel_size=3, padding=1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.block(x)
