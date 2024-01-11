"""
The Neural Network is used to evaluate the position of the game.
"""

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Awaitable, Callable, Dict, List, NewType, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from einops import rearrange

from c4 import N_COLS, N_ROWS, Pos, pos_from_bytes, pos_to_bytes


Policy = NewType("Policy", np.ndarray)
"""Represents a N_COLS-dimensional vector of probabilities."""

Value = NewType("Value", float)
"""
Represents the value of a position, a continuous number in [-1, 1].

1 is a win for the 1 player.
-1 is a win for the -1 player.
0 is a draw.
"""

EPS = 1e-8  # Epsilon small constant to avoid log(0)


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
        return self.step(batch, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, log_prefix="val")

    def step(self, batch, log_prefix):
        # Forward pass
        game_ids, inputs, policy_targets, value_targets = batch
        policy_logprob, value_pred = self.forward(inputs)
        value_pred = rearrange(value_pred, "b 1 -> b")
        policy_logprob_targets = torch.log(policy_targets + EPS)

        # Losses
        policy_loss = self.policy_kl_div(policy_logprob_targets, policy_logprob)
        value_loss = self.value_mse(value_pred, value_targets)
        loss = 6 * policy_loss + value_loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        self.log(f"{log_prefix}_policy_kl_div", policy_loss)
        self.log(f"{log_prefix}_value_mse", value_loss)
        return loss


EvalPos = Callable[[Pos], Awaitable[Tuple[Policy, Value]]]


def create_batcher(
    model: ConnectFourNet,
    device: torch.device,
    nn_flush_freq_s: float = 0.01,
    nn_max_batch_size: int = 20000,
) -> Tuple[asyncio.Future, EvalPos]:
    """
    Enables batch inference. Returns an end_signal and an EvalPos function. The function queues
    the position to be evaluated in a future batch. The batch is evaluated in a background thread.

    Set the end_signal to any value to stop the loop.
    """

    end_signal = asyncio.Future()
    pos_queue: List[Tuple[Pos, asyncio.Future]] = []
    nn_bg_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nn_bg_thread")
    nn_last_flush_s = time.time()

    async def enqueue_pos(pos: Pos) -> Tuple[Policy, Value]:
        nonlocal pos_queue
        future = asyncio.get_running_loop().create_future()
        pos_queue.append((pos, future))
        return await future

    async def nn_flush_loop():
        nonlocal nn_last_flush_s
        while not end_signal.done():
            elapsed = time.time() - nn_last_flush_s
            if elapsed >= nn_flush_freq_s or len(pos_queue) >= nn_max_batch_size:
                await nn_flush()
                await asyncio.sleep(0)
            else:
                await asyncio.sleep(0.01)

    async def nn_flush():
        nonlocal nn_last_flush_s, pos_queue

        if len(pos_queue) == 0:
            # If there are no positions to evaluate, exit early and reset the timer
            nn_last_flush_s = time.time()
            await asyncio.sleep(0.1)
            return

        # Group identical positions together to avoid duplicate work
        pos_dict: Dict[bytes, List[asyncio.Future[Tuple[Policy, Value]]]] = defaultdict(
            list
        )
        for pos, future in pos_queue:
            pos_dict[pos_to_bytes(pos)].append(future)
        pos_queue.clear()

        positions_bytes = list(pos_dict.keys())
        positions = [pos_from_bytes(s) for s in positions_bytes]
        pos_tensor = torch.from_numpy(np.array(positions)).float().to(device)
        policies_logprobs, values = await asyncio.get_running_loop().run_in_executor(
            nn_bg_thread, run_model_no_grad, model, pos_tensor
        )
        policies_logprobs = policies_logprobs.cpu().numpy()
        policies = np.exp(policies_logprobs)
        values = values.cpu().numpy()
        assert len(positions) == len(policies) == len(values)

        for i in range(len(positions)):
            for future in pos_dict[positions_bytes[i]]:
                future.set_result((policies[i], values[i]))
            del pos_dict[positions_bytes[i]]

        nn_last_flush_s = time.time()

    asyncio.create_task(nn_flush_loop())
    return end_signal, enqueue_pos


def run_model_no_grad(model, pos_tensor):
    with torch.no_grad():
        return model(pos_tensor)
