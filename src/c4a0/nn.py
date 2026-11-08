from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Self, cast

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, Field, model_validator
from torch.optim.adam import Adam

from c4a0.utils import GenerationTrainer
from c4a0_rust import N_COLS, N_ROWS

if TYPE_CHECKING:
    from c4a0.training import GenConfig

Head = Literal["fc", "conv"]
Arch = Literal["resnet", "resnet_gpool", "transformer", "hybrid"]


class ModelConfig(BaseModel):
    """Configuration for ConnectFourNet."""

    arch: Arch = "resnet"
    n_layers: int = Field(default=6, ge=1)
    d_model: int = Field(default=128, ge=1)
    n_heads: int = Field(default=4, ge=1)

    head: Head = "fc"
    """Policy and value head architecture."""

    value_hidden: int = Field(default=256, ge=1)
    """Hidden value layer width for convolutional and token heads."""

    n_residual_blocks: int
    """The number of residual blocks."""

    conv_filter_size: int
    """The number of filters for the conv layers in the residual blocks."""

    n_policy_layers: int
    """Number of fully connected layers in the policy head; ignored for conv heads."""

    n_value_layers: int
    """Number of fully connected layers in the value head; ignored for conv heads."""

    lr_schedule: dict[int, float]
    """
    Learning rate schedule. The first item in the tuple indicates which gen_n the learning rate
    begins to be effective for. The second item in the tuple is the learning rate.
    """

    l2_reg: float
    """L2 weight decay regularization for the optimizer"""

    @model_validator(mode="after")
    def validate_attention_width(self) -> Self:
        if self.arch in ("transformer", "hybrid") and self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads for token models")
        return self


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
    followed by separate policy and value heads (fully connected or convolutional).
    """

    EPS = 1e-8  # Epsilon small constant to avoid log(0)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.lr_schedule = config.lr_schedule
        self.l2_reg = config.l2_reg
        self.head = config.head
        self.arch = config.arch

        if config.arch in ("transformer", "hybrid"):
            self.tokens = TokenTrunk(config)
            self.token_policy = nn.Linear(config.d_model, 1)
            self.token_value = nn.Sequential(
                nn.Linear(config.d_model, config.value_hidden),
                nn.GELU(),
                nn.Linear(config.value_hidden, 2),
                nn.Tanh(),
            )
            self.save_hyperparameters(config.model_dump())
            return

        block = (
            GlobalPoolResidualBlock if config.arch == "resnet_gpool" else ResidualBlock
        )
        self.conv = nn.Sequential(
            nn.Conv2d(2, config.conv_filter_size, kernel_size=3, padding=1),
            *[block(config.conv_filter_size) for i in range(config.n_residual_blocks)],
        )

        if config.head == "conv":
            self.conv_policy = nn.Sequential(
                nn.Conv2d(config.conv_filter_size, 2, kernel_size=1),
                nn.BatchNorm2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(2 * N_ROWS * N_COLS, N_COLS),
                nn.LogSoftmax(dim=1),
            )
            self.conv_value = nn.Sequential(
                nn.Conv2d(config.conv_filter_size, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(N_ROWS * N_COLS, config.value_hidden),
                nn.ReLU(),
                nn.Linear(config.value_hidden, 2),
                nn.Tanh(),
            )
        else:
            self._init_fc_heads(config)

        self.save_hyperparameters(config.model_dump())

    def _init_fc_heads(self, config: ModelConfig) -> None:
        # Preserve module names, initialization order and the dummy trunk pass for FC
        # models: existing pickles and fixed-seed experiments depend on all three.
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

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(self, "arch", "resnet") in ("transformer", "hybrid"):
            x = self.tokens(x)  # batch, row, column, feature
            policy_logprobs = (
                self.token_policy(x.mean(dim=1)).squeeze(-1).log_softmax(dim=1)
            )
            q_values = self.token_value(x.mean(dim=(1, 2)))
            return policy_logprobs, q_values[:, 0], q_values[:, 1]
        x = self.conv(x)
        # Pickles saved before head selection have no `head` attribute.
        if getattr(self, "head", "fc") == "conv":
            policy_logprobs = self.conv_policy(x)
            q_values = self.conv_value(x)
        else:
            x = rearrange(x, "b c h w -> b (c h w)")
            policy_logprobs = self.fc_policy(x)
            q_values = self.fc_value(x)  # b 2
        q_penalty, q_no_penalty = q_values.split(1, dim=1)  # both: b 1
        q_penalty = q_penalty.squeeze(1)  # b
        q_no_penalty = q_no_penalty.squeeze(1)  # b
        return policy_logprobs, q_penalty, q_no_penalty

    def forward_numpy(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """
        Forward pass with input/output as numpy. Model is run in inference mode. Used for self play.
        """
        self.eval()
        pos = torch.from_numpy(x).to(self.device)
        with torch.inference_mode():
            policy, q_penalty, q_no_penalty = self.forward(pos)
        policy_np = policy.cpu().numpy()
        q_penalty_np = q_penalty.cpu().numpy()
        q_no_penalty_np = q_no_penalty.cpu().numpy()
        return policy_np, q_penalty_np, q_no_penalty_np

    def _calculate_conv_output_size(self) -> int:
        """Helper function to calculate the output size of the convolutional block."""
        # Apply the convolutional layers to a dummy input
        dummy_input = torch.zeros(1, 2, N_ROWS, N_COLS)
        with torch.no_grad():
            dummy_output = self.conv(dummy_input)
        return int(torch.numel(dummy_output))

    def configure_optimizers(self) -> Adam:
        gen_n: int = cast(GenerationTrainer, self.trainer).gen_n
        assert gen_n is not None, "please pass gen_n to trainer"
        schedule = sorted(list(self.lr_schedule.items()))
        _, lr = schedule.pop(0)
        for gen_threshold, gen_rate in schedule:
            if gen_n < gen_threshold:
                break
            lr = gen_rate

        logger.info("using lr {} for gen_n {}", lr, gen_n)
        optimizer = Adam(self.parameters(), lr=lr, weight_decay=self.l2_reg)
        return optimizer

    def training_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, log_prefix="train")

    def validation_step(
        self, batch: Sequence[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, log_prefix="val")

    def step(self, batch: Sequence[torch.Tensor], log_prefix: str) -> torch.Tensor:
        # Forward pass
        pos, policy_target, q_penalty_target, q_no_penalty_target = batch
        policy_logprob, q_penalty_pred, q_no_penalty_pred = self.forward(pos)
        policy_logprob_targets = torch.log(policy_target + self.EPS)

        # Losses. Computed functionally: torchmetrics modules cache their last result with its
        # autograd graph, which made BestModelCheckpoint's deepcopy of the model fail.
        policy_loss = nn.functional.kl_div(
            policy_logprob,
            policy_logprob_targets,
            reduction="batchmean",
            log_target=True,
        )
        q_penalty_loss = nn.functional.mse_loss(q_penalty_pred, q_penalty_target)
        q_no_penalty_loss = nn.functional.mse_loss(
            q_no_penalty_pred, q_no_penalty_target
        )
        loss = policy_loss + q_penalty_loss + q_no_penalty_loss

        value_loss = q_penalty_loss + q_no_penalty_loss

        self.log(f"{log_prefix}_loss", loss, prog_bar=True)
        self.log(f"{log_prefix}_policy_kl_div", policy_loss)
        self.log(f"{log_prefix}_q_penalty_mse", q_penalty_loss)
        self.log(f"{log_prefix}_q_no_penalty_mse", q_no_penalty_loss)
        self.log(f"{log_prefix}_value_mse", value_loss)
        return loss


class NumpyEvaluator:
    """One self-play run's inference state, outside the pickled/trained network.

    Rust invokes callbacks serially on its NN thread. Blocking output transfer completes
    the input copy before that thread can overwrite the pinned buffer on the next call.
    """

    def __init__(
        self, model: ConnectFourNet, config: "GenConfig | None" = None
    ) -> None:
        self.model = model.eval()
        self.host: torch.Tensor | None = None
        self.positions: torch.Tensor | None = None
        self.bf16 = config.bf16 if config else False
        self.forward: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = model.forward
        self.compiled = False
        if config and config.compile:
            try:
                # Compile a bound forward, leaving the module and its pickle untouched.
                self.forward = torch.compile(model.forward, dynamic=True)
                self.compiled = True
            except Exception as error:
                logger.warning(
                    "Cannot compile self-play forward; using eager: {}", error
                )

    def _evaluate(
        self, pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.compiled:
            return self.forward(pos)
        try:
            return self.forward(pos)
        except Exception as error:
            # Backend availability errors often arrive on the first call, not at compile().
            logger.warning("Compiled self-play forward failed; using eager: {}", error)
            self.forward = self.model.forward
            self.compiled = False
            return self.forward(pos)

    @torch.inference_mode()
    def __call__(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        pos = torch.from_numpy(x)
        if self.model.device.type == "cuda":
            if self.host is None or self.host.shape[0] < len(x):
                self.host = torch.empty(x.shape, dtype=torch.float32, pin_memory=True)
                self.positions = torch.empty_like(self.host, device=self.model.device)
            assert self.positions is not None
            self.host[: len(x)].copy_(pos)
            pos = self.positions[: len(x)]
            pos.copy_(self.host[: len(x)], non_blocking=True)
        else:
            pos = pos.to(self.model.device)
        with torch.autocast(pos.device.type, dtype=torch.bfloat16, enabled=self.bf16):
            policy, q_penalty, q_no_penalty = self._evaluate(pos)
        # The Rust callback API always consumes f32, including with bf16 autocast.
        policy = policy.float()
        q_penalty = q_penalty.float()
        q_no_penalty = q_no_penalty.float()
        if pos.device.type == "cuda":
            # One device-to-host copy and synchronization for all three outputs.
            outputs = torch.cat(
                (policy, q_penalty[:, None], q_no_penalty[:, None]), dim=1
            )
            array = outputs.cpu().numpy()
            return array[:, :N_COLS], array[:, N_COLS], array[:, N_COLS + 1]
        return policy.cpu().numpy(), q_penalty.cpu().numpy(), q_no_penalty.cpu().numpy()


class ResidualBlock(pl.LightningModule):
    def __init__(self, n_channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GlobalPoolResidualBlock(nn.Module):
    """Broadcast mean/max context after conv1, immediately before conv2."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.pool_bias = nn.Linear(2 * n_channels, n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = self.conv1(x)
        context = torch.cat(
            (activations.mean(dim=(2, 3)), activations.amax(dim=(2, 3))), dim=1
        )
        activations = activations + self.pool_bias(context)[:, :, None, None]
        return x + self.norm(self.conv2(activations)).relu()


class TokenTrunk(nn.Module):
    """42 row-major tokens with separable learned positions and pre-norm attention.

    Token models always use token heads, independent of the fc/conv head option.
    Hybrid's convolutional stem uses n_residual_blocks and conv_filter_size.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        channels = 2
        self.stem: nn.Module = nn.Identity()
        if config.arch == "hybrid":
            channels = config.conv_filter_size
            self.stem = nn.Sequential(
                nn.Conv2d(2, channels, 3, padding=1),
                *[ResidualBlock(channels) for _ in range(config.n_residual_blocks)],
            )
        self.embed = nn.Linear(channels, config.d_model)
        self.row_embedding = nn.Parameter(torch.empty(N_ROWS, 1, config.d_model))
        self.col_embedding = nn.Parameter(torch.empty(1, N_COLS, config.d_model))
        nn.init.normal_(self.row_embedding, std=0.02)
        nn.init.normal_(self.col_embedding, std=0.02)
        # Construct each layer independently rather than cloning identical initial weights.
        self.layers = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=4 * config.d_model,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.n_layers)
            ],
            nn.LayerNorm(config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(self.stem(x).permute(0, 2, 3, 1))
        x = x + self.row_embedding + self.col_embedding
        x = self.layers(x.flatten(1, 2))
        return x.unflatten(1, (N_ROWS, N_COLS))
