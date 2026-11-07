import copy
from typing import Generic, TypeVar, cast

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)


def lightning_accelerator(device: torch.device) -> str:
    """The Lightning accelerator that trains on `device`. `"auto"` would pick a GPU even when the
    caller asked for CPU, leaving the trained model on a different device than its inputs."""
    return {"cuda": "gpu", "mps": "mps"}.get(device.type, "cpu")


def get_torch_device() -> torch.device:
    """Tries to use cuda or mps, if available, otherwise falls back to cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


M = TypeVar("M", bound=pl.LightningModule)


class GenerationTrainer(Trainer):
    """Lightning trainer with the generation metadata consumed by the LR schedule."""

    gen_n: int
    # Lightning initializes this through its callback connector.
    callbacks: list[Callback]


class BestModelCheckpoint(Callback, Generic[M]):
    """
    PyTorch Lightning callback that keeps track of the best model in memory during training.

    This callback monitors a specified metric and saves the model with the best
    score in memory. It can be used to retrieve the best model after training.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min") -> None:
        """
        Initialize the BestModelCheckpoint callback.

        Args:
            monitor (str): Name of the metric to monitor. Defaults to 'val_loss'.
            mode (str): One of {'min', 'max'}. In 'min' mode, the lowest metric value is considered
                        best, in 'max' mode the highest. Defaults to 'min'.
        """
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_model: M | None = None
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_end(
        self, trainer: Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Check if the current model is the best so far.

        This method is called after each validation epoch. It compares the current
        monitored metric with the best one so far and updates the best model if necessary.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning module.
        """
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        # fit supplies M; the Lightning callback API accepts the base module type.
        tracked_model = cast(M, pl_module)
        if self.mode == "min":
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_model = copy.deepcopy(tracked_model)
        elif self.mode == "max":
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model = copy.deepcopy(tracked_model)

    def get_best_model(self) -> M:
        """
        Returns the best model found during training.
        """
        assert self.best_model is not None, "no model checkpoint called"
        return self.best_model


def tensor_batches(
    tensors: tuple[torch.Tensor, ...], batch_size: int, shuffle: bool, device: str
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Whole-batch indexing into tensors kept on `device`. The default per-sample collate pins one
    CPU core and starves the GPU (about 33% utilization for the default net, and about 5 of
    8 minutes per self-play generation with a replay window)."""
    ds = TensorDataset(*(t.to(device) for t in tensors))
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(
        ds, batch_size=None, sampler=BatchSampler(sampler, batch_size, False)
    )
