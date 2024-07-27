import copy
from sys import platform
from typing import Generic, Optional, TypeVar

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer


def get_torch_device() -> torch.device:
    """Tries to use cuda or mps, if available, otherwise falls back to cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if platform == "darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS unavailable because the current torch install was not built with MPS enabled."
            )
        else:
            raise RuntimeError(
                "MPS unavailable because the current MacOS version is not 12.3+ and/or you do not "
                "have an MPS-enabled device on this machine."
            )

    return torch.device("cpu")


M = TypeVar("M", bound=pl.LightningModule)


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
        self.best_model: Optional[M] = None
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_end(self, trainer: Trainer, pl_module: M) -> None:
        """
        Check if the current model is the best so far.

        This method is called after each validation epoch. It compares the current
        monitored metric with the best one so far and updates the best model if necessary.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The current PyTorch Lightning module.
        """
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        if self.mode == "min":
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_model = copy.deepcopy(pl_module)
        elif self.mode == "max":
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_model = copy.deepcopy(pl_module)

    def get_best_model(self) -> M:
        """
        Returns the best model found during training.
        """
        assert self.best_model is not None, "no model checkpoint called"
        return self.best_model
