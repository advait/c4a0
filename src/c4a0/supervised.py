"""
The supervised ceiling: train the self-play architecture directly on solver labels to learn
whether it can represent solver-perfect play at all. Targets match what self-play trains on:
policy over moves, game outcome without penalty, and outcome with the ply penalty at the
perfect-play terminal ply.
"""

import time
from typing import Literal, Optional

from loguru import logger
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from c4a0.evaluation import SuiteEval
from c4a0.labels import Labels
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.utils import BestModelCheckpoint

import c4a0_rust

TargetVersion = Literal["pons_terminal_ply_v1"]
TARGET_VERSION: TargetVersion = "pons_terminal_ply_v1"


def terminal_ply(
    score: NDArray[np.integer], ply: NDArray[np.integer]
) -> NDArray[np.int64]:
    """Ply of the final position under perfect play, from Pons score `score` at ply `ply`.

    Pons scores a win as 22 minus the winner's stone count including the winning stone. The
    first player's k-th stone lands on ply 2k - 1, the second player's on ply 2k. The player to
    move wins when score > 0, the opponent when score < 0. Draws end at ply 42.
    """
    s = score.astype(np.int64)
    parity = ply.astype(np.int64) % 2
    win = 43 - 2 * s + parity
    loss = 44 + 2 * s - parity
    return np.where(s > 0, win, np.where(s < 0, loss, 42))


def solver_targets(
    labels: Labels, c_ply_penalty: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pos, policy, q_penalty, q_no_penalty) tensors: policy uniform over solver-best moves,
    q_no_penalty the solver value, q_penalty the value times (1 - c_ply_penalty * terminal ply)
    as in `Pos::terminal_value_with_ply_penalty`."""
    pos = c4a0_rust.encode_positions(labels.moves)
    best = labels.best.astype(np.float32)
    policy = best / best.sum(axis=1, keepdims=True)
    value = labels.value.astype(np.float32)
    end = terminal_ply(labels.best_score, labels.ply)
    q_penalty = (value * (1.0 - c_ply_penalty * end)).astype(np.float32)
    return (
        torch.from_numpy(pos),
        torch.from_numpy(policy),
        torch.from_numpy(q_penalty),
        torch.from_numpy(value),
    )


def _mirrored(
    tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    pos, policy, q_penalty, q_no_penalty = tensors
    return (
        torch.cat([pos, pos.flip(-1)]),
        torch.cat([policy, policy.flip(-1)]),
        torch.cat([q_penalty, q_penalty]),
        torch.cat([q_no_penalty, q_no_penalty]),
    )


def _batches(
    tensors: tuple[torch.Tensor, ...], batch_size: int, shuffle: bool, device: str
) -> DataLoader:
    """Whole-batch indexing into tensors kept on `device`. The default per-sample collate pins one
    CPU core and starves the GPU (about 33% utilization for the default net)."""
    ds = TensorDataset(*(t.to(device) for t in tensors))
    sampler = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    return DataLoader(
        ds, batch_size=None, sampler=BatchSampler(sampler, batch_size, False)
    )


class _EpochLog(pl.Callback):
    """One log line per epoch, so a detached run's progress is visible in its log."""

    def on_train_start(self, trainer, pl_module):
        self.t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        logger.info(
            "epoch {} train_loss {:.4f} val_loss {:.4f} elapsed {:.0f}s",
            trainer.current_epoch,
            float(m.get("train_loss", float("nan"))),
            float(m.get("val_loss", float("nan"))),
            time.perf_counter() - self.t0,
        )


def fit_supervised(
    model: ConnectFourNet,
    train: Labels,
    val: Labels,
    c_ply_penalty: float,
    batch_size: int,
    max_epochs: int,
    patience: int = 10,
) -> tuple[ConnectFourNet, int]:
    """Trains on `train` (plus mirror images) with the self-play loss, early stopping on `val`
    loss. Returns the best model and the number of epochs run."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = _batches(
        _mirrored(solver_targets(train, c_ply_penalty)), batch_size, True, device
    )
    val_dl = _batches(solver_targets(val, c_ply_penalty), batch_size, False, device)
    best_model_cb = BestModelCheckpoint(monitor="val_loss", mode="min")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[
            best_model_cb,
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ],
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.gen_n = 0  # type: ignore
    trainer.callbacks.append(_EpochLog())  # type: ignore
    model.train()
    trainer.fit(model, train_dl, val_dl)
    return best_model_cb.get_best_model(), trainer.current_epoch


class CeilingReport(BaseModel):
    net: ModelConfig
    train_labels: str
    train_sha256: str
    n_train: int
    epochs: int
    batch_size: int
    seed: int
    c_ply_penalty: float
    target_version: TargetVersion
    dev: SuiteEval
    test: Optional[SuiteEval] = None
    """Scored only on request, after the configuration was chosen on dev."""
