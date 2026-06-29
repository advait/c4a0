from types import SimpleNamespace

import pytest
import torch

from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training import parse_lr_schedule
from c4a0.utils import BestModelCheckpoint


def make_model() -> ConnectFourNet:
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=8,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule=parse_lr_schedule([0, 1e-3]),
            l2_reg=0.0,
        )
    )


def test_best_model_checkpoint_ignores_sanity_validation():
    callback: BestModelCheckpoint[ConnectFourNet] = BestModelCheckpoint(
        monitor="val_loss",
        mode="min",
    )
    model = make_model()

    sanity_trainer = SimpleNamespace(
        sanity_checking=True,
        callback_metrics={"val_loss": torch.tensor(0.01)},
    )
    callback.on_validation_end(sanity_trainer, model)  # type: ignore[arg-type]

    assert callback.best_model is None
    assert callback.best_score == float("inf")

    trainer = SimpleNamespace(
        sanity_checking=False,
        callback_metrics={"val_loss": torch.tensor(1.25)},
    )
    callback.on_validation_end(trainer, model)  # type: ignore[arg-type]

    assert callback.best_model is not None
    assert callback.best_score == pytest.approx(1.25)
