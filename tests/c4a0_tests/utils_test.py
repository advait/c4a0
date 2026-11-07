import pytest
import torch
from pytorch_lightning import Trainer

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


def test_best_model_checkpoint_ignores_sanity_validation() -> None:
    callback: BestModelCheckpoint[ConnectFourNet] = BestModelCheckpoint(
        monitor="val_loss",
        mode="min",
    )
    model = make_model()

    sanity_trainer = Trainer(logger=False, enable_checkpointing=False)
    sanity_trainer.callback_metrics["val_loss"] = torch.tensor(0.01)
    sanity_trainer.sanity_checking = True
    callback.on_validation_end(sanity_trainer, model)

    assert callback.best_model is None
    assert callback.best_score == float("inf")

    trainer = Trainer(logger=False, enable_checkpointing=False)
    trainer.callback_metrics["val_loss"] = torch.tensor(1.25)
    callback.on_validation_end(trainer, model)

    assert callback.best_model is not None
    assert callback.best_score == pytest.approx(1.25)


def test_lightning_accelerator_follows_the_requested_device() -> None:
    import torch

    from c4a0.utils import lightning_accelerator

    assert lightning_accelerator(torch.device("cpu")) == "cpu"
    assert lightning_accelerator(torch.device("cuda")) == "gpu"
    assert lightning_accelerator(torch.device("mps")) == "mps"


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    from c4a0.utils import get_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert get_torch_device() == torch.device(expected)
