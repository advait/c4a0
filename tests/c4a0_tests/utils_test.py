import pickle
from pathlib import Path

import pytest
import torch
from pytorch_lightning import Trainer

from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training import parse_lr_schedule
from c4a0.utils import BestModelCheckpoint, load_model


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


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_load_model_round_trips_to_requested_device(
    tmp_path: Path, device: str
) -> None:
    model = make_model()
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(model))

    restored = load_model(path, torch.device(device))

    assert restored.device == torch.device(device)
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, tensor in restored.state_dict().items():
        assert tensor.device == torch.device(device)
        assert tensor.shape == model.state_dict()[name].shape
        if device == "cpu":
            torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)


def test_load_legacy_cuda_pickle_on_cpu_without_cuda(
    legacy_cuda_model: tuple[ConnectFourNet, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, path = legacy_cuda_model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        RuntimeError, match="Attempting to deserialize object on a CUDA device"
    ):
        pickle.loads(path.read_bytes())

    restored = load_model(path, "cpu")

    for name, tensor in restored.state_dict().items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, model.state_dict()[name], rtol=0, atol=0)


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


def test_fixed_batch_budget_samples_with_replacement_on_requested_device() -> None:
    from torch.utils.data import BatchSampler, RandomSampler, TensorDataset

    from c4a0.utils import tensor_batches

    # More draws than rows proves replacement; each batch has the requested size.
    data = torch.arange(3)
    loader = tensor_batches((data,), 8, True, torch.device("cpu"), n_batches=5)
    assert isinstance(loader.dataset, TensorDataset)
    assert loader.dataset.tensors[0].data_ptr() == data.data_ptr()
    assert isinstance(loader.sampler, BatchSampler)
    assert isinstance(loader.sampler.sampler, RandomSampler)
    assert loader.sampler.sampler.replacement
    batches = list(loader)
    assert len(batches) == 5
    assert all(batch[0].shape == (8,) for batch in batches)
    assert all(0 <= int(v) < 3 for batch in batches for v in batch[0])


def test_cuda_memory_helpers_are_noops_off_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from c4a0.utils import (
        CUDA_MEMORY_FRACTION_ENV,
        limit_cuda_memory,
        release_cuda_cache,
    )

    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "0.1")
    limit_cuda_memory("cpu")
    release_cuda_cache(torch.device("cpu"))


def test_cuda_memory_fraction_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    from c4a0.utils import CUDA_MEMORY_FRACTION_ENV, limit_cuda_memory

    calls: list[tuple[float, int]] = []

    def record(fraction: float, device: object) -> None:
        # Mirrors the real API, which rejects a torch.device without an index.
        assert isinstance(device, int), device
        calls.append((fraction, device))

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", record)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "1.5")
    with pytest.raises(ValueError, match="must be in"):
        limit_cuda_memory("cuda")
    monkeypatch.setenv(CUDA_MEMORY_FRACTION_ENV, "0.08")
    limit_cuda_memory("cuda")
    assert calls == [(0.08, 0)]
