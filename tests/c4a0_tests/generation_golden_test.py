"""CPU golden captured from master 57d3a2a, before generation optimizations."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from c4a0.nn import ConnectFourNet
from c4a0.training import EvalConfig, SampleDataModule, TrainingGen
from c4a0_tests.training_test import (
    EXPERIMENT_CONFIG,
    _tiny_loop,
    _tiny_suite,
)

GOLDEN = Path(__file__).parent / "fixtures" / "generation_golden.json"


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def capture_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    losses: dict[str, list[float]] = {"train": [], "val": []}
    original_step = ConnectFourNet.step

    def step(
        self: ConnectFourNet, batch: list[torch.Tensor], log_prefix: str
    ) -> torch.Tensor:
        loss = original_step(self, batch, log_prefix)
        losses[log_prefix].append(loss.detach().item())
        return loss

    monkeypatch.setattr(ConnectFourNet, "step", step)
    config = EXPERIMENT_CONFIG.model_copy(
        update={
            "train_max_epochs": 2,
            "training_batch_size": 64,
            "self_play_batch_size": 1,
        }
    )
    _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), max_gens=2, config=config)
    generations: list[dict[str, object]] = []
    for gen in reversed(TrainingGen.load_all(str(tmp_path))):
        games = gen.get_games(str(tmp_path))
        samples: list[list[str]] = []
        if games is not None:
            train, val = games.split_train_test_numpy(0.8, 1337)
            dm = SampleDataModule(train, val, 64)
            for loader in (dm.train_dataloader(), dm.val_dataloader()):
                # Hash the unshuffled, augmented dataset rather than random batches.
                assert isinstance(loader.dataset, TensorDataset)
                samples.append([tensor_hash(t) for t in loader.dataset.tensors])
        state = games.__getstate__() if games is not None else None
        assert state is None or isinstance(state, bytes)
        generations.append(
            {
                "games": hashlib.sha256(state).hexdigest()
                if state is not None
                else None,
                "samples": samples,
                "val_loss": gen.val_loss,
                "model": {
                    key: tensor_hash(t)
                    for key, t in gen.get_model(str(tmp_path)).state_dict().items()
                },
            }
        )
    return {"generations": generations, "losses": losses}


def test_two_generation_master_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert capture_generations(tmp_path, monkeypatch) == json.loads(GOLDEN.read_text())
