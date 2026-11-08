"""CPU golden captured from master 57d3a2a, before generation optimizations."""

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import c4a0_rust
from c4a0.nn import ConnectFourNet
from c4a0.training import EvalConfig, SampleDataModule, TrainingGen
from c4a0_tests.training_test import (
    EXPERIMENT_CONFIG,
    _tiny_loop,
    _tiny_suite,
)

if TYPE_CHECKING:
    from c4a0_rust import EvalPosCb

GOLDEN = Path(__file__).parent / "fixtures" / "generation_golden.json"


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def capture_generations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_off: bool = False
) -> dict[str, object]:
    random.seed(1337)
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # Master returns games in worker completion order, so its split is otherwise
    # scheduling-dependent even at batch size one. Serialize games only in this test.
    native_play = c4a0_rust.play_games

    def serial_play(
        reqs: Sequence[c4a0_rust.GameMetadata],
        max_nn_batch_size: int,
        n_mcts_iterations: int,
        c_exploration: float,
        c_ply_penalty: float,
        callback: "EvalPosCb",
        root_noise_alpha: float | None = None,
        root_noise_frac: float = 0.0,
        value_target_q_weight: float = 0.0,
        random_start_frac: float = 0.0,
        random_start_max_ply: int = 12,
    ) -> c4a0_rust.PlayGamesResult:
        result = c4a0_rust.PlayGamesResult()
        for req in reqs:
            result = result + native_play(
                [req],
                max_nn_batch_size,
                n_mcts_iterations,
                c_exploration,
                c_ply_penalty,
                callback,
                root_noise_alpha=root_noise_alpha,
                root_noise_frac=root_noise_frac,
                value_target_q_weight=value_target_q_weight,
                random_start_frac=random_start_frac,
                random_start_max_ply=random_start_max_ply,
            )
        return result

    monkeypatch.setattr(c4a0_rust, "play_games", serial_play)
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
    if explicit_off:
        config = config.model_copy(
            update={"bf16": False, "compile": False, "train_steps": None}
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


@pytest.mark.parametrize("explicit_off", [False, True])
def test_two_generation_master_golden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_off: bool
) -> None:
    assert capture_generations(tmp_path, monkeypatch, explicit_off) == json.loads(
        GOLDEN.read_text()
    )
