#!/usr/bin/env python
"""Compare master sample conversion with bulk conversion, checking every output byte."""

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Callable
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset

import c4a0_rust
from c4a0.training import SampleDataModule


class _DataModule(Protocol):
    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]: ...
    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, ...]]: ...


def _uniform(
    _model_id: int, positions: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    n = len(positions)
    return (
        np.zeros((n, 7), np.float32),
        np.zeros(n, np.float32),
        np.zeros(n, np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=700_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--baseline", default="b5eca94")
    args = parser.parse_args()
    if args.samples < 1 or args.runs < 1:
        parser.error("samples and runs must be positive")
    torch.set_num_threads(args.torch_threads)
    # This commit adds timing only; its DataModule is unchanged from master 57d3a2a.
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline}:src/c4a0/training.py"]
    )
    legacy = ModuleType("_bench_legacy_training")
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, "baseline/training.py", "exec"), legacy.__dict__)
    factory = cast(
        Callable[[list[c4a0_rust.Sample], list[c4a0_rust.Sample], int], _DataModule],
        legacy.SampleDataModule,
    )
    games = c4a0_rust.play_games(
        [c4a0_rust.GameMetadata(i, 0, 0) for i in range(8)], 8, 4, 2.0, 0.01, _uniform
    )
    n_original = sum(len(g.samples) for g in games.results)
    games = c4a0_rust.PlayGamesResult.concat(
        [games] * max(1, args.samples // (2 * n_original))
    )
    for run in range(args.runs):
        expected: list[list[str]] | None = None
        for arm in ("legacy", "bulk"):
            gc.collect()
            started = time.perf_counter()
            dm: _DataModule
            if arm == "legacy":
                train, val = games.split_train_test(0.8, 1337)
                dm = factory(train, val, 2000)
                del train, val
            else:
                train_arrays, val_arrays = games.split_train_test_numpy(0.8, 1337)
                dm = SampleDataModule(train_arrays, val_arrays, 2000)
                del train_arrays, val_arrays
            datasets = [dm.train_dataloader().dataset, dm.val_dataloader().dataset]
            elapsed = time.perf_counter() - started
            hashes: list[list[str]] = []
            n_samples = 0
            for dataset in datasets:
                assert isinstance(dataset, TensorDataset)
                n_samples += len(dataset)
                hashes.append(
                    [
                        hashlib.sha256(t.numpy().tobytes()).hexdigest()
                        for t in dataset.tensors
                    ]
                )
            if arm == "legacy":
                expected = hashes
            else:
                assert hashes == expected
            print(
                json.dumps(
                    {
                        "run": run,
                        "arm": arm,
                        "seconds": elapsed,
                        "samples": n_samples,
                        "equal": True,
                    }
                ),
                flush=True,
            )
            del dm, datasets, dataset


if __name__ == "__main__":
    main()
