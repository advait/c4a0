#!/usr/bin/env python
"""Append exactly one measured generation to a COPY of an existing training run."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from c4a0.benchmark import GenerationMetrics
from c4a0.training import TrainingGen, default_eval_config, train_single_gen
from c4a0.utils import get_torch_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--phase", choices=("all", "training"), default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--torch-threads", type=int, default=16)
    parser.add_argument(
        "--eval-mcts-iterations", type=int, nargs="*", default=[64, 256]
    )
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = get_torch_device() if args.device == "auto" else torch.device(args.device)
    metrics = GenerationMetrics(device)
    metrics.synchronize()
    start = time.perf_counter()
    parent = TrainingGen.load_latest(str(args.base_dir))
    gen = train_single_gen(
        str(args.base_dir),
        device,
        parent,
        parent.config,
        default_eval_config(args.eval_mcts_iterations),
        metrics=metrics,
        reuse_games=args.phase == "training",
    )
    metrics.synchronize()
    print(
        json.dumps(
            {
                "phase": args.phase,
                "device": str(device),
                "seed": args.seed,
                "torch_threads": args.torch_threads,
                "generation": gen.gen_n,
                "config": gen.config.model_dump(),
                "seconds": metrics.seconds,
                "epochs": metrics.epochs,
                "optimizer_steps": metrics.optimizer_steps,
                "val_loss": gen.val_loss,
                "total_seconds": time.perf_counter() - start,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
