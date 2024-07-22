#!/usr/bin/env python

from typing import Literal
import warnings

import clipstick
from pydantic import BaseModel
import torch

from c4a0.sweep import sweep
from c4a0.training import training_loop
from c4a0.utils import get_torch_device


class Train(BaseModel):
    """Trains a model via self-play."""

    n_self_play_games: int = 2000
    """number of games per generation"""

    self_play_batch_size: int = 2000
    """batch size for self play inference"""

    training_batch_size: int = 2000
    """batch size for training"""

    def run(self, args: "MainArgs"):
        training_loop(
            base_dir=args.training_base_dir,
            device=torch.device(args.device),
            n_self_play_games=self.n_self_play_games,
            n_mcts_iterations=args.n_mcts_iterations,
            exploration_constant=args.exploration_constant,
            self_play_batch_size=self.self_play_batch_size,
            training_batch_size=self.training_batch_size,
        )


class Sweep(BaseModel):
    """Performs a hyperparameter sweep."""

    def run(self, args: "MainArgs"):
        sweep(args.training_base_dir)


class MainArgs(BaseModel):
    """c4a0: self-improving connect four AI."""

    sub_command: Train | Sweep

    n_mcts_iterations: int = 150
    """number of MCTS iterations per move"""

    exploration_constant: float = 1.4
    """MCTS exploration constant"""

    device: Literal["cuda", "mps", "cpu"] = str(get_torch_device())  # type: ignore
    """pytroch device"""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """log level"""

    training_base_dir: str = "training"
    """base directory for training data"""


def main():
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    args = clipstick.parse(MainArgs)
    args.sub_command.run(args)


if __name__ == "__main__":
    main()
