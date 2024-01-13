#!/usr/bin/env python

import asyncio
import logging
import multiprocessing as mp
from typing import Literal
import warnings

import clipstick
from pydantic import BaseModel
import torch

from training import train_gen
from utils import get_torch_device


class Train(BaseModel):
    """Trains a model via self-play."""

    n_games: int = 2000
    """number of games per generation"""

    batch_size: int = 100
    """batch size for training"""


class Tournament(BaseModel):
    """Runs a tournament between multiple models."""

    n_top_models: int = 5
    """Number of top models to use for the tournament (based on last gen's tournament results)."""


class MainArgs(BaseModel):
    """c4a0: self-improving connect four AI."""

    sub_command: Train | Tournament

    n_processes: int = mp.cpu_count() - 1
    """number of processes to use for self-play"""

    mcts_iterations: int = 150
    """number of MCTS iterations per move"""

    exploration_constant: float = 1.4
    """MCTS exploration constant"""

    device: Literal["cuda", "mps", "cpu"] = str(get_torch_device())  # type: ignore
    """pytroch device"""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    """log level"""


async def main():
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    args = clipstick.parse(MainArgs)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    if isinstance(args.sub_command, Train):
        while True:
            await train_gen(
                n_games=args.sub_command.n_games,
                n_processes=args.n_processes,
                mcts_iterations=args.mcts_iterations,
                exploration_constant=args.exploration_constant,
                batch_size=args.sub_command.batch_size,
                device=torch.device(args.device),
            )
    elif isinstance(args.sub_command, Tournament):
        raise NotImplementedError
    else:
        logger.error("Invalid sub-command: %s", args.sub_command)


if __name__ == "__main__":
    asyncio.run(main())
