#!/usr/bin/env python

from argparse import ArgumentParser
import asyncio
import logging
import multiprocessing as mp
from sys import platform
from typing import List

import torch
from model_storage import load_all_models
from tournament import ModelPlayer, Player, RandomPlayer, UniformPlayer, play_tournament

from training import train


async def main():
    default_device = get_torch_device()

    parser = ArgumentParser()
    parser.add_argument(
        "--n-games", type=int, default=100, help="number of games per generation"
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=mp.cpu_count() - 1,
        help="number of processes to use for self-play",
    )
    parser.add_argument(
        "--mcts-iterations",
        type=int,
        default=150,
        help="number of MCTS iterations per move",
    )
    parser.add_argument(
        "--exploration-constant",
        type=float,
        default=1.4,
        help="MCTS exploration constant",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="batch size for training"
    )
    parser.add_argument(
        "--device", type=str, default=default_device, help="pytorch device"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    await do_training(args)
    # await do_tournament(args)


async def do_training(args):
    while True:
        await train(
            n_games=args.n_games,
            n_processes=args.n_processes,
            mcts_iterations=args.mcts_iterations,
            exploration_constant=args.exploration_constant,
            batch_size=args.batch_size,
            device=args.device,
        )


async def do_tournament(args):
    logger = logging.getLogger(__name__)
    models = load_all_models()
    players: List[Player] = [
        ModelPlayer(f"gen{gen}", model, args.device) for gen, model in models
    ]
    players += [RandomPlayer(), UniformPlayer()]
    results = await play_tournament(
        players,
        games_per_match=args.n_games,
        exploration_constant=args.exploration_constant,
        mcts_iterations=args.mcts_iterations,
    )
    logger.info("Tournament results:")
    logger.info("\n".join(f"{p.name}: {score}" for p, score in results))


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if platform == "darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            raise RuntimeError(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

    return torch.device("cpu")


if __name__ == "__main__":
    asyncio.run(main())
