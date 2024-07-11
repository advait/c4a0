#!/usr/bin/env python

import asyncio
import logging
from typing import List, Literal
import warnings

import clipstick
from pydantic import BaseModel
import torch

from c4a0.tournament import ModelID, ModelPlayer, Player
from c4a0.training import TrainingState, train_gen
from c4a0.utils import get_torch_device


class Train(BaseModel):
    """Trains a model via self-play."""

    n_games: int = 2000
    """number of games per generation"""

    batch_size: int = 2000
    """batch size for training"""

    async def run(self, args: "MainArgs"):
        while True:
            await train_gen(
                n_games=self.n_games,
                mcts_iterations=args.mcts_iterations,
                exploration_constant=args.exploration_constant,
                batch_size=self.batch_size,
                device=torch.device(args.device),
            )


class Tournament(BaseModel):
    """Runs a tournament between multiple models."""

    gen_id: List[int]
    """Which model generations should play in the tournament."""

    games_per_match: int = 25
    """Number of games to play between each pair of models."""

    async def run(self, args: "MainArgs"):
        state = TrainingState.load_training_state()
        models = [
            (ModelID(gen_id), state.get_model(ModelID(gen_id)))
            for gen_id in self.gen_id
        ]

        players: List[Player] = [
            ModelPlayer(
                gen_id,
                model,
                torch.device(args.device),
            )
            for gen_id, model in models
        ]
        return
        # result = await play_tournament(
        #     players=players,
        #     games_per_match=self.games_per_match,
        #     exploration_constant=args.exploration_constant,
        #     mcts_iterations=args.mcts_iterations,
        # )
        # print(result.scores_table())


class MainArgs(BaseModel):
    """c4a0: self-improving connect four AI."""

    sub_command: Train | Tournament

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

    await args.sub_command.run(args)


if __name__ == "__main__":
    asyncio.run(main())
