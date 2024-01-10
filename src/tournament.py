"""
Round-robin tournament to determine which model is the best.
"""


import abc
import asyncio
from collections import defaultdict
import itertools
import logging
from typing import Dict, List, Tuple
import numpy as np

import torch
from tqdm import tqdm
from c4 import N_COLS, Pos

from nn import ConnectFourNet, Policy, Value, create_batcher
from self_play import GameID, gen_game


class Player(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        raise NotImplementedError

    def close(self):
        pass


class ModelPlayer(Player):
    def __init__(
        self,
        name: str,
        model: ConnectFourNet,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(name)
        self.device = device
        self.end_signal, self.enqueue_pos = create_batcher(model, device)

    def close(self):
        self.end_signal.set_result(True)

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        return await self.enqueue_pos(pos)


class RandomPlayer(Player):
    def __init__(self):
        super().__init__("random")

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        policy_logits = np.random.rand(N_COLS)
        policy_exp = np.exp(policy_logits)
        policy = policy_exp / policy_exp.sum()
        value = Value(np.random.rand() * 2 - 1)  # [-1, 1]
        return policy, value


class UniformPlayer(Player):
    def __init__(self):
        super().__init__("uniform")

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        policy = Policy(np.ones(N_COLS) / N_COLS)
        value = Value(0.0)
        return policy, value


async def play_tournament(
    players: List[Player],
    games_per_match: int,
    exploration_constant: float,
    mcts_iterations: int,
) -> List[Tuple[Player, float]]:
    """Players a round-robin tournament, returning the total score of each player."""
    logger = logging.getLogger(__name__)
    pairings = list(itertools.permutations(players, 2))
    results: Dict[Player, float] = defaultdict(lambda: 0.0)
    game_pbar = tqdm(total=len(pairings) * games_per_match)

    async def play_game(p0: Player, p1: Player):
        samples = await gen_game(
            game_id=GameID(0),
            eval_pos0=p0.eval_pos,
            eval_pos1=p1.eval_pos,
            exploration_constant=exploration_constant,
            mcts_iterations=mcts_iterations,
            submit_mcts_iter=None,
        )
        game_id, pos, policy, value = samples[0]
        results[p0] += (value + 1) / 2
        results[p1] += (-value + 1) / 2

    logger.info(f"Beginning tournament with {len(players)} players")
    coros = [play_game(p0, p1) for p0, p1 in pairings for _ in range(games_per_match)]
    await asyncio.gather(*coros)

    for player in players:
        player.close()

    ret = list(results.items())
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret
