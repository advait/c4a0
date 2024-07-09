"""
Round-robin tournament to determine which model is the best.
"""

import abc
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import itertools
import logging
from typing import Dict, List, NewType, Optional, Tuple
import numpy as np

from tabulate import tabulate
import torch
from tqdm import tqdm

from c4a0.nn import ConnectFourNet
from c4a0_rust import N_COLS  # type: ignore

PlayerName = NewType("PlayerName", str)

ModelID = NewType("ModelID", int)


class Player(abc.ABC):
    name: PlayerName

    def __init__(self, name: str):
        self.name = PlayerName(name)

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class ModelPlayer(Player):
    """Player whose policy and value are determined by a ConnectFourNet."""

    gen_id: ModelID
    model: ConnectFourNet
    device: torch.device

    def __init__(self, gen_id: ModelID, model: ConnectFourNet, device: torch.device):
        super().__init__(f"gen{gen_id}")
        self.gen_id = gen_id
        self.model = model
        self.model.to(device)

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.forward_numpy(x)


class RandomPlayer(Player):
    """Player that provides a random policy and value."""

    def __init__(self):
        super().__init__("random")

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = x.shape[0]
        policy_logits = np.random.rand(batch_size, N_COLS)
        value = np.random.rand(batch_size) * 2 - 1  # [-1, 1]
        return policy_logits, value


class UniformPlayer(Player):
    """Player that provides a uniform policy and 0 value."""

    def __init__(self):
        super().__init__("uniform")

    def forward_numpy(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = x.shape[0]
        policy_logits = np.ones(batch_size, N_COLS)
        value = np.zeros(batch_size)
        return policy_logits, value


'''

@dataclass
class TournamentGame:
    """Represents a single game between two players."""

    p0: Player
    p1: Player
    score: float  # From the perspective of p0 (either 0, 0.5, or 1)
    samples: List[Sample]


@dataclass
class TournamentResult:
    """Represents the results from a tournamnet."""

    players: Dict[PlayerName, Player]
    games_per_match: int
    exploitation_constant: float
    mcts_iterations: int
    games: List[TournamentGame] = field(default_factory=list)
    date: datetime = field(default_factory=datetime.now)

    def get_scores(self) -> List[Tuple[Player, float]]:
        scores: Dict[PlayerName, float] = defaultdict(lambda: 0.0)
        for game in self.games:
            scores[game.p0.name] += game.score
            scores[game.p1.name] += 1 - game.score
        ret = [(self.players[name], score) for name, score in scores.items()]
        ret.sort(key=lambda x: x[1], reverse=True)
        return ret

    def scores_table(self) -> str:
        return tabulate(
            [(player.name, score) for player, score in self.get_scores()],
            headers=["Player", "Score"],
            tablefmt="github",
        )

    def get_top_models(self) -> List[GenID]:
        """Returns the top models from the tournament in descending order of performance."""
        return [
            player.gen_id
            for player, _ in self.get_scores()
            if isinstance(player, ModelPlayer)
        ]


async def play_tournament(
    players: List[Player],
    games_per_match: int,
    exploration_constant: float,
    mcts_iterations: int,
) -> TournamentResult:
    """Players a round-robin tournament, returning the total score of each player."""
    logger = logging.getLogger(__name__)
    tournament = TournamentResult(
        players={p.name: p for p in players},
        games_per_match=games_per_match,
        exploitation_constant=exploration_constant,
        mcts_iterations=mcts_iterations,
    )

    pairings = list(itertools.permutations(players, 2))
    logger.info(f"Beginning tournament with {len(players)} players")
    approx_mcts_iters = len(pairings) * games_per_match * 21 * mcts_iterations
    mcts_pbar = tqdm(total=approx_mcts_iters, desc="mcts iterations", unit="it")

    async def play_game(p0: Player, p1: Player):
        samples = await gen_game(
            game_id=GameID(0),
            eval_pos0=p0.eval_pos,
            eval_pos1=p1.eval_pos,
            exploration_constant=exploration_constant,
            mcts_iterations=mcts_iterations,
            submit_mcts_iter=lambda: mcts_pbar.update(1),
        )
        game_id, pos, policy, value = samples[0]
        game = TournamentGame(
            p0=p0,
            p1=p1,
            score=(value + 1) / 2,
            samples=samples,
        )
        tournament.games.append(game)

    coros = [play_game(p0, p1) for p0, p1 in pairings for _ in range(games_per_match)]
    await asyncio.gather(*coros)

    for player in players:
        player.close()

    return tournament

'''
