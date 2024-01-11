"""
Round-robin tournament to determine which model is the best.
"""

import abc
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
import itertools
import logging
from typing import Dict, Iterator, List, NewType, Optional, Tuple
import numpy as np

import torch
from tqdm import tqdm
from c4 import N_COLS, Pos

from nn import ConnectFourNet, EvalPos, Policy, Value, create_batcher
from self_play import GameID, Sample, gen_game

PlayerName = NewType("PlayerName", str)

GenID = NewType("GenID", int)


class Player(abc.ABC):
    name: PlayerName

    def __init__(self, name: str):
        self.name = PlayerName(name)

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        raise NotImplementedError

    def close(self):
        pass


class ModelPlayer(Player):
    """Player whose policy and value are determined by a ConnectFourNet."""

    gen_id: GenID
    model: ConnectFourNet
    device: torch.device

    _end_signal: Optional[asyncio.Future] = None
    _enqueue_pos: Optional[EvalPos] = None

    def __init__(self, gen_id: GenID, model: ConnectFourNet, device: torch.device):
        super().__init__(f"gen{gen_id}")
        self.gen_id = gen_id
        self.model = model
        self.model.to(device)
        self.device = device

    def close(self):
        if self._end_signal is not None:
            self._end_signal.set_result(True)

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        if self._enqueue_pos is None:
            self._end_signal, self._enqueue_pos = create_batcher(
                self.model, self.device
            )
        return await self._enqueue_pos(pos)


class RandomPlayer(Player):
    """Player that provides a random policy and value."""

    def __init__(self):
        super().__init__("random")

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        policy_logits = np.random.rand(N_COLS)
        policy_exp = np.exp(policy_logits)
        policy = policy_exp / policy_exp.sum()
        value = Value(np.random.rand() * 2 - 1)  # [-1, 1]
        return policy, value


class UniformPlayer(Player):
    """Player that provides a uniform policy and 0 value."""

    def __init__(self):
        super().__init__("uniform")

    async def eval_pos(self, pos: Pos) -> Tuple[Policy, Value]:
        policy = Policy(np.ones(N_COLS) / N_COLS)
        value = Value(0.0)
        return policy, value


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

    def get_scores(self) -> List[Tuple[Player, float]]:
        scores: Dict[PlayerName, float] = defaultdict(lambda: 0.0)
        for game in self.games:
            scores[game.p0.name] += game.score
            scores[game.p1.name] += 1 - game.score
        ret = [(self.players[name], score) for name, score in scores.items()]
        ret.sort(key=lambda x: x[1], reverse=True)
        return ret

    def scores_table(self) -> str:
        return "\n".join(
            f"{player.name}: {score}" for player, score in self.get_scores()
        )

    def get_top_models(self) -> Iterator[GenID]:
        """Returns the top models from the tournament in descending order of performance."""
        logger = logging.getLogger(__name__)
        scores = self.get_scores()
        for player, score in scores:
            if not isinstance(player, ModelPlayer):
                logger.warning(f"Top player is not a model: {player.name}")
                continue
            yield player.gen_id
        raise ValueError("No models found in tournament results")

    def get_top_model(self) -> GenID:
        return next(self.get_top_models())


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
