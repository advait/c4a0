"""
Round-robin tournament to determine which model is the best.
"""

import abc
import itertools
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import NewType

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from tabulate import tabulate

import c4a0_rust
from c4a0.nn import ConnectFourNet
from c4a0_rust import N_COLS

PlayerName = NewType("PlayerName", str)

ModelID = NewType("ModelID", int)


class Player(abc.ABC):
    name: PlayerName
    model_id: ModelID

    def __init__(self, name: str, model_id: ModelID) -> None:
        self.name = PlayerName(name)
        self.model_id = model_id

    def forward_numpy(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        raise NotImplementedError


class ModelPlayer(Player):
    """Player whose policy and value are determined by a ConnectFourNet."""

    model: ConnectFourNet
    device: torch.device

    def __init__(
        self, model_id: ModelID, model: ConnectFourNet, device: torch.device
    ) -> None:
        super().__init__(f"gen{model_id}", model_id)
        self.model_id = model_id
        self.model = model
        self.model.to(device)

    def forward_numpy(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        return self.model.forward_numpy(x)


class RandomPlayer(Player):
    """Player that provides a random policy and value."""

    def __init__(self, model_id: ModelID) -> None:
        super().__init__("random", model_id)

    def forward_numpy(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        batch_size = x.shape[0]
        policy_logits = torch.rand(batch_size, N_COLS).numpy()
        q_value = (torch.rand(batch_size) * 2 - 1).numpy()  # [-1, 1]
        return policy_logits, q_value, q_value


class UniformPlayer(Player):
    """Player that provides a uniform policy and 0 value."""

    def __init__(self, model_id: ModelID) -> None:
        super().__init__("uniform", model_id)

    def forward_numpy(
        self, x: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        batch_size = x.shape[0]
        policy_logits = torch.ones(batch_size, N_COLS).numpy()
        q_value = torch.zeros(batch_size).numpy()
        return policy_logits, q_value, q_value


@dataclass
class TournamentResult:
    """Represents the results from a tournamnet."""

    model_ids: list[ModelID]
    date: datetime = field(default_factory=datetime.now)
    games: c4a0_rust.PlayGamesResult | None = None

    def get_scores(self) -> list[tuple[ModelID, float]]:
        assert self.games is not None, "tournament has not been played"
        scores: dict[ModelID, float] = defaultdict(lambda: 0.0)
        for result in self.games.results:
            player0_score = result.player0_score()
            scores[ModelID(result.metadata.player0_id)] += player0_score
            scores[ModelID(result.metadata.player1_id)] += 1 - player0_score

        ret = list(scores.items())
        ret.sort(key=lambda x: x[1], reverse=True)
        return ret

    def scores_table(self, get_name: Callable[[int], str]) -> str:
        return tabulate(
            [(get_name(id), score) for id, score in self.get_scores()],
            headers=["Player", "Score"],
            tablefmt="github",
        )

    def get_top_models(self) -> list[ModelID]:
        """Returns the top models from the tournament in descending order of performance."""
        return [model_id for model_id, _ in self.get_scores()]


def play_tournament(
    players: Sequence[Player],
    games_per_match: int,
    batch_size: int,
    mcts_iterations: int,
    exploration_constant: float,
    c_ply_penalty: float = 0.01,
) -> TournamentResult:
    """Plays a round-robin tournament, returning the total score of each player."""
    assert games_per_match % 2 == 0, "games_per_match must be even"

    gen_id_to_player = {player.model_id: player for player in players}
    tournament = TournamentResult(
        model_ids=[player.model_id for player in players],
    )
    player_ids = [player.model_id for player in players]
    pairings = list(itertools.permutations(player_ids, 2)) * int(games_per_match / 2)
    reqs = [c4a0_rust.GameMetadata(id, p0, p1) for id, (p0, p1) in enumerate(pairings)]

    logger.info(f"Beginning tournament with {len(players)} players")
    tournament.games = c4a0_rust.play_games(
        reqs,
        batch_size,
        mcts_iterations,
        exploration_constant,
        c_ply_penalty,
        lambda player_id, pos: gen_id_to_player[ModelID(player_id)].forward_numpy(pos),
        greedy_after_ply=None,
        root_noise_alpha=None,
        root_noise_frac=0.0,
        value_target_q_weight=0.0,
    )
    logger.info(f"Finished tournament with {len(tournament.games.results)} games")

    return tournament
