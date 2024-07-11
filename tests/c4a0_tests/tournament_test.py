import pytest
import torch

from c4a0.nn import ConnectFourNet
from c4a0.tournament import (
    play_tournament,
    ModelPlayer,
    RandomPlayer,
    UniformPlayer,
    ModelID,
)


@pytest.mark.asyncio
async def test_tournament():
    model = ConnectFourNet()
    model.eval()  # Disable batch normalization
    model_player = ModelPlayer(
        model_id=ModelID(0),
        model=model,
        device=torch.device("cpu"),
    )
    players = [model_player, RandomPlayer(), UniformPlayer()]

    tournament = await play_tournament(
        players=players,
        games_per_match=2,
        mcts_iterations=10,
        exploration_constant=1.4,
    )

    assert len(tournament.games) == 2 * 2 * 3
    for game in tournament.games:
        assert game.p0 in players
        assert game.p1 in players
        assert 0 <= game.score <= 1
