import torch

from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.tournament import (
    play_tournament,
    ModelPlayer,
    RandomPlayer,
    UniformPlayer,
    ModelID,
)
from c4a0.training import parse_lr_schedule


def make_model() -> ConnectFourNet:
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=8,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule=parse_lr_schedule([0, 1e-3]),
            l2_reg=0.0,
        )
    )


def test_tournament():
    model = make_model()
    model.eval()  # Disable batch normalization
    model_player = ModelPlayer(
        model_id=ModelID(0),
        model=model,
        device=torch.device("cpu"),
    )
    players = [model_player, RandomPlayer(ModelID(1)), UniformPlayer(ModelID(2))]

    tournament = play_tournament(
        players=players,
        games_per_match=2,
        batch_size=16,
        mcts_iterations=2,
        exploration_constant=1.4,
    )

    assert tournament.games is not None
    assert len(tournament.games.results) == 2 * 3
    player_ids = {player.model_id for player in players}
    for game in tournament.games.results:
        assert game.metadata.player0_id in player_ids
        assert game.metadata.player1_id in player_ids
        assert 0 <= game.player0_score() <= 1
