#!/usr/bin/env python

import warnings

import torch
import typer

from c4a0.nn import ModelConfig
from c4a0.sweep import perform_sweep
from c4a0.training import training_loop
from c4a0.utils import get_torch_device

app = typer.Typer()


@app.command()
def train(
    base_dir: str = "training",
    device: str = str(get_torch_device()),
    n_self_play_games: int = 2000,
    n_mcts_iterations: int = 150,
    exploration_constant: float = 1.4,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    learning_rate: float = 2e-3,
    l2_reg: float = 4e-4,
):
    """Trains a model via self-play."""
    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        learning_rate=learning_rate,
        l2_reg=l2_reg,
    )
    training_loop(
        base_dir=base_dir,
        device=torch.device(device),
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        exploration_constant=exploration_constant,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        model_config=model_config,
    )


@app.command()
def sweep(base_dir: str = "training"):
    """Perofrms a hyperparameter sweep."""
    perform_sweep(base_dir)


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    app()
