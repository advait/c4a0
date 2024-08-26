#!/usr/bin/env python

from typing import List
import warnings

import torch
import typer

from c4a0.nn import ModelConfig
from c4a0.sweep import perform_sweep
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer
from c4a0.training import TrainingGen, training_loop
from c4a0.utils import get_torch_device

import c4a0_rust  # type: ignore

app = typer.Typer()


@app.command()
def train(
    base_dir: str = "training",
    device: str = str(get_torch_device()),
    n_self_play_games: int = 2000,
    n_mcts_iterations: int = 150,
    c_exploration: float = 1.4,
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: List[float] = [0, 2e-3, 10, 8e-4],
    l2_reg: float = 4e-4,
):
    """Trains a model via self-play."""
    assert len(lr_schedule) % 2 == 0, "lr_schedule must have an even number of elements"
    schedule = {}
    for i in range(0, len(lr_schedule), 2):
        threshold = int(lr_schedule[i])
        assert (
            threshold == lr_schedule[i]
        ), "lr_schedule must alternate between gen_id (int) and lr (float)"
        lr = lr_schedule[i + 1]
        schedule[threshold] = lr

    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=schedule,
        l2_reg=l2_reg,
    )
    training_loop(
        base_dir=base_dir,
        device=torch.device(device),
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        model_config=model_config,
    )


@app.command()
def sweep(base_dir: str = "training"):
    """Perofrms a hyperparameter sweep."""
    perform_sweep(base_dir)


@app.command()
def ui(
    base_dir: str = "training",
    max_mcts_iters: int = 100,
    c_exploration: float = 1.4,
    c_ply_penalty: float = 0.01,
    model: str = "best",
):
    """Play interactive games"""
    gen = TrainingGen.load_latest(base_dir)
    if model == "best":
        nn = gen.get_model(base_dir)
    elif model == "random":
        nn = RandomPlayer(ModelID(0))
    elif model == "uniform":
        nn = UniformPlayer(ModelID(0))
    else:
        raise ValueError(f"unrecognized model: {model}")

    c4a0_rust.run_tui(  # type: ignore
        lambda model_id, x: nn.forward_numpy(x),
        max_mcts_iters,
        c_exploration,
        c_ply_penalty,
    )


@app.command()
def score(
    base_dir: str = "training",
    solver_path: str = "/home/advait/connect4/c4solver",
    book_path: str = "/home/advait/connect4/7x6.book",
    cache_path: str = "./solution_cache.pkl",
):
    """Score the model"""
    gens = TrainingGen.load_all(base_dir)
    for gen in gens:
        print(f"Getting games for: {gen.gen_n}")
        games = gen.get_games(base_dir)  # type: ignore
        if not games:
            continue
        print(f"Scoring: {gen.gen_n}")
        score = games.score_policies(solver_path, book_path, cache_path)  # type: ignore
        print(f"Score: {score}")


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    app()
