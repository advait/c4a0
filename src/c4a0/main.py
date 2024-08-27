#!/usr/bin/env python

from pathlib import Path
from typing import List
import warnings

from loguru import logger
import optuna
import torch
import typer

from c4a0.nn import ModelConfig
from c4a0.sweep import perform_hparam_sweep
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer
from c4a0.training import SolverConfig, TrainingGen, parse_lr_schedule, training_loop
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
    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
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
def nn_sweep(base_dir: str = "training"):
    """
    Performs a hyperparameter sweep to determine best nn model params based on existing training
    data.
    """
    perform_hparam_sweep(base_dir)


@app.command()
def mcts_sweep(
    device: str = str(get_torch_device()),
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: List[float] = [0, 2e-3],
    l2_reg: float = 4e-4,
    base_training_dir: str = "training-sweeps",
    optuna_db_path: str = "optuna.db",
    n_trials: int = 100,
    max_gens_per_trial: int = 10,
    solver_path: str = "/home/advait/connect4/c4solver",
    book_path: str = "/home/advait/connect4/7x6.book",
    solutions_path: str = "./solutions.db",
):
    """
    Performs sweep of MCTS hyperparameters (e.g. n_self_play_games, n_mcts_iterations,
    c_exploration) to determine optimal values by performing `n_trials` independent training
    runs, each with `max_gens_per_trial` generations, seeking to maximize the solver score.
    """
    base_path = Path(base_training_dir)
    base_path.mkdir(exist_ok=True)

    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
    )

    def objective(trial: optuna.Trial):
        trial_path = base_path / f"trial_{trial.number}"
        trial_path.mkdir(exist_ok=False)
        gen = training_loop(
            base_dir=str(trial_path),
            device=torch.device(device),
            n_self_play_games=trial.suggest_int("n_self_play_games", 1000, 5000),
            n_mcts_iterations=trial.suggest_int("n_mcts_iterations", 100, 1500),
            c_exploration=trial.suggest_float("c_exploration", 0.5, 12.0),
            c_ply_penalty=c_ply_penalty,
            self_play_batch_size=self_play_batch_size,
            training_batch_size=training_batch_size,
            model_config=model_config,
            max_gens=max_gens_per_trial,
            solver_config=SolverConfig(
                solver_path=solver_path,
                book_path=book_path,
                solutions_path=solutions_path,
            ),
        )
        logger.info(
            "Trial {} completed. Solver score: {}", trial.number, gen.solver_score
        )
        score = gen.solver_score
        assert score is not None
        return score

    storage_name = f"sqlite:///{optuna_db_path}"
    study = optuna.create_study(
        study_name="mcts_sweep",
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)


@app.command()
def score(
    base_dir: str = "training",
    solver_path: str = "/home/advait/connect4/c4solver",
    book_path: str = "/home/advait/connect4/7x6.book",
    solutions_path: str = "./solutions.db",
):
    """Score the model"""
    gens = TrainingGen.load_all(base_dir)
    for gen in gens:
        print(f"Getting games for: {gen.gen_n}")
        games = gen.get_games(base_dir)  # type: ignore
        if not games:
            continue
        print(f"Scoring: {gen.gen_n}")
        score = games.score_policies(solver_path, book_path, solutions_path)  # type: ignore
        print(f"Score: {score}")


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    app()
