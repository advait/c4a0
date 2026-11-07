#!/usr/bin/env python

import hashlib
import math
import os
import pickle
import sys
import warnings
from pathlib import Path

import optuna
import torch
import typer
from loguru import logger

# Ensure that the parent directory of this file exists on Python path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import c4a0_rust  # noqa: E402
from c4a0.evaluation import SuiteEval, evaluate_model  # noqa: E402
from c4a0.labels import (  # noqa: E402
    SUITE_V1,
    Recipe,
    Split,
    load_suite,
    read_labels,
    suite_path,
    write_labels,
)
from c4a0.labels import (  # noqa: E402
    label as label_positions,
)
from c4a0.nn import ConnectFourNet, ModelConfig  # noqa: E402
from c4a0.supervised import TARGET_VERSION, CeilingReport, fit_supervised  # noqa: E402
from c4a0.sweep import perform_hparam_sweep  # noqa: E402
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer  # noqa: E402
from c4a0.training import (  # noqa: E402
    GenConfig,
    TrainingGen,
    default_eval_config,
    parse_lr_schedule,
    parse_temperature_schedule,
    training_loop,
)
from c4a0.utils import get_torch_device  # noqa: E402

app = typer.Typer()


@app.command()
def train(
    base_dir: str = "training",
    device: str = str(get_torch_device()),
    # These parameters were chosen based on the results of the nn_sweep and mcts_sweep
    n_self_play_games: int = 1700,
    n_mcts_iterations: int = 1400,
    c_exploration: float = 2.0,
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: list[float] = typer.Option(
        default_factory=lambda: [0, 2e-3, 10, 8e-4]
    ),
    l2_reg: float = 4e-4,
    max_gens: int | None = None,
    greedy_after_ply: int | None = typer.Option(None, min=0),
    root_noise_alpha: float | None = None,
    root_noise_frac: float = typer.Option(0.0, min=0.0, max=1.0),
    value_target_q_weight: float = typer.Option(0.0, min=0.0, max=1.0),
    temperature_schedule: str = "4:4.0,8:2.0",
    mask_terminal_policy: bool = False,
    replay_gens: int = typer.Option(1, min=1),
    eval_mcts_iterations: list[int] = typer.Option(default_factory=list),
) -> None:
    """Trains a model via self-play. Each generation's saved model is scored on the v1 dev
    suite (raw network, plus MCTS at each --eval-mcts-iterations budget)."""

    if not math.isfinite(value_target_q_weight):
        raise typer.BadParameter(
            "must be finite and between 0 and 1", param_hint="--value-target-q-weight"
        )
    try:
        parsed_temperature_schedule = parse_temperature_schedule(temperature_schedule)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--temperature-schedule") from exc
    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
    )

    config = GenConfig(
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        greedy_after_ply=greedy_after_ply,
        root_noise_alpha=root_noise_alpha,
        root_noise_frac=root_noise_frac,
        value_target_q_weight=value_target_q_weight,
        temperature_schedule=tuple(parsed_temperature_schedule),
        mask_terminal_policy=mask_terminal_policy,
        replay_gens=replay_gens,
    )
    training_loop(
        base_dir=base_dir,
        device=torch.device(device),
        config=config,
        model_config=model_config,
        max_gens=max_gens,
        eval_config=default_eval_config(eval_mcts_iterations),
    )


@app.command()
def play(
    base_dir: str = "training",
    max_mcts_iters: int = 1400,
    c_exploration: float = 2.0,
    c_ply_penalty: float = 0.01,
    model: str = "best",
) -> None:
    """Play interactive games"""
    gen = TrainingGen.load_latest(base_dir)
    nn: ConnectFourNet | RandomPlayer | UniformPlayer
    if model == "best":
        nn = gen.get_model(base_dir)
    elif model == "random":
        nn = RandomPlayer(ModelID(0))
    elif model == "uniform":
        nn = UniformPlayer(ModelID(0))
    else:
        raise ValueError(f"unrecognized model: {model}")

    c4a0_rust.run_tui(
        lambda model_id, x: nn.forward_numpy(x),
        max_mcts_iters,
        c_exploration,
        c_ply_penalty,
    )


@app.command()
def nn_sweep(base_dir: str = "training") -> None:
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
    # These NN parameters were chosen based on the results of the nn_sweep
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: list[float] = typer.Option(default_factory=lambda: [0, 2e-3]),
    l2_reg: float = 4e-4,
    # End NN parameters
    base_training_dir: str = "training-sweeps",
    optuna_db_path: str = "optuna.db",
    n_trials: int = 100,
    max_gens_per_trial: int = 10,
) -> None:
    """
    Performs sweep of MCTS hyperparameters (e.g. n_self_play_games, n_mcts_iterations,
    c_exploration) to determine optimal values by performing `n_trials` independent training
    runs, each with `max_gens_per_trial` generations, seeking to maximize the final generation's
    raw-network top-1 agreement with solver-best moves on the v1 dev suite.
    """
    eval_config = default_eval_config()
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

    def objective(trial: optuna.Trial) -> float:
        trial_path = base_path / f"trial_{trial.number}"
        trial_path.mkdir(exist_ok=False)
        gen = training_loop(
            base_dir=str(trial_path),
            device=torch.device(device),
            config=GenConfig(
                n_self_play_games=trial.suggest_int("n_self_play_games", 1000, 5000),
                n_mcts_iterations=trial.suggest_int("n_mcts_iterations", 100, 1500),
                c_exploration=trial.suggest_float("c_exploration", 0.5, 12.0),
                c_ply_penalty=c_ply_penalty,
                self_play_batch_size=self_play_batch_size,
                training_batch_size=training_batch_size,
            ),
            model_config=model_config,
            max_gens=max_gens_per_trial,
            eval_config=eval_config,
        )
        score = gen.evals[eval_config.suite.name].raw.top1
        logger.info("Trial {} completed. Dev raw top1: {}", trial.number, score)
        assert score is not None
        return score

    storage_name = f"sqlite:///{optuna_db_path}"
    study = optuna.create_study(
        # The objective changed to saved-child dev top-1 (v1); never resume legacy studies.
        study_name="mcts_sweep-v1-dev-top1",
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)


@app.command()
def build_suite(
    solver_path: str = "solver/c4solver",
    book_path: str = "solver/7x6.book",
    n_procs: int = 32,
) -> None:
    """Rebuilds the committed v1 dev and test suites (deterministic: same bytes every run)."""
    for split, recipe in SUITE_V1.items():
        labels = label_positions(recipe, solver_path, book_path, n_procs)
        path = suite_path(split)
        write_labels(path, labels)
        logger.info("Wrote {} rows to {}", len(labels), path)


@app.command()
def label(
    out: Path,
    split: Split = Split.TRAIN,
    per_ply: int = 24000,
    seed: int = 7,
    first_ply: int = 0,
    last_ply: int = 41,
    solver_path: str = "solver/c4solver",
    book_path: str = "solver/7x6.book",
    n_procs: int = 32,
) -> None:
    """Samples and solver-labels positions in a split (train by default, for the ceiling)."""
    recipe = Recipe(
        version=1, split=split, seed=seed, per_ply=per_ply, plies=(first_ply, last_ply)
    )
    labels = label_positions(recipe, solver_path, book_path, n_procs)
    write_labels(out, labels)
    logger.info("Wrote {} rows to {}", len(labels), out)


@app.command()
def ceiling(
    labels: Path,
    out_dir: Path,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr: float = 2e-3,
    l2_reg: float = 4e-4,
    batch_size: int = 4096,
    max_epochs: int = 100,
    patience: int = 10,
    seed: int = 0,
    c_exploration: float = 2.0,
    c_ply_penalty: float = 0.01,
    mcts_iterations: list[int] = typer.Option(default_factory=list),
    report_test: bool = False,
) -> None:
    """
    Supervised ceiling: trains the self-play architecture on solver labels (train split only),
    early-stopping on dev loss, and scores the saved model on dev (and test with --report-test,
    once the configuration is chosen). Writes model.pkl and report.json to out_dir.
    """
    train = read_labels(labels)
    if train.recipe.split != Split.TRAIN:
        raise typer.BadParameter(f"{labels} is a {train.recipe.split} file, not train")
    dev = load_suite(Split.DEV)
    torch.manual_seed(seed)
    config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule={0: lr},
        l2_reg=l2_reg,
    )
    model, epochs = fit_supervised(
        ConnectFourNet(config),
        train,
        dev.labels,
        c_ply_penalty,
        batch_size,
        max_epochs,
        patience,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    def score(split: Split) -> SuiteEval:
        return evaluate_model(
            model_path, load_suite(split), mcts_iterations, c_exploration, c_ply_penalty
        )

    report = CeilingReport(
        net=config,
        train_labels=str(labels),
        train_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
        n_train=len(train),
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        c_ply_penalty=c_ply_penalty,
        target_version=TARGET_VERSION,
        dev=score(Split.DEV),
        test=score(Split.TEST) if report_test else None,
    )
    tmp = out_dir / "report.json.tmp"
    tmp.write_text(report.model_dump_json(indent=2))
    os.replace(tmp, out_dir / "report.json")
    logger.info(
        "Ceiling dev: raw top1 {}, mcts top1 {}, value accuracy {}",
        report.dev.raw.top1,
        {n: e.top1 for n, e in report.dev.mcts.items()},
        report.dev.value.accuracy,
    )


@app.command("eval")
def eval_gens(
    base_dir: str = "training",
    split: Split = Split.DEV,
    mcts_iterations: list[int] = typer.Option(default_factory=list),
) -> None:
    """Scores every published generation's model.pkl on a suite and stores it in the
    generation's evals (replacing any earlier score on that suite)."""
    suite = load_suite(split)
    for gen in reversed(TrainingGen.load_all(base_dir)):
        model_path = Path(gen.gen_folder(base_dir)) / "model.pkl"
        gen.evals[suite.name] = evaluate_model(
            model_path, suite, mcts_iterations, gen.c_exploration, gen.c_ply_penalty
        )
        gen.save_metadata(base_dir)
        e = gen.evals[suite.name]
        logger.info(
            "Gen {}: raw top1 {}, mcts top1 {}",
            gen.gen_n,
            e.raw.top1,
            {n: m.top1 for n, m in e.mcts.items()},
        )


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    app()
