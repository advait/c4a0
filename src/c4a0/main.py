#!/usr/bin/env python

import hashlib
import math
import os
import pickle
import sys
import warnings
from pathlib import Path

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
    DEFAULT_BOOK,
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
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer  # noqa: E402
from c4a0.training import (  # noqa: E402
    GenConfig,
    TrainingGen,
    default_eval_config,
    parse_lr_schedule,
    training_loop,
)
from c4a0.utils import (  # noqa: E402
    get_torch_device,
    limit_cuda_memory,
    load_model,
)

app = typer.Typer()
FINAL_MODEL = Path(__file__).resolve().parents[2] / "models/c4a0-final.pkl"


@app.callback()
def _cap_gpu_memory() -> None:
    """Applies $C4A0_CUDA_MEMORY_FRACTION before any command touches CUDA, so `ceiling`, `eval`
    and `play` are capped like `train`."""
    if torch.cuda.is_available():
        limit_cuda_memory("cuda")


@app.command()
def train(
    base_dir: str = "training",
    device: str = "auto",
    n_self_play_games: int = 2048,
    n_mcts_iterations: int = 256,
    c_exploration: float = 2.0,
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 8,
    conv_filter_size: int = 64,
    value_hidden: int = typer.Option(256, min=1, help="Value head hidden width."),
    lr_schedule: list[float] = typer.Option(
        default_factory=lambda: [0, 2e-3, 10, 8e-4], show_default="0 2e-3 10 8e-4"
    ),
    l2_reg: float = 4e-4,
    max_gens: int | None = None,
    root_noise_alpha: float | None = 1.0,
    root_noise_frac: float = typer.Option(0.25, min=0.0, max=1.0),
    random_start_frac: float = typer.Option(
        0.5, min=0.0, max=1.0, help="Fraction of self-play games using random openings."
    ),
    random_start_max_ply: int = typer.Option(
        12, min=1, help="Maximum random opening length; capped at 41 plies."
    ),
    value_target_q_weight: float = typer.Option(1.0, min=0.0, max=1.0),
    leaf_minimax_depth: int = typer.Option(
        0,
        min=0,
        help="Plies of exact search on each MCTS leaf in self-play and eval; 0 disables.",
    ),
    replay_gens: int = typer.Option(8, min=1),
    train_max_epochs: int = typer.Option(
        20, min=1, help="Epoch cap with --train-anneal-epochs 0."
    ),
    train_patience: int = typer.Option(
        3, min=1, help="Early-stopping patience with --train-anneal-epochs 0."
    ),
    bf16: bool = typer.Option(False, help="BF16 autocast for self-play and training."),
    compile: bool = typer.Option(False, help="Compile the self-play network forward."),
    train_steps: int | None = typer.Option(
        None, min=1, help="Fixed optimizer steps; publish the final model."
    ),
    train_anneal_epochs: int = typer.Option(
        8,
        min=0,
        help="Train exactly this many epochs per generation, cosine-annealing the rate to "
        "1%. 0 trains at a constant rate with early stopping.",
    ),
    mirror_inference: bool = typer.Option(
        False,
        help="Average each network call with its left-right mirror (2x inference cost).",
    ),
    eval_mcts_iterations: list[int] = typer.Option(
        default_factory=lambda: [64, 256], show_default="64 256"
    ),
) -> None:
    """Trains a model via self-play. Each generation's saved model is scored on the v1 dev
    suite (raw network, plus MCTS at each --eval-mcts-iterations budget)."""

    if not math.isfinite(value_target_q_weight):
        raise typer.BadParameter(
            "must be finite and between 0 and 1", param_hint="--value-target-q-weight"
        )
    if not math.isfinite(random_start_frac):
        raise typer.BadParameter(
            "must be finite and between 0 and 1", param_hint="--random-start-frac"
        )
    model_config = ModelConfig(
        value_hidden=value_hidden,
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
        mirror_inference=mirror_inference,
    )

    config = GenConfig(
        value_hidden=value_hidden,
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        root_noise_alpha=root_noise_alpha,
        root_noise_frac=root_noise_frac,
        random_start_frac=random_start_frac,
        random_start_max_ply=random_start_max_ply,
        value_target_q_weight=value_target_q_weight,
        leaf_minimax_depth=leaf_minimax_depth,
        replay_gens=replay_gens,
        train_max_epochs=train_max_epochs,
        train_patience=train_patience,
        bf16=bf16,
        compile=compile,
        train_steps=train_steps,
        train_anneal_epochs=train_anneal_epochs or None,
    )
    training_loop(
        base_dir=base_dir,
        device=get_torch_device() if device == "auto" else torch.device(device),
        config=config,
        model_config=model_config,
        max_gens=max_gens,
        eval_config=default_eval_config(eval_mcts_iterations),
    )


@app.command()
def play(
    model: str = typer.Option(
        "final",
        help="`final` (the shipped models/c4a0-final.pkl), `latest` (your newest "
        "generation in --base-dir), `uniform`, `random`, or a model .pkl path.",
    ),
    base_dir: str = "training",
    max_mcts_iters: int = 1400,
    c_exploration: float = 2.0,
    c_ply_penalty: float = 0.01,
) -> None:
    """Play interactive games, by default against the shipped final model."""
    nn: ConnectFourNet | RandomPlayer | UniformPlayer
    if model == "latest":
        try:
            gen = TrainingGen.load_latest(base_dir)
        except FileNotFoundError:
            typer.echo(
                "No trained generations found. Run `mise run train` first.", err=True
            )
            raise typer.Exit(1) from None
        nn = gen.get_model(base_dir)
    elif model == "random":
        nn = RandomPlayer(ModelID(0))
    elif model == "uniform":
        nn = UniformPlayer(ModelID(0))
    else:
        nn = load_model(FINAL_MODEL if model == "final" else model)

    c4a0_rust.run_tui(
        lambda model_id, x: nn.forward_numpy(x),
        max_mcts_iters,
        c_exploration,
        c_ply_penalty,
    )


THREADS_HELP = "Solver threads; 0 uses every core."


@app.command()
def build_suite(
    threads: int = typer.Option(0, min=0, help=THREADS_HELP),
) -> None:
    """Rebuilds the committed v1 dev and test suites (deterministic: same bytes every run).
    Needs no opening book."""
    for split, recipe in SUITE_V1.items():
        labels = label_positions(recipe, threads=threads or None)
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
    book: Path = typer.Option(
        DEFAULT_BOOK,
        help="Native solver opening book, from `mise run solver:book`. Without it the "
        "early plies of a large job take hours.",
    ),
    threads: int = typer.Option(0, min=0, help=THREADS_HELP),
) -> None:
    """Samples and solver-labels positions in a split (train by default, for the ceiling)."""
    if not book.is_file():
        raise typer.BadParameter(
            f"{book} not found; build it with `mise run solver:book` (about 22 min on 32 cores)",
            param_hint="--book",
        )
    recipe = Recipe(
        version=1, split=split, seed=seed, per_ply=per_ply, plies=(first_ply, last_ply)
    )
    labels = label_positions(recipe, book, threads or None)
    write_labels(out, labels)
    logger.info("Wrote {} rows to {}", len(labels), out)


@app.command()
def ceiling(
    labels: Path,
    out_dir: Path,
    n_residual_blocks: int = 8,
    conv_filter_size: int = 64,
    value_hidden: int = typer.Option(256, min=1, help="Value head hidden width."),
    lr: float = 2e-3,
    l2_reg: float = 4e-4,
    batch_size: int = 4096,
    max_epochs: int = typer.Option(100, help="Epoch cap with --anneal-epochs 0."),
    patience: int = typer.Option(
        10, help="Early-stopping patience with --anneal-epochs 0."
    ),
    anneal_epochs: int = typer.Option(
        40,
        min=0,
        help="Train exactly this many epochs, cosine-annealing the rate to 1%. 0 trains at "
        "a constant rate, early-stopping on dev loss.",
    ),
    seed: int = 0,
    c_exploration: float = 2.0,
    c_ply_penalty: float = 0.01,
    mcts_iterations: list[int] = typer.Option(
        default_factory=list, show_default="none"
    ),
    report_test: bool = False,
) -> None:
    """
    Supervised ceiling: trains the self-play architecture on solver labels (train split only),
    cosine-annealed for --anneal-epochs (or early-stopping on dev loss with 0), and scores the saved model on dev (and test with --report-test,
    once the configuration is chosen). Writes model.pkl and report.json to out_dir.
    """
    train = read_labels(labels)
    if train.recipe.split != Split.TRAIN:
        raise typer.BadParameter(f"{labels} is a {train.recipe.split} file, not train")
    dev = load_suite(Split.DEV)
    torch.manual_seed(seed)
    anneal = anneal_epochs or None
    config = ModelConfig(
        value_hidden=value_hidden,
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
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
        anneal,
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
        parameter_count=sum(p.numel() for p in model.parameters()),
        train_labels=str(labels),
        train_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
        n_train=len(train),
        epochs=epochs,
        anneal_epochs=anneal,
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
    mcts_iterations: list[int] = typer.Option(
        default_factory=list, show_default="none"
    ),
) -> None:
    """Scores every published generation's model.pkl on a suite and stores it in the
    generation's evals (replacing any earlier score on that suite)."""
    suite = load_suite(split)
    for gen in reversed(TrainingGen.load_all(base_dir)):
        model_path = Path(gen.gen_folder(base_dir)) / "model.pkl"
        gen.evals[suite.name] = evaluate_model(
            model_path,
            suite,
            mcts_iterations,
            gen.c_exploration,
            gen.c_ply_penalty,
            leaf_minimax_depth=gen.leaf_minimax_depth,
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
