#!/usr/bin/env python
"""Self-play-only training metric judged by Pascal Pons's Connect Four solver.

The solver is deliberately used only after training, as an evaluation judge. It must
never provide moves, values, samples, replay data, or loss targets to training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import torch

from c4a0.nn import ModelConfig
from c4a0.training import TrainingGen, parse_lr_schedule, training_loop

import c4a0_rust  # type: ignore


@dataclass(frozen=True)
class EvalConfig:
    train_gens: int
    train_games: int
    train_mcts: int
    eval_games: int
    eval_mcts: int
    self_play_batch_size: int
    training_batch_size: int
    c_exploration: float
    c_ply_penalty: float
    n_residual_blocks: int
    conv_filter_size: int
    n_policy_layers: int
    n_value_layers: int
    learning_rate: float
    l2_reg: float
    eval_game_id_offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="solver/c4solver")
    parser.add_argument("--book", default="solver/7x6.book")
    parser.add_argument("--cache", default="solutions-autoresearch.db")
    parser.add_argument("--output-dir", default="autoresearch/eval-runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-gens", type=int, default=3)
    parser.add_argument("--train-games", type=int, default=64)
    parser.add_argument("--train-mcts", type=int, default=48)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--eval-mcts", type=int, default=48)
    parser.add_argument("--self-play-batch-size", type=int, default=128)
    parser.add_argument("--training-batch-size", type=int, default=128)
    parser.add_argument("--c-exploration", type=float, default=6.6)
    parser.add_argument("--c-ply-penalty", type=float, default=0.01)
    parser.add_argument("--n-residual-blocks", type=int, default=1)
    parser.add_argument("--conv-filter-size", type=int, default=16)
    parser.add_argument("--n-policy-layers", type=int, default=1)
    parser.add_argument("--n-value-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--eval-game-id-offset", type=int, default=1_000_000)
    return parser.parse_args()


def make_run_dir(output_dir: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"solver-alignment-{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def model_config(config: EvalConfig) -> ModelConfig:
    return ModelConfig(
        n_residual_blocks=config.n_residual_blocks,
        conv_filter_size=config.conv_filter_size,
        n_policy_layers=config.n_policy_layers,
        n_value_layers=config.n_value_layers,
        lr_schedule=parse_lr_schedule([0, config.learning_rate]),
        l2_reg=config.l2_reg,
    )


def train_self_play_only(
    run_dir: Path, config: EvalConfig, device: torch.device
) -> TrainingGen:
    """Train from self-play data only. Solver output is intentionally unavailable here."""
    return training_loop(
        base_dir=str(run_dir / "training"),
        device=device,
        n_self_play_games=config.train_games,
        n_mcts_iterations=config.train_mcts,
        c_exploration=config.c_exploration,
        c_ply_penalty=config.c_ply_penalty,
        self_play_batch_size=config.self_play_batch_size,
        training_batch_size=config.training_batch_size,
        model_config=model_config(config),
        max_gens=config.train_gens,
        solver_config=None,
    )


def generate_eval_games(
    run_dir: Path,
    gen: TrainingGen,
    config: EvalConfig,
    device: torch.device,
) -> c4a0_rust.PlayGamesResult:  # type: ignore[name-defined]
    model = gen.get_model(str(run_dir / "training"))
    model.to(device)
    model.eval()
    reqs = [
        c4a0_rust.GameMetadata(config.eval_game_id_offset + game_id, 0, 0)  # type: ignore[attr-defined]
        for game_id in range(config.eval_games)
    ]
    return c4a0_rust.play_games(  # type: ignore[attr-defined]
        reqs,
        config.self_play_batch_size,
        config.eval_mcts,
        config.c_exploration,
        config.c_ply_penalty,
        lambda model_id, pos: model.forward_numpy(pos),
    )


def score_alignment(
    games: c4a0_rust.PlayGamesResult,  # type: ignore[name-defined]
    solver_path: str,
    book_path: str,
    cache_path: Path,
) -> float:
    return float(games.score_policies(solver_path, book_path, str(cache_path)))


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        train_gens=args.train_gens,
        train_games=args.train_games,
        train_mcts=args.train_mcts,
        eval_games=args.eval_games,
        eval_mcts=args.eval_mcts,
        self_play_batch_size=args.self_play_batch_size,
        training_batch_size=args.training_batch_size,
        c_exploration=args.c_exploration,
        c_ply_penalty=args.c_ply_penalty,
        n_residual_blocks=args.n_residual_blocks,
        conv_filter_size=args.conv_filter_size,
        n_policy_layers=args.n_policy_layers,
        n_value_layers=args.n_value_layers,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
        eval_game_id_offset=args.eval_game_id_offset,
    )


def write_metrics(
    run_dir: Path,
    config: EvalConfig,
    gen: TrainingGen,
    games: c4a0_rust.PlayGamesResult,  # type: ignore[name-defined]
    metric: float,
    solver_path: str,
    book_path: str,
    cache_path: Path,
) -> None:
    metadata = {
        "metric": metric,
        "metric_name": "solver_alignment_mcts_top_move",
        "metric_direction": "higher_is_better",
        "solver_is_eval_only": True,
        "solver_path": solver_path,
        "book_path": book_path,
        "cache_path": str(cache_path),
        "config": asdict(config),
        "final_generation": gen.model_dump(mode="json"),
        "eval_games": len(games.results),
        "eval_unique_positions": games.unique_positions(),
        "training_dir": str(run_dir / "training"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with open(run_dir / "eval_games.pkl", "wb") as f:
        import pickle

        pickle.dump(games, f)


def main() -> int:
    args = parse_args()
    solver = Path(args.solver)
    book = Path(args.book)
    if not solver.exists() or not solver.is_file():
        print(f"solver not found: {solver}", file=sys.stderr)
        return 2
    if not book.exists() or not book.is_file():
        print(f"book not found: {book}", file=sys.stderr)
        return 2

    config = build_config(args)
    run_dir = make_run_dir(args.output_dir)
    device = torch.device(args.device)
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = run_dir / cache_path

    gen = train_self_play_only(run_dir, config, device)
    games = generate_eval_games(run_dir, gen, config, device)
    metric = score_alignment(games, str(solver), str(book), cache_path)
    write_metrics(
        run_dir, config, gen, games, metric, str(solver), str(book), cache_path
    )

    print(f"{metric:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
