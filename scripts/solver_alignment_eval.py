#!/usr/bin/env python
"""Self-play-only training metric judged by Pascal Pons's Connect Four solver.

The solver is deliberately used only after training, as an evaluation judge. It must
never provide moves, values, samples, replay data, or loss targets to training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import uuid

import numpy as np
import torch

from c4a0.nn import ModelConfig
from c4a0.training import TrainingGen, parse_lr_schedule, training_loop

import c4a0_rust  # type: ignore


@dataclass(frozen=True)
class BenchmarkTier:
    """Resolved scale defaults for one solver-alignment benchmark tier."""

    train_gens: int
    train_games: int
    train_mcts: int
    eval_games: int
    eval_mcts: int
    self_play_batch_size: int
    training_batch_size: int
    replay_window: int
    eval_opening_depth: int
    select_best_generation_by_solver: bool
    selection_eval_games: int
    selection_eval_mcts: int
    description: str


BENCHMARK_TIERS: dict[str, BenchmarkTier] = {
    "smoke": BenchmarkTier(
        train_gens=1,
        train_games=16,
        train_mcts=16,
        eval_games=128,
        eval_mcts=32,
        self_play_batch_size=64,
        training_batch_size=64,
        replay_window=1,
        eval_opening_depth=2,
        select_best_generation_by_solver=False,
        selection_eval_games=16,
        selection_eval_mcts=16,
        description="Fast local sanity check; not a convergence signal.",
    ),
    "dev": BenchmarkTier(
        train_gens=5,
        train_games=512,
        train_mcts=128,
        eval_games=2_048,
        eval_mcts=128,
        self_play_batch_size=256,
        training_batch_size=256,
        replay_window=1,
        eval_opening_depth=6,
        select_best_generation_by_solver=True,
        selection_eval_games=512,
        selection_eval_mcts=128,
        description="Moderate iteration signal for candidate development.",
    ),
    "candidate": BenchmarkTier(
        train_gens=15,
        train_games=2_048,
        train_mcts=384,
        eval_games=10_000,
        eval_mcts=384,
        self_play_batch_size=512,
        training_batch_size=512,
        replay_window=1,
        eval_opening_depth=6,
        select_best_generation_by_solver=True,
        selection_eval_games=2_048,
        selection_eval_mcts=256,
        description="Primary autoresearch acceptance tier with thousands of games.",
    ),
    "gate": BenchmarkTier(
        train_gens=30,
        train_games=5_000,
        train_mcts=800,
        eval_games=50_000,
        eval_mcts=800,
        self_play_batch_size=1_024,
        training_batch_size=1_024,
        replay_window=1,
        eval_opening_depth=6,
        select_best_generation_by_solver=True,
        selection_eval_games=5_000,
        selection_eval_mcts=512,
        description="Expensive stability gate before treating a change as durable.",
    ),
    "long": BenchmarkTier(
        train_gens=50,
        train_games=10_000,
        train_mcts=1_600,
        eval_games=100_000,
        eval_mcts=1_600,
        self_play_batch_size=2_048,
        training_batch_size=2_048,
        replay_window=1,
        eval_opening_depth=6,
        select_best_generation_by_solver=True,
        selection_eval_games=10_000,
        selection_eval_mcts=800,
        description="Long-run convergence tier; intended for overnight/multi-day runs.",
    ),
}


@dataclass(frozen=True)
class EvalConfig:
    benchmark_tier: str
    train_gens: int
    train_games: int
    train_mcts: int
    eval_games: int
    eval_mcts: int
    self_play_batch_size: int
    training_batch_size: int
    replay_window: int
    c_exploration: float
    c_ply_penalty: float
    n_residual_blocks: int
    conv_filter_size: int
    n_policy_layers: int
    n_value_layers: int
    learning_rate: float
    l2_reg: float
    policy_loss_weight: float
    q_penalty_loss_weight: float
    q_no_penalty_loss_weight: float
    eval_game_id_offset: int
    eval_temperature: float | None
    eval_opening_depth: int
    seed: int
    select_best_generation_by_solver: bool
    selection_eval_games: int
    selection_eval_mcts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="solver/c4solver")
    parser.add_argument("--book", default="solver/7x6.book")
    parser.add_argument(
        "--cache",
        default="autoresearch/solver-cache/top-move-solutions.db",
        help="Stable RocksDB solver-cache path. Relative paths resolve from repo cwd.",
    )
    parser.add_argument(
        "--run-local-cache",
        action="store_true",
        help="Resolve a relative --cache under this eval run directory instead of repo cwd.",
    )
    parser.add_argument("--output-dir", default="autoresearch/eval-runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tier",
        "--benchmark-tier",
        dest="benchmark_tier",
        choices=sorted(BENCHMARK_TIERS),
        default="candidate",
        help="Scale preset. Explicit scalar args override individual preset values.",
    )
    parser.add_argument("--train-gens", type=int, default=None)
    parser.add_argument("--train-games", type=int, default=None)
    parser.add_argument("--train-mcts", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=None)
    parser.add_argument("--eval-mcts", type=int, default=None)
    parser.add_argument("--self-play-batch-size", type=int, default=None)
    parser.add_argument("--training-batch-size", type=int, default=None)
    parser.add_argument("--replay-window", type=int, default=None)
    parser.add_argument("--c-exploration", type=float, default=6.6)
    parser.add_argument("--c-ply-penalty", type=float, default=0.0)
    parser.add_argument("--n-residual-blocks", type=int, default=1)
    parser.add_argument("--conv-filter-size", type=int, default=16)
    parser.add_argument("--n-policy-layers", type=int, default=1)
    parser.add_argument("--n-value-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--l2-reg", type=float, default=0.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--q-penalty-loss-weight", type=float, default=1.0)
    parser.add_argument("--q-no-penalty-loss-weight", type=float, default=1.0)
    parser.add_argument("--eval-game-id-offset", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=None,
        help="Override eval game move temperature; 0.0 makes eval trajectories greedy.",
    )
    parser.add_argument(
        "--eval-opening-depth",
        type=int,
        default=None,
        help="Use deterministic legal opening prefixes of this depth for eval games only (0-6).",
    )
    parser.add_argument(
        "--select-best-generation-by-solver",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use solver-eval-only champion selection after training.",
    )
    parser.add_argument("--selection-eval-games", type=int, default=None)
    parser.add_argument("--selection-eval-mcts", type=int, default=None)
    return parser.parse_args()


def make_run_dir(output_dir: str, benchmark_tier: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(output_dir)
        / f"solver-alignment-{benchmark_tier}-{stamp}-{uuid.uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_cache_path(cache: str, run_dir: Path, run_local_cache: bool) -> Path:
    cache_path = Path(cache)
    if not cache_path.is_absolute():
        cache_path = (
            run_dir / cache_path if run_local_cache else Path.cwd() / cache_path
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    return cache_path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(config: EvalConfig) -> ModelConfig:
    return ModelConfig(
        n_residual_blocks=config.n_residual_blocks,
        conv_filter_size=config.conv_filter_size,
        n_policy_layers=config.n_policy_layers,
        n_value_layers=config.n_value_layers,
        lr_schedule=parse_lr_schedule([0, config.learning_rate]),
        l2_reg=config.l2_reg,
        policy_loss_weight=config.policy_loss_weight,
        q_penalty_loss_weight=config.q_penalty_loss_weight,
        q_no_penalty_loss_weight=config.q_no_penalty_loss_weight,
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
        replay_window=config.replay_window,
        max_gens=config.train_gens,
        solver_config=None,
    )


def eval_opening_moves(game_index: int, depth: int) -> list[int]:
    if depth < 0 or depth > 6:
        raise ValueError("eval_opening_depth must be in [0, 6]")
    moves: list[int] = []
    value = game_index
    for _ in range(depth):
        moves.append(value % 7)
        value //= 7
    return moves


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
        c4a0_rust.GameMetadata(  # type: ignore[attr-defined]
            config.eval_game_id_offset + game_id,
            0,
            0,
            eval_opening_moves(game_id, config.eval_opening_depth),
        )
        for game_id in range(config.eval_games)
    ]
    return c4a0_rust.play_games(  # type: ignore[attr-defined]
        reqs,
        config.self_play_batch_size,
        config.eval_mcts,
        config.c_exploration,
        config.c_ply_penalty,
        lambda model_id, pos: model.forward_numpy(pos),
        config.eval_temperature,
    )


def score_alignment(
    games: c4a0_rust.PlayGamesResult,  # type: ignore[name-defined]
    solver_path: str,
    book_path: str,
    cache_path: Path,
) -> float:
    return float(games.score_top_moves(solver_path, book_path, str(cache_path)))


def select_best_generation_by_solver(
    run_dir: Path,
    trained_final_gen: TrainingGen,
    config: EvalConfig,
    device: torch.device,
    solver_path: str,
    book_path: str,
    cache_path: Path,
) -> tuple[TrainingGen, list[dict[str, float | int | str]]]:
    selection_config = replace(
        config,
        eval_games=config.selection_eval_games,
        eval_mcts=config.selection_eval_mcts,
        eval_game_id_offset=config.eval_game_id_offset + 10_000_000,
    )
    generations = [
        gen
        for gen in TrainingGen.load_all(str(run_dir / "training"))
        if 0 < gen.gen_n <= trained_final_gen.gen_n
    ]
    generations.sort(key=lambda gen: gen.gen_n)

    rows: list[dict[str, float | int | str]] = []
    best_gen = trained_final_gen
    best_metric = float("-inf")
    for gen in generations:
        games = generate_eval_games(run_dir, gen, selection_config, device)
        metric = score_alignment(games, solver_path, book_path, cache_path)
        row = {
            "gen_n": gen.gen_n,
            "created_at": gen.created_at.isoformat(),
            "metric": metric,
            "eval_games": len(games.results),
            "eval_nonterminal_samples": games.nonterminal_sample_count(),
            "eval_unique_positions": games.unique_positions(),
        }
        rows.append(row)
        if metric > best_metric:
            best_metric = metric
            best_gen = gen
    return best_gen, rows


def wilson_interval(
    success_rate: float, n: int, z: float = 1.959963984540054
) -> dict[str, float | int]:
    """Approximate 95% Wilson interval for strict 0/1 top-move agreement."""
    if n <= 0:
        return {"n": n, "lower": 0.0, "upper": 0.0, "half_width": 0.0}
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (success_rate + z2 / (2.0 * n)) / denom
    margin = (
        z
        * ((success_rate * (1.0 - success_rate) / n + z2 / (4.0 * n * n)) ** 0.5)
        / denom
    )
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return {"n": n, "lower": lower, "upper": upper, "half_width": (upper - lower) / 2.0}


def build_config(args: argparse.Namespace) -> EvalConfig:
    tier = BENCHMARK_TIERS[args.benchmark_tier]

    def tier_or_override(name: str) -> int:
        value = getattr(args, name)
        return int(value if value is not None else getattr(tier, name))

    return EvalConfig(
        benchmark_tier=args.benchmark_tier,
        train_gens=tier_or_override("train_gens"),
        train_games=tier_or_override("train_games"),
        train_mcts=tier_or_override("train_mcts"),
        eval_games=tier_or_override("eval_games"),
        eval_mcts=tier_or_override("eval_mcts"),
        self_play_batch_size=tier_or_override("self_play_batch_size"),
        training_batch_size=tier_or_override("training_batch_size"),
        replay_window=tier_or_override("replay_window"),
        c_exploration=args.c_exploration,
        c_ply_penalty=args.c_ply_penalty,
        n_residual_blocks=args.n_residual_blocks,
        conv_filter_size=args.conv_filter_size,
        n_policy_layers=args.n_policy_layers,
        n_value_layers=args.n_value_layers,
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
        policy_loss_weight=args.policy_loss_weight,
        q_penalty_loss_weight=args.q_penalty_loss_weight,
        q_no_penalty_loss_weight=args.q_no_penalty_loss_weight,
        eval_game_id_offset=args.eval_game_id_offset,
        eval_temperature=args.eval_temperature,
        eval_opening_depth=tier_or_override("eval_opening_depth"),
        seed=args.seed,
        select_best_generation_by_solver=(
            tier.select_best_generation_by_solver
            if args.select_best_generation_by_solver is None
            else args.select_best_generation_by_solver
        ),
        selection_eval_games=tier_or_override("selection_eval_games"),
        selection_eval_mcts=tier_or_override("selection_eval_mcts"),
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
    argv: list[str],
    trained_final_gen: TrainingGen | None = None,
    selection_rows: list[dict[str, float | int | str]] | None = None,
) -> None:
    nonterminal_samples = games.nonterminal_sample_count()
    metadata = {
        "metric": metric,
        "metric_name": "solver_alignment_mcts_strict_top_move",
        "metric_direction": "higher_is_better",
        "metric_confidence_interval": wilson_interval(metric, nonterminal_samples),
        "solver_is_eval_only": True,
        "solver_path": solver_path,
        "book_path": book_path,
        "cache_path": str(cache_path),
        "benchmark_tier": config.benchmark_tier,
        "benchmark_tier_description": BENCHMARK_TIERS[
            config.benchmark_tier
        ].description,
        "config": asdict(config),
        "final_generation": gen.model_dump(mode="json"),
        "trained_final_generation": (trained_final_gen or gen).model_dump(mode="json"),
        "selection_rows": selection_rows or [],
        "eval_games": len(games.results),
        "eval_samples": games.sample_count(),
        "eval_nonterminal_samples": nonterminal_samples,
        "eval_unique_positions": games.unique_positions(),
        "training_dir": str(run_dir / "training"),
        "git_commit": git_commit(),
        "argv": argv,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with open(run_dir / "eval_games.pkl", "wb") as f:
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
    run_dir = make_run_dir(args.output_dir, config.benchmark_tier)
    device = torch.device(args.device)
    cache_path = resolve_cache_path(args.cache, run_dir, args.run_local_cache)

    seed_everything(config.seed)
    trained_final_gen = train_self_play_only(run_dir, config, device)
    if config.select_best_generation_by_solver:
        gen, selection_rows = select_best_generation_by_solver(
            run_dir,
            trained_final_gen,
            config,
            device,
            str(solver),
            str(book),
            cache_path,
        )
    else:
        gen = trained_final_gen
        selection_rows = []
    games = generate_eval_games(run_dir, gen, config, device)
    metric = score_alignment(games, str(solver), str(book), cache_path)
    write_metrics(
        run_dir,
        config,
        gen,
        games,
        metric,
        str(solver),
        str(book),
        cache_path,
        sys.argv[1:],
        trained_final_gen=trained_final_gen,
        selection_rows=selection_rows,
    )

    print(f"{metric:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
