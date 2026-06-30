import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "solver_alignment_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("solver_alignment_eval", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
solver_alignment_eval = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = solver_alignment_eval
_SPEC.loader.exec_module(solver_alignment_eval)


def test_solver_alignment_training_is_self_play_only(monkeypatch, tmp_path):
    captured = {}

    def fake_training_loop(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(gen_n=3)

    monkeypatch.setattr(solver_alignment_eval, "training_loop", fake_training_loop)
    config = solver_alignment_eval.EvalConfig(
        benchmark_tier="smoke",
        train_gens=3,
        train_games=4,
        train_mcts=4,
        eval_games=4,
        eval_mcts=4,
        self_play_batch_size=8,
        training_batch_size=8,
        replay_window=1,
        c_exploration=6.6,
        c_ply_penalty=0.01,
        n_residual_blocks=1,
        conv_filter_size=8,
        n_policy_layers=1,
        n_value_layers=1,
        learning_rate=1e-3,
        l2_reg=0.0,
        eval_game_id_offset=100,
    )

    gen = solver_alignment_eval.train_self_play_only(
        Path(tmp_path),
        config,
        torch.device("cpu"),
    )

    assert gen.gen_n == 3
    assert captured["solver_config"] is None
    assert captured["max_gens"] == config.train_gens
    assert captured["n_self_play_games"] == config.train_games
    assert captured["replay_window"] == config.replay_window


def test_benchmark_tier_defaults_and_overrides():
    args = SimpleNamespace(
        benchmark_tier="candidate",
        train_gens=None,
        train_games=123,
        train_mcts=None,
        eval_games=None,
        eval_mcts=456,
        self_play_batch_size=None,
        training_batch_size=None,
        replay_window=None,
        c_exploration=6.6,
        c_ply_penalty=0.0,
        n_residual_blocks=1,
        conv_filter_size=16,
        n_policy_layers=1,
        n_value_layers=1,
        learning_rate=5e-4,
        l2_reg=0.0,
        eval_game_id_offset=1_000_000,
    )

    config = solver_alignment_eval.build_config(args)
    tier = solver_alignment_eval.BENCHMARK_TIERS["candidate"]

    assert config.benchmark_tier == "candidate"
    assert config.train_gens == tier.train_gens
    assert config.train_games == 123
    assert config.train_mcts == tier.train_mcts
    assert config.replay_window == tier.replay_window
    assert config.eval_games == tier.eval_games
    assert config.eval_mcts == 456
    assert config.eval_games >= 10_000
