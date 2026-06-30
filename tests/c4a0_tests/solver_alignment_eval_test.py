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
        train_gens=3,
        train_games=4,
        train_mcts=4,
        eval_games=4,
        eval_mcts=4,
        self_play_batch_size=8,
        training_batch_size=8,
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
