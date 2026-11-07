import pickle
import shutil
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Never

import numpy as np
import pytest
import torch

from c4a0 import training
from c4a0.evaluation import evaluate_model
from c4a0.labels import Split, Suite, load_suite
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training import (
    EvalConfig,
    SampleDataModule,
    SampleTensor,
    TrainingGen,
    parse_lr_schedule,
    training_loop,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TINY_CONFIG = ModelConfig(
    n_residual_blocks=1,
    conv_filter_size=8,
    n_policy_layers=1,
    n_value_layers=1,
    lr_schedule=parse_lr_schedule([0, 1e-3]),
    l2_reg=0.0,
)


def _tiny_suite() -> Suite:
    """50 non-trivial dev rows."""
    dev = load_suite(Split.DEV)
    rows = np.flatnonzero(~dev.labels.trivial)[::40][:50]
    return Suite(labels=dev.labels.take(rows), name="v1/dev-50", sha256=dev.sha256)


def _tiny_loop(
    base_dir: Path, eval_config: EvalConfig, max_gens: int = 1
) -> TrainingGen:
    return training_loop(
        base_dir=str(base_dir),
        device=torch.device("cpu"),
        n_self_play_games=4,
        n_mcts_iterations=4,
        c_exploration=6.6,
        c_ply_penalty=0.01,
        self_play_batch_size=16,
        training_batch_size=10_000,
        model_config=TINY_CONFIG,
        max_gens=max_gens,
        eval_config=eval_config,
    )


def _manual_loss(model: ConnectFourNet, samples: Sequence[SampleTensor]) -> float:
    pos = torch.stack([sample[0] for sample in samples])
    policy_target = torch.stack([sample[1] for sample in samples])
    q_penalty_target = torch.stack([sample[2] for sample in samples])
    q_no_penalty_target = torch.stack([sample[3] for sample in samples])
    model.eval()
    with torch.no_grad():
        policy_logprob, q_penalty_pred, q_no_penalty_pred = model(pos)
        policy_loss = (
            (policy_target * (torch.log(policy_target + model.EPS) - policy_logprob))
            .sum(dim=1)
            .mean()
        )
        q_penalty_loss = ((q_penalty_pred - q_penalty_target) ** 2).mean()
        q_no_penalty_loss = ((q_no_penalty_pred - q_no_penalty_target) ** 2).mean()
    return (policy_loss + q_penalty_loss + q_no_penalty_loss).item()


def _sum_abs_model_diff(left: ConnectFourNet, right: ConnectFourNet) -> float:
    return sum(
        float(
            (value.detach().cpu() - right.state_dict()[key].detach().cpu()).abs().sum()
        )
        for key, value in left.state_dict().items()
    )


def test_training_loop_saves_trained_best_model_and_matching_metadata(
    tmp_path: Path,
) -> None:
    gen = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()))

    assert gen.parent is not None
    parent = TrainingGen.load(str(tmp_path), gen.parent)
    parent_model = parent.get_model(str(tmp_path))
    saved_model = gen.get_model(str(tmp_path))
    assert _sum_abs_model_diff(parent_model, saved_model) > 0.0

    games = gen.get_games(str(tmp_path))
    assert games is not None
    train, validation = games.split_train_test(0.8, 1337)
    data_module = SampleDataModule(list(train), list(validation), 10_000)

    assert gen.val_loss == pytest.approx(
        _manual_loss(saved_model, data_module.validation_data),
        abs=1e-5,
    )

    with open(gen.gen_folder(str(tmp_path)) + "/model.pkl", "rb") as f:
        pickled_model = pickle.load(f)
    assert _sum_abs_model_diff(saved_model, pickled_model) == pytest.approx(0.0)


def test_each_generation_records_evals_of_its_own_saved_model(tmp_path: Path) -> None:
    suite = _tiny_suite()
    gen = _tiny_loop(tmp_path, EvalConfig(suite, mcts_iterations=(4,)))

    assert gen.parent is not None
    for g in [gen, TrainingGen.load(str(tmp_path), gen.parent)]:
        reloaded = TrainingGen.load(str(tmp_path), g.created_at)
        assert list(reloaded.evals) == ["v1/dev-50"]
        recorded = reloaded.evals["v1/dev-50"]
        model_path = Path(g.gen_folder(str(tmp_path))) / "model.pkl"
        assert recorded == evaluate_model(model_path, suite, (4,), 6.6, 0.01)
        assert sum(t.n for t in recorded.raw.by_bucket.values()) == 50
        assert sorted(recorded.mcts) == [4]
    assert gen.parent is not None
    child = TrainingGen.load(str(tmp_path), gen.created_at)
    parent = TrainingGen.load(str(tmp_path), gen.parent)
    child_sha = child.evals["v1/dev-50"].model_sha256
    assert child_sha != parent.evals["v1/dev-50"].model_sha256


def test_failed_evaluation_publishes_no_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _tiny_suite()
    published = _tiny_loop(tmp_path, EvalConfig(suite))

    def failing_eval(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("eval failed")

    monkeypatch.setattr(training, "evaluate_model", failing_eval)
    with pytest.raises(RuntimeError):
        _tiny_loop(tmp_path, EvalConfig(suite))

    gen_dirs = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert len(gen_dirs) == 3
    assert not (gen_dirs[-1] / "metadata.json").exists()
    assert (gen_dirs[-1] / "model.pkl").exists()
    assert TrainingGen.load_latest(str(tmp_path)).created_at == published.created_at
    assert [g.gen_n for g in TrainingGen.load_all(str(tmp_path))] == [1, 0]


def test_legacy_metadata_loads_with_empty_evals(tmp_path: Path) -> None:
    created_at = datetime.fromisoformat("2024-06-01T12:34:56.789012")
    gen_dir = tmp_path / created_at.isoformat()
    gen_dir.mkdir()
    shutil.copy(FIXTURES / "legacy_metadata.json", gen_dir / "metadata.json")

    gen = TrainingGen.load_latest(str(tmp_path))
    assert gen.gen_n == 3
    assert gen.solver_score == pytest.approx(0.6789)
    assert gen.evals == {}
