import json
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Never

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from numpy.typing import NDArray
from pydantic import ValidationError
from pytorch_lightning.callbacks import EarlyStopping

import c4a0_rust
from c4a0 import training
from c4a0.evaluation import evaluate_model
from c4a0.labels import Split, Suite, load_suite
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training import (
    EvalConfig,
    GenConfig,
    SampleDataModule,
    TrainingGen,
    parse_lr_schedule,
    replay_training_games,
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
TINY_GEN_CONFIG = GenConfig(
    n_self_play_games=4,
    n_mcts_iterations=4,
    c_exploration=2.0,
    c_ply_penalty=0.01,
    self_play_batch_size=16,
    training_batch_size=10_000,
)
EXPERIMENT_CONFIG = TINY_GEN_CONFIG.model_copy(
    update={
        "root_noise_alpha": 0.3,
        "root_noise_frac": 0.25,
        "value_target_q_weight": 0.6,
        "replay_gens": 2,
    }
)


def _tiny_suite() -> Suite:
    """50 non-trivial dev rows."""
    dev = load_suite(Split.DEV)
    rows = np.flatnonzero(~dev.labels.trivial)[::40][:50]
    return Suite(labels=dev.labels.take(rows), name="v1/dev-50", sha256=dev.sha256)


def _tiny_loop(
    base_dir: Path,
    eval_config: EvalConfig,
    max_gens: int = 1,
    config: GenConfig = TINY_GEN_CONFIG,
) -> TrainingGen:
    return training_loop(
        base_dir=str(base_dir),
        device=torch.device("cpu"),
        config=config,
        model_config=TINY_CONFIG,
        max_gens=max_gens,
        eval_config=eval_config,
    )


def _manual_loss(model: ConnectFourNet, samples: tuple[torch.Tensor, ...]) -> float:
    pos, policy_target, q_penalty_target, q_no_penalty_target = samples
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
    gen = _tiny_loop(
        tmp_path,
        EvalConfig(_tiny_suite()),
        max_gens=2,
        config=EXPERIMENT_CONFIG,
    )
    for published in TrainingGen.load_all(str(tmp_path)):
        reloaded = TrainingGen.load(str(tmp_path), published.created_at)
        assert reloaded.config == gen.config
        metadata = json.loads(
            (Path(published.gen_folder(str(tmp_path))) / "metadata.json").read_text()
        )
        assert "config" not in metadata
        assert {
            name: metadata[name] for name in GenConfig.model_fields
        } == gen.config.model_dump(mode="json")
    assert gen.value_target_q_weight == 0.6
    assert TrainingGen.load(str(tmp_path), gen.created_at).value_target_q_weight == 0.6
    assert gen.replay_gens == 2
    assert TrainingGen.load(str(tmp_path), gen.created_at).replay_gens == 2
    assert gen.root_noise_alpha == 0.3
    assert gen.root_noise_frac == 0.25
    assert TrainingGen.load(str(tmp_path), gen.created_at).root_noise_frac == 0.25

    assert gen.parent is not None
    parent = TrainingGen.load(str(tmp_path), gen.parent)
    parent_model = parent.get_model(str(tmp_path))
    saved_model = gen.get_model(str(tmp_path))
    assert _sum_abs_model_diff(parent_model, saved_model) > 0.0

    games = gen.get_games(str(tmp_path))
    assert games is not None
    assert (
        len(games.results) == 4
    )  # Save only new games, never duplicate replay in games.pkl.
    games = replay_training_games(str(tmp_path), parent, games, gen.replay_gens)
    assert len(games.results) == 8
    train, validation = games.split_train_test_numpy(0.8, 1337)
    data_module = SampleDataModule(train, validation, 10_000)

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
        assert recorded == evaluate_model(model_path, suite, (4,), 2.0, 0.01)
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
    assert TrainingGen.load(str(tmp_path), created_at).value_target_q_weight == 0.0

    gen = TrainingGen.load_latest(str(tmp_path))
    assert gen.gen_n == 3
    assert gen.solver_score == pytest.approx(0.6789)
    assert gen.evals == {}
    assert gen.root_noise_alpha is None
    assert gen.root_noise_frac == 0.0
    assert gen.random_start_frac == 0.0
    assert gen.random_start_max_ply == 12
    assert gen.replay_gens == 1
    assert gen.train_max_epochs == 100
    assert gen.train_patience == 10
    assert gen.head == "fc"
    assert gen.value_hidden == 256
    assert gen.config.n_self_play_games == 1700
    # Saving a legacy generation backfills config defaults and keeps its old values.
    legacy = json.loads((FIXTURES / "legacy_metadata.json").read_text())
    gen.save_metadata(str(tmp_path))
    saved = json.loads((gen_dir / "metadata.json").read_text())
    assert {name: saved[name] for name in legacy} == legacy
    assert TrainingGen.load(str(tmp_path), created_at).config == gen.config


def test_generation_config_is_frozen() -> None:
    config = GenConfig.model_validate_json(
        (FIXTURES / "legacy_metadata.json").read_text()
    )
    with pytest.raises(ValidationError, match="frozen"):
        config.__setattr__("n_mcts_iterations", 8)


@pytest.mark.parametrize("head", ["other"])
def test_head_config_rejects_unknown_architecture(head: str) -> None:
    for config in (TINY_CONFIG, TINY_GEN_CONFIG):
        with pytest.raises(ValidationError):
            type(config).model_validate({**config.model_dump(), "head": head})


@pytest.mark.parametrize("width", [0, -1])
def test_head_config_requires_positive_value_width(width: int) -> None:
    for config in (TINY_CONFIG, TINY_GEN_CONFIG):
        with pytest.raises(ValidationError):
            type(config).model_validate({**config.model_dump(), "value_hidden": width})


@pytest.mark.parametrize("change", [{"head": "fc"}, {"value_hidden": 17}])
def test_resume_rejects_head_changes_before_self_play(
    tmp_path: Path, change: dict[str, object]
) -> None:
    config = TINY_GEN_CONFIG.model_copy(update={"head": "conv"})
    evaluation = EvalConfig(_tiny_suite())
    parent = TrainingGen.load_latest_with_default(
        str(tmp_path),
        config,
        TINY_CONFIG.model_copy(update={"head": "conv"}),
        evaluation,
    )
    with pytest.raises(ValueError, match="Head options differ from the saved model"):
        training.train_single_gen(
            str(tmp_path),
            torch.device("cpu"),
            parent,
            config.model_copy(update=change),
            evaluation,
        )


def test_conv_self_play_training_saves_reloaded_heads_and_metadata(
    tmp_path: Path,
) -> None:
    config = TINY_GEN_CONFIG.model_copy(
        update={
            "head": "conv",
            "value_hidden": 17,
            "training_batch_size": 16,
            "train_max_epochs": 2,
            "train_patience": 1,
        }
    )
    gen = training_loop(
        base_dir=str(tmp_path),
        device=torch.device("cpu"),
        config=config,
        model_config=TINY_CONFIG.model_copy(
            update={
                "head": config.head,
                "value_hidden": config.value_hidden,
                "n_policy_layers": 99,
                "n_value_layers": 0,
            }
        ),
        max_gens=1,
        eval_config=EvalConfig(_tiny_suite(), (4,)),
    )
    assert gen.val_loss is not None and np.isfinite(gen.val_loss)
    child, root = TrainingGen.load_all(str(tmp_path))
    for published in (root, child):
        assert published.config == config
        metadata = json.loads(
            (Path(published.gen_folder(str(tmp_path))) / "metadata.json").read_text()
        )
        assert metadata["head"] == "conv"
        assert metadata["value_hidden"] == 17
        model = published.get_model(str(tmp_path))
        assert model.head == "conv"
        assert model.hparams["value_hidden"] == 17
        assert 4 in published.evals["v1/dev-50"].mcts
    assert (
        _sum_abs_model_diff(
            child.get_model(str(tmp_path)), root.get_model(str(tmp_path))
        )
        > 0
    )


def test_random_starts_run_one_generation_and_reach_metadata(tmp_path: Path) -> None:
    config = TINY_GEN_CONFIG.model_copy(
        update={
            "random_start_frac": 1.0,
            "random_start_max_ply": 17,
            "train_max_epochs": 1,
        }
    )
    gen = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), config=config)
    assert gen.gen_n == 1
    for published in TrainingGen.load_all(str(tmp_path)):
        metadata = json.loads(
            (Path(published.gen_folder(str(tmp_path))) / "metadata.json").read_text()
        )
        assert metadata["random_start_frac"] == 1.0
        assert metadata["random_start_max_ply"] == 17
        assert TrainingGen.load(str(tmp_path), published.created_at).config == config
    games = gen.get_games(str(tmp_path))
    assert games is not None
    assert len(games.results) == config.n_self_play_games
    for game in games.results:
        first_ply = int(game.samples[0].to_numpy()[0].sum())
        assert 1 <= first_ply <= 17
        assert all(
            int(sample.to_numpy()[0].sum()) == first_ply + offset
            for offset, sample in enumerate(game.samples)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("random_start_frac", -0.1),
        ("random_start_frac", 1.1),
        ("random_start_frac", float("nan")),
        ("random_start_frac", float("inf")),
        ("random_start_max_ply", 0),
        ("random_start_max_ply", -1),
    ],
)
def test_generation_config_rejects_invalid_random_starts(
    field: str, value: float
) -> None:
    with pytest.raises(ValidationError, match=field):
        GenConfig.model_validate({**TINY_GEN_CONFIG.model_dump(), field: value})


@pytest.mark.parametrize("field", ["train_max_epochs", "train_patience"])
@pytest.mark.parametrize("value", [0, -1])
def test_generation_config_rejects_invalid_training_length(
    field: str, value: int
) -> None:
    with pytest.raises(ValidationError, match=field):
        GenConfig.model_validate({**TINY_GEN_CONFIG.model_dump(), field: value})


@pytest.mark.parametrize(
    "max_epochs,patience,expected_epochs", [(1, 3, 1), (7, 1, 2), (7, 2, 3)]
)
def test_training_length_controls_real_fits_and_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_epochs: int,
    patience: int,
    expected_epochs: int,
) -> None:
    observed: list[tuple[int, int | None, int]] = []

    class ObservedStopping(EarlyStopping):
        def on_train_start(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
        ) -> None:
            # Treat all finite losses as a plateau after the first epoch, so a real
            # Lightning fit exercises patience independently of model improvements.
            self.min_delta = -1e6  # EarlyStopping negates min_delta for mode="min".

        def on_train_end(
            self, trainer: pl.Trainer, pl_module: pl.LightningModule
        ) -> None:
            observed.append((trainer.current_epoch, trainer.max_epochs, self.patience))

    monkeypatch.setattr(training, "EarlyStopping", ObservedStopping)
    config = TINY_GEN_CONFIG.model_copy(
        update={"train_max_epochs": max_epochs, "train_patience": patience}
    )
    gen = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), max_gens=2, config=config)
    assert observed == [(expected_epochs, max_epochs, patience)] * 2
    for published in TrainingGen.load_all(str(tmp_path)):
        assert published.config == config
        metadata = json.loads(
            (Path(published.gen_folder(str(tmp_path))) / "metadata.json").read_text()
        )
        assert metadata["train_max_epochs"] == max_epochs
        assert metadata["train_patience"] == patience
    assert gen.gen_n == 2


def test_replay_window_loads_only_recent_ancestors(tmp_path: Path) -> None:
    def uniform_eval(
        _model_id: int, pos: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        return (
            np.zeros((len(pos), 7), dtype=np.float32),
            np.zeros(len(pos), dtype=np.float32),
            np.zeros(len(pos), dtype=np.float32),
        )

    generations: list[TrainingGen] = []
    for n in [0, 1, 2, 3, 99]:
        gen = TrainingGen(
            created_at=datetime(2024, 1, 1) + timedelta(seconds=n),
            gen_n=n,
            n_mcts_iterations=2,
            c_exploration=2.0,
            c_ply_penalty=0.01,
            self_play_batch_size=8,
            training_batch_size=8,
            parent=generations[-1].created_at if generations and n != 99 else None,
        )
        gen.save_metadata(str(tmp_path))
        games = (
            None
            if n == 0
            else c4a0_rust.play_games(
                [c4a0_rust.GameMetadata(n * 10 + i, 0, 0) for i in range(2)],
                8,
                2,
                2.0,
                0.01,
                uniform_eval,
                root_noise_alpha=None,
                root_noise_frac=0.0,
                value_target_q_weight=0.0,
            )
        )
        with open(Path(gen.gen_folder(str(tmp_path))) / "games.pkl", "wb") as f:
            pickle.dump(games, f)
        generations.append(gen)

    current = generations[3].get_games(str(tmp_path))
    assert current is not None
    parent = generations[2]
    for window, expected_ids in [
        (1, [30, 31]),
        (2, [20, 21, 30, 31]),
        (99, [10, 11, 20, 21, 30, 31]),
    ]:
        combined = replay_training_games(str(tmp_path), parent, current, window)
        assert sorted(g.metadata.game_id for g in combined.results) == expected_ids
    # A two-generation window does not even load metadata beyond its boundary.
    (Path(generations[1].gen_folder(str(tmp_path))) / "metadata.json").unlink()
    assert len(replay_training_games(str(tmp_path), parent, current, 2).results) == 4
    # Default returns exactly the current object and requires no history files.
    (Path(parent.gen_folder(str(tmp_path))) / "games.pkl").unlink()
    assert replay_training_games(str(tmp_path), parent, current, 1) is current
    with pytest.raises(ValueError, match="at least 1"):
        replay_training_games(str(tmp_path), parent, current, 0)


def test_each_generation_gets_fresh_game_ids(tmp_path: Path) -> None:
    gen = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), max_gens=2)
    assert gen.parent is not None
    parent = TrainingGen.load(str(tmp_path), gen.parent)
    ids: list[set[int]] = []
    for g in (parent, gen):
        games = g.get_games(str(tmp_path))
        assert games is not None
        ids.append({r.metadata.game_id for r in games.results})
    assert len(ids[0]) == len(ids[1]) == 4
    assert ids[0].isdisjoint(ids[1])


def test_legacy_metadata_ignores_retired_controls(tmp_path: Path) -> None:
    metadata = json.loads((FIXTURES / "legacy_metadata.json").read_text())
    metadata.update(
        greedy_after_ply=8,
        mask_terminal_policy=True,
        temperature_schedule=[[2, 1.5]],
    )
    gen = TrainingGen.model_validate(metadata)
    gen.save_metadata(str(tmp_path))
    loaded = TrainingGen.load(str(tmp_path), gen.created_at)
    assert loaded == gen
    assert loaded.gen_n == metadata["gen_n"]
    assert (
        not {"greedy_after_ply", "mask_terminal_policy", "temperature_schedule"}
        & loaded.model_dump().keys()
    )


def test_generation_metrics_and_reusing_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from c4a0.benchmark import GenerationMetrics

    config = TINY_GEN_CONFIG.model_copy(
        update={"train_max_epochs": 1, "replay_gens": 2}
    )
    parent = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), config=config)

    def unexpected_play(*args: object, **kwargs: object) -> Never:
        raise AssertionError("training benchmark must skip self-play")

    monkeypatch.setattr(c4a0_rust, "play_games", unexpected_play)
    metrics = GenerationMetrics(torch.device("cpu"))
    child = training.train_single_gen(
        str(tmp_path),
        torch.device("cpu"),
        parent,
        config,
        EvalConfig(_tiny_suite()),
        metrics=metrics,
        reuse_games=True,
    )
    assert child.gen_n == parent.gen_n + 1
    assert child.parent == parent.created_at
    games = child.get_games(str(tmp_path))
    old_games = parent.get_games(str(tmp_path))
    assert games is not None and old_games is not None
    assert games.__getstate__() == old_games.__getstate__()
    assert metrics.epochs == metrics.optimizer_steps == 1
    assert set(metrics.seconds) == {
        "setup",
        "self_play",
        "replay_load",
        "sample_conversion",
        "training",
        "evaluation",
        "publish",
    }
    assert all(t >= 0 for t in metrics.seconds.values())


@pytest.mark.parametrize("steps", [1, 5, 11])
def test_fixed_steps_publish_final_model_and_record_its_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: int
) -> None:
    observed: list[dict[str, torch.Tensor]] = []
    original_fit = training.GenerationTrainer.fit

    def fit(
        self: training.GenerationTrainer,
        model: ConnectFourNet,
        datamodule: SampleDataModule,
    ) -> None:
        assert not any(isinstance(cb, EarlyStopping) for cb in self.callbacks)
        original_fit(self, model, datamodule)
        assert self.global_step == steps
        assert self.current_epoch == 1
        observed.append({k: v.detach().clone() for k, v in model.state_dict().items()})

    monkeypatch.setattr(training.GenerationTrainer, "fit", fit)
    config = TINY_GEN_CONFIG.model_copy(
        update={"train_steps": steps, "training_batch_size": 16}
    )
    gen = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), config=config)
    saved = gen.get_model(str(tmp_path))
    assert len(observed) == 1
    assert all(torch.equal(t, saved.state_dict()[k]) for k, t in observed[0].items())
    assert gen.train_steps == steps
    assert TrainingGen.load(str(tmp_path), gen.created_at).train_steps == steps
    games = gen.get_games(str(tmp_path))
    assert games is not None
    train, val = games.split_train_test_numpy(0.8, 1337)
    dm = SampleDataModule(train, val, 16)
    assert gen.val_loss == pytest.approx(
        _manual_loss(saved, dm.validation_data), abs=1e-5
    )


def test_bf16_self_play_and_training_fit_and_metadata(tmp_path: Path) -> None:
    config = TINY_GEN_CONFIG.model_copy(
        update={"bf16": True, "train_steps": 2, "training_batch_size": 16}
    )
    gen = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), config=config)
    assert gen.bf16 is True
    assert gen.val_loss is not None and np.isfinite(gen.val_loss)
    assert TrainingGen.load(str(tmp_path), gen.created_at).bf16 is True
    # Mixed precision keeps the published parameters in float32.
    assert all(
        t.dtype == torch.float32 for t in gen.get_model(str(tmp_path)).parameters()
    )


@pytest.mark.parametrize("steps", [0, -1])
def test_invalid_fixed_steps(steps: int) -> None:
    with pytest.raises(ValidationError, match="train_steps"):
        GenConfig.model_validate({**TINY_GEN_CONFIG.model_dump(), "train_steps": steps})


def test_compile_fallback_reaches_generation_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported(*args: object, **kwargs: object) -> Never:
        raise RuntimeError("unsupported test backend")

    monkeypatch.setattr(torch, "compile", unsupported)
    config = TINY_GEN_CONFIG.model_copy(update={"compile": True, "train_steps": 1})
    gen = _tiny_loop(tmp_path, EvalConfig(_tiny_suite()), config=config)
    assert gen.compile is True
    assert TrainingGen.load(str(tmp_path), gen.created_at).compile is True


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_sample_datasets_live_on_requested_device_and_loaders_share_them(
    device: str,
) -> None:
    from torch.utils.data import TensorDataset

    arrays = (
        np.zeros((8, 2, 6, 7), np.float32),
        np.zeros((8, 7), np.float32),
        np.zeros(8, np.float32),
        np.zeros(8, np.float32),
    )
    dm = SampleDataModule(arrays, arrays, 4, device)
    for tensors, loader in [
        (dm.training_data, dm.train_dataloader()),
        (dm.validation_data, dm.val_dataloader()),
    ]:
        assert isinstance(loader.dataset, TensorDataset)
        for source, stored in zip(tensors, loader.dataset.tensors, strict=True):
            assert source.device.type == device
            assert source.data_ptr() == stored.data_ptr()
        assert all(t.device.type == device for t in next(iter(loader)))
