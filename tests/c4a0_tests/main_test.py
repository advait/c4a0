import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from c4a0 import main
from c4a0.nn import ConnectFourNet
from c4a0.training import GenConfig

if TYPE_CHECKING:
    from c4a0_rust import EvalPosCb


def _plain(output: str) -> str:
    """CLI output without terminal escape codes. Rich styles errors when it detects CI (e.g.
    GITHUB_ACTIONS), and the codes can split an option name."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


@pytest.mark.parametrize(
    "flags,expected",
    [
        (
            [],
            {
                "root_noise_alpha": 1.0,
                "root_noise_frac": 0.25,
                "random_start_frac": 0.5,
                "random_start_max_ply": 12,
                "value_target_q_weight": 1.0,
                "leaf_minimax_depth": 0,
                "replay_gens": 8,
                "train_max_epochs": 20,
                "train_patience": 3,
                "bf16": False,
                "compile": False,
                "train_steps": None,
                "head": "fc",
                "value_hidden": 256,
            },
        ),
        (
            [
                "--head",
                "conv",
                "--value-hidden",
                "17",
                "--bf16",
                "--compile",
                "--train-steps",
                "5",
                "--root-noise-alpha",
                "0.3",
                "--root-noise-frac",
                "0.25",
                "--random-start-frac",
                "0.5",
                "--random-start-max-ply",
                "17",
                "--value-target-q-weight",
                "0.6",
                "--leaf-minimax-depth",
                "6",
                "--replay-gens",
                "3",
                "--train-max-epochs",
                "7",
                "--train-patience",
                "2",
            ],
            {
                "root_noise_alpha": 0.3,
                "root_noise_frac": 0.25,
                "random_start_frac": 0.5,
                "random_start_max_ply": 17,
                "value_target_q_weight": 0.6,
                "leaf_minimax_depth": 6,
                "replay_gens": 3,
                "train_max_epochs": 7,
                "train_patience": 2,
                "bf16": True,
                "compile": True,
                "train_steps": 5,
                "head": "conv",
                "value_hidden": 17,
            },
        ),
    ],
)
def test_train_cli_forwards_experiment_flags(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], expected: dict[str, object]
) -> None:
    captured: dict[str, object] = {}

    def capture_training(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(main, "training_loop", capture_training)
    result = CliRunner().invoke(main.app, ["train", *flags])
    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert isinstance(config, GenConfig)
    assert config.c_exploration == 2.0
    for key, value in expected.items():
        assert config.model_dump()[key] == value
    net = captured["model_config"]
    assert isinstance(net, main.ModelConfig)
    assert net.head == config.head
    assert net.value_hidden == config.value_hidden


@pytest.mark.parametrize("command", ["train", "ceiling"])
@pytest.mark.parametrize(
    "flags", [["--head", "other"], ["--value-hidden", "0"], ["--value-hidden", "-1"]]
)
def test_head_options_reject_invalid_values(command: str, flags: list[str]) -> None:
    args = ["labels.npz", "output"] if command == "ceiling" else []
    result = CliRunner().invoke(main.app, [command, *args, *flags])
    assert result.exit_code == 2
    assert flags[0] in _plain(result.output)


@pytest.mark.parametrize("head", ["fc", "conv"])
def test_ceiling_head_options_reach_model_pickle_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, head: str
) -> None:
    import json
    import pickle

    from c4a0.labels import Labels, Recipe, Split, write_labels
    from c4a0.nn import ConnectFourNet
    from c4a0_tests.labels_test import PONS_ROWS, make_labels
    from c4a0_tests.training_test import _tiny_suite

    labels = tmp_path / "train.tsv"
    write_labels(
        labels,
        make_labels(
            PONS_ROWS,
            Recipe(version=1, split=Split.TRAIN, seed=1, per_ply=4, plies=(0, 41)),
        ),
    )
    suite = _tiny_suite()
    monkeypatch.setattr(main, "load_suite", lambda _split: suite)

    def capture_fit(
        model: ConnectFourNet,
        train: Labels,
        val: Labels,
        c_ply_penalty: float,
        batch_size: int,
        max_epochs: int,
        patience: int,
        anneal_epochs: int | None,
    ) -> tuple[ConnectFourNet, int]:
        assert anneal_epochs is None
        assert model.head == head
        assert model.hparams["value_hidden"] == 17
        return model.eval(), 0

    monkeypatch.setattr(main, "fit_supervised", capture_fit)
    output = tmp_path / "ceiling"
    result = CliRunner().invoke(
        main.app,
        [
            "ceiling",
            str(labels),
            str(output),
            "--head",
            head,
            "--value-hidden",
            "17",
            "--n-residual-blocks",
            "1",
            "--conv-filter-size",
            "8",
            "--n-policy-layers",
            "1",
            "--n-value-layers",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((output / "report.json").read_text())
    assert report["net"]["head"] == head
    assert report["net"]["value_hidden"] == 17
    with (output / "model.pkl").open("rb") as f:
        restored = pickle.load(f)
    assert restored.head == head
    assert restored.hparams["value_hidden"] == 17


@pytest.mark.parametrize("flags", [["--replay-gens", "0"]])
def test_train_cli_rejects_invalid_window(flags: list[str]) -> None:
    result = CliRunner().invoke(main.app, ["train", *flags])
    assert result.exit_code == 2


@pytest.mark.parametrize(
    "flag", ["--train-max-epochs", "--train-patience", "--train-steps"]
)
@pytest.mark.parametrize("value", ["0", "-1"])
def test_train_cli_rejects_invalid_training_length(flag: str, value: str) -> None:
    result = CliRunner().invoke(main.app, ["train", flag, value])
    assert result.exit_code == 2
    assert flag in _plain(result.output)


@pytest.mark.parametrize("frac", ["-0.1", "1.1", "nan", "inf", "-inf"])
def test_train_cli_rejects_invalid_random_start_fraction(frac: str) -> None:
    result = CliRunner().invoke(main.app, ["train", "--random-start-frac", frac])
    assert result.exit_code == 2
    assert "random-start-frac" in _plain(result.output)


@pytest.mark.parametrize("ply", ["0", "-1"])
def test_train_cli_rejects_invalid_random_start_max_ply(ply: str) -> None:
    result = CliRunner().invoke(main.app, ["train", "--random-start-max-ply", ply])
    assert result.exit_code == 2
    assert "random-start-max-ply" in _plain(result.output)


def test_train_cli_rejects_negative_leaf_minimax_depth() -> None:
    result = CliRunner().invoke(main.app, ["train", "--leaf-minimax-depth", "-1"])
    assert result.exit_code == 2
    assert "leaf-minimax-depth" in _plain(result.output)


@pytest.mark.parametrize("weight", ["-0.1", "1.1", "nan", "inf", "-inf"])
def test_train_cli_rejects_invalid_value_target_weight(weight: str) -> None:
    result = CliRunner().invoke(main.app, ["train", "--value-target-q-weight", weight])
    assert result.exit_code == 2
    assert "value-target-q-weight" in _plain(result.output)


@pytest.mark.parametrize("device", ["auto", "cpu"])
@pytest.mark.parametrize("budgets", [[], ["--eval-mcts-iterations", "8"]])
def test_train_defaults_and_device_selection(
    monkeypatch: pytest.MonkeyPatch, device: str, budgets: list[str]
) -> None:
    import torch

    from c4a0.nn import ModelConfig
    from c4a0.training import EvalConfig

    captured: dict[str, object] = {}

    def capture_training(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(main, "training_loop", capture_training)
    monkeypatch.setattr(main, "get_torch_device", lambda: torch.device("mps"))
    result = CliRunner().invoke(main.app, ["train", "--device", device, *budgets])
    assert result.exit_code == 0, result.output
    assert captured["device"] == torch.device("mps" if device == "auto" else "cpu")
    config = captured["config"]
    assert isinstance(config, GenConfig)
    assert config.n_self_play_games == 2048
    assert config.n_mcts_iterations == 256
    assert config.c_exploration == 2.0
    assert config.c_ply_penalty == 0.01
    net = captured["model_config"]
    assert isinstance(net, ModelConfig)
    assert net.n_residual_blocks == 4
    assert net.conv_filter_size == 64
    evaluation = captured["eval_config"]
    assert isinstance(evaluation, EvalConfig)
    assert evaluation.mcts_iterations == ((8,) if budgets else (64, 256))
    assert captured["base_dir"] == "training"


@pytest.mark.parametrize("model", ["uniform", "random"])
def test_play_baselines_need_no_training_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    import numpy as np

    called: list[bool] = []

    def run_tui(callback: "EvalPosCb", *_args: object) -> None:
        logits, qp, qn = callback(0, main.c4a0_rust.encode_positions([""]))
        assert logits.shape == (1, 7)
        assert qp.shape == qn.shape == (1,)
        assert np.isfinite(logits).all()
        called.append(True)

    def unexpected_load(_base_dir: str) -> None:
        pytest.fail("baseline play must not load a generation")

    monkeypatch.setattr(main.c4a0_rust, "run_tui", run_tui)
    monkeypatch.setattr(main.TrainingGen, "load_latest", unexpected_load)
    result = CliRunner().invoke(
        main.app, ["play", "--base-dir", str(tmp_path / "missing"), "--model", model]
    )
    assert result.exit_code == 0, result.output
    assert called == [True]


@pytest.mark.parametrize("command", ["play", "eval"])
def test_cli_loads_legacy_cuda_model_on_cpu(
    legacy_cuda_model: tuple[ConnectFourNet, Path],
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    import hashlib
    from datetime import datetime

    import numpy as np
    import torch

    from c4a0.labels import Suite
    from c4a0.training import TrainingGen
    from c4a0_tests.labels_test import PONS_ROWS, make_labels
    from c4a0_tests.training_test import TINY_GEN_CONFIG

    model, path = legacy_cuda_model
    base_dir = str(path.parent)
    gen = TrainingGen(
        created_at=datetime.now(), gen_n=0, **TINY_GEN_CONFIG.model_dump()
    )
    folder = Path(gen.gen_folder(base_dir))
    folder.mkdir()
    model_path = folder / "model.pkl"
    path.rename(model_path)
    gen.save_metadata(base_dir)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    called: list[bool] = []

    def run_tui(callback: "EvalPosCb", *_args: object) -> None:
        positions = main.c4a0_rust.encode_positions(["3344"])
        for actual, expected in zip(
            callback(0, positions), model.forward_numpy(positions), strict=True
        ):
            np.testing.assert_array_equal(actual, expected)
        called.append(True)

    suite = Suite(labels=make_labels(PONS_ROWS), name="test/tiny", sha256="abc")
    monkeypatch.setattr(main.c4a0_rust, "run_tui", run_tui)
    monkeypatch.setattr(main, "load_suite", lambda _split: suite)
    flags = ["--mcts-iterations", "4"] if command == "eval" else []
    result = CliRunner().invoke(main.app, [command, "--base-dir", base_dir, *flags])

    assert result.exit_code == 0, result.output
    if command == "play":
        assert called == [True]
    else:
        score = TrainingGen.load_latest(base_dir).evals[suite.name]
        assert score.model_sha256 == hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert set(score.mcts) == {4}


@pytest.mark.parametrize("existing_dir", [False, True])
def test_play_best_without_generations_explains_how_to_train(
    tmp_path: Path, existing_dir: bool
) -> None:
    base = tmp_path / "training"
    if existing_dir:
        base.mkdir()
    result = CliRunner().invoke(main.app, ["play", "--base-dir", str(base)])
    assert result.exit_code == 1
    assert (
        result.output == "No trained generations found. Run `mise run train` first.\n"
    )
