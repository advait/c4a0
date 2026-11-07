import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from c4a0 import main
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
                "replay_gens": 8,
                "train_max_epochs": 20,
                "train_patience": 3,
            },
        ),
        (
            [
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
                "replay_gens": 3,
                "train_max_epochs": 7,
                "train_patience": 2,
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


@pytest.mark.parametrize("flags", [["--replay-gens", "0"]])
def test_train_cli_rejects_invalid_window(flags: list[str]) -> None:
    result = CliRunner().invoke(main.app, ["train", *flags])
    assert result.exit_code == 2


@pytest.mark.parametrize("flag", ["--train-max-epochs", "--train-patience"])
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
