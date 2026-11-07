import pytest
from typer.testing import CliRunner

from c4a0 import main
from c4a0.training import GenConfig


@pytest.mark.parametrize(
    "flags,expected",
    [
        (
            [],
            {
                "root_noise_alpha": None,
                "root_noise_frac": 0.0,
                "random_start_frac": 0.0,
                "random_start_max_ply": 12,
                "value_target_q_weight": 0.0,
                "replay_gens": 1,
                "train_max_epochs": 100,
                "train_patience": 10,
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
    assert flag in result.output


@pytest.mark.parametrize("frac", ["-0.1", "1.1", "nan", "inf", "-inf"])
def test_train_cli_rejects_invalid_random_start_fraction(frac: str) -> None:
    result = CliRunner().invoke(main.app, ["train", "--random-start-frac", frac])
    assert result.exit_code == 2
    assert "random-start-frac" in result.output


@pytest.mark.parametrize("ply", ["0", "-1"])
def test_train_cli_rejects_invalid_random_start_max_ply(ply: str) -> None:
    result = CliRunner().invoke(main.app, ["train", "--random-start-max-ply", ply])
    assert result.exit_code == 2
    assert "random-start-max-ply" in result.output


@pytest.mark.parametrize("weight", ["-0.1", "1.1", "nan", "inf", "-inf"])
def test_train_cli_rejects_invalid_value_target_weight(weight: str) -> None:
    result = CliRunner().invoke(main.app, ["train", "--value-target-q-weight", weight])
    assert result.exit_code == 2
    assert "value-target-q-weight" in result.output
