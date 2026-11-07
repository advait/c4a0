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
                "greedy_after_ply": None,
                "root_noise_alpha": None,
                "root_noise_frac": 0.0,
                "value_target_q_weight": 0.0,
                "temperature_schedule": ((4, 4.0), (8, 2.0)),
                "mask_terminal_policy": False,
                "replay_gens": 1,
            },
        ),
        (
            [
                "--greedy-after-ply",
                "8",
                "--root-noise-alpha",
                "0.3",
                "--root-noise-frac",
                "0.25",
                "--value-target-q-weight",
                "0.6",
                "--temperature-schedule",
                "2:1.5",
                "--mask-terminal-policy",
                "--replay-gens",
                "3",
            ],
            {
                "greedy_after_ply": 8,
                "root_noise_alpha": 0.3,
                "root_noise_frac": 0.25,
                "value_target_q_weight": 0.6,
                "temperature_schedule": ((2, 1.5),),
                "mask_terminal_policy": True,
                "replay_gens": 3,
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


@pytest.mark.parametrize(
    "flags", [["--greedy-after-ply", "-1"], ["--replay-gens", "0"]]
)
def test_train_cli_rejects_invalid_cutoff_or_window(flags: list[str]) -> None:
    result = CliRunner().invoke(main.app, ["train", *flags])
    assert result.exit_code == 2


@pytest.mark.parametrize("weight", ["-0.1", "1.1", "nan", "inf", "-inf"])
def test_train_cli_rejects_invalid_value_target_weight(weight: str) -> None:
    result = CliRunner().invoke(main.app, ["train", "--value-target-q-weight", weight])
    assert result.exit_code == 2
    assert "value-target-q-weight" in result.output


@pytest.mark.parametrize(
    "spec,message",
    [
        ("", "until_ply:temperature"),
        ("2", "until_ply:temperature"),
        ("2:1:3", "until_ply:temperature"),
        ("x:1", "integer"),
        ("2.5:1", "integer"),
        ("-1:1", "non-negative"),
        ("4:4,4:2", "strictly increasing"),
        ("8:4,4:2", "strictly increasing"),
        ("2:0", "positive"),
        ("2:-1", "positive"),
        ("2:nan", "finite"),
        ("2:inf", "finite"),
        ("2:nope", "number"),
        ("2:1,", "until_ply:temperature"),
    ],
)
def test_train_cli_rejects_invalid_temperature_schedule(
    spec: str, message: str
) -> None:
    result = CliRunner().invoke(main.app, ["train", "--temperature-schedule", spec])
    assert result.exit_code == 2
    assert "temperature-schedule" in result.output
    assert message in " ".join(result.output.replace("│", " ").split())
