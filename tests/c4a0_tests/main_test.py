import pytest
from typer.testing import CliRunner

from c4a0 import main


@pytest.mark.parametrize(
    "flags,expected",
    [
        (
            [],
            {
                "greedy_after_ply": None,
                "root_noise_alpha": None,
                "root_noise_frac": 0.0,
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
                "--mask-terminal-policy",
                "--replay-gens",
                "3",
            ],
            {
                "greedy_after_ply": 8,
                "root_noise_alpha": 0.3,
                "root_noise_frac": 0.25,
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
    assert captured["c_exploration"] == 2.0
    for key, value in expected.items():
        assert captured[key] == value


@pytest.mark.parametrize(
    "flags", [["--greedy-after-ply", "-1"], ["--replay-gens", "0"]]
)
def test_train_cli_rejects_invalid_cutoff_or_window(flags: list[str]) -> None:
    result = CliRunner().invoke(main.app, ["train", *flags])
    assert result.exit_code == 2
