"""Architecture contracts shared by self-play and supervised training."""

import copy
import json
import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from typer.testing import CliRunner

import c4a0_rust
from c4a0 import main
from c4a0.nn import (
    Arch,
    ConnectFourNet,
    GlobalPoolResidualBlock,
    Head,
    ModelConfig,
    NumpyEvaluator,
)
from c4a0.training import EvalConfig, GenConfig, TrainingGen, train_single_gen
from c4a0_tests.nn_test import _batch
from c4a0_tests.training_test import FIXTURES, TINY_CONFIG, TINY_GEN_CONFIG, _tiny_suite

NEW_ARCHES: tuple[Arch, ...] = ("resnet_gpool", "transformer", "hybrid")


@pytest.fixture(autouse=True)
def one_thread() -> Iterator[None]:
    # Small CPU attention/conv jobs otherwise spend most time scheduling threads.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def tiny_config(arch: Arch, head: Head = "conv") -> ModelConfig:
    return ModelConfig.model_validate(
        {
            **TINY_CONFIG.model_dump(),
            "arch": arch,
            "head": head,
            "n_layers": 2,
            "d_model": 16,
            "n_heads": 4,
            "value_hidden": 17,
        }
    )


@pytest.mark.parametrize("arch", NEW_ARCHES)
@pytest.mark.parametrize("head", ["fc", "conv"])
@pytest.mark.parametrize("size", [1, 5])
def test_architecture_contract_and_pickle(arch: Arch, head: Head, size: int) -> None:
    model = ConnectFourNet(tiny_config(arch, head))
    positions = c4a0_rust.encode_positions(["1234", "33", "", "1212", "123"][:size])
    # FC BatchNorm1d needs multiple examples in training, as before.
    if size == 1 and arch == "resnet_gpool" and head == "fc":
        model.eval()
    policy, qp, qn = model(torch.from_numpy(positions))
    assert policy.shape == (size, 7)
    assert qp.shape == qn.shape == (size,)
    assert all(torch.isfinite(t).all() for t in (policy, qp, qn))
    assert (policy <= 0).all()
    torch.testing.assert_close(policy.exp().sum(1), torch.ones(size))
    assert (qp.abs() <= 1).all() and (qn.abs() <= 1).all()
    (policy.square().mean() + qp.square().mean() + qn.square().mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    copy.deepcopy(model)
    expected = model.forward_numpy(positions)
    restored = pickle.loads(pickle.dumps(model))
    for callback in (restored.forward_numpy, NumpyEvaluator(restored)):
        for actual, reference in zip(callback(positions), expected, strict=True):
            assert actual.dtype == np.float32
            assert actual.shape == reference.shape
            assert actual.tobytes() == reference.tobytes()


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
@pytest.mark.parametrize("arch", NEW_ARCHES)
@pytest.mark.parametrize("head", ["fc", "conv"])
def test_architecture_learns_in_a_few_cpu_steps(arch: Arch, head: Head) -> None:
    torch.manual_seed(123)
    model = ConnectFourNet(tiny_config(arch, head)).train()
    batch = _batch(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    losses: list[float] = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = model.step(batch, "train")
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0] * 0.8, losses


@pytest.mark.parametrize("pool_index", [0, 1])
def test_global_pooling_connects_distant_cells(pool_index: int) -> None:
    block = GlobalPoolResidualBlock(1).eval()
    with torch.no_grad():
        for layer in (block.conv1, block.conv2):
            layer.weight.zero_()
            layer.weight[0, 0, 1, 1] = 1
            assert layer.bias is not None
            layer.bias.zero_()
        block.pool_bias.weight.zero_()
        block.pool_bias.weight[0, pool_index] = 1
        block.pool_bias.bias.zero_()
    empty = torch.zeros(1, 1, 6, 7)
    distant = empty.clone()
    distant[0, 0, 0, 0] = 1
    effect = (block(distant) - block(empty))[0, 0, 5, 6]
    expected = (1 / 42 if pool_index == 0 else 1) / (1 + block.norm.eps) ** 0.5
    assert effect.item() == pytest.approx(expected)


@pytest.mark.parametrize("arch", ["transformer", "hybrid"])
def test_token_models_ignore_spatial_head_settings(arch: Arch) -> None:
    torch.manual_seed(123)
    model = ConnectFourNet(tiny_config(arch, "fc"))
    torch.manual_seed(123)
    other = ConnectFourNet(
        tiny_config(arch).model_copy(
            update={"n_policy_layers": 99, "n_value_layers": 99}
        )
    )
    assert model.state_dict().keys() == other.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, other.state_dict()[name])


@pytest.mark.parametrize("command", ["train", "ceiling"])
@pytest.mark.parametrize(
    "flags",
    [
        ["--arch", "unknown"],
        ["--n-layers", "0"],
        ["--d-model", "0"],
        ["--n-heads", "0"],
        ["--arch", "transformer", "--d-model", "15"],
    ],
)
def test_invalid_architecture_cli_options(command: str, flags: list[str]) -> None:
    args = ["missing.tsv", "output"] if command == "ceiling" else []
    result = CliRunner().invoke(main.app, [command, *args, *flags])
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("arch", NEW_ARCHES)
def test_train_forwards_architecture_options(
    monkeypatch: pytest.MonkeyPatch, arch: Arch
) -> None:
    captured: dict[str, object] = {}

    def capture(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(main, "training_loop", capture)
    result = CliRunner().invoke(
        main.app,
        [
            "train",
            "--arch",
            arch,
            "--n-layers",
            "2",
            "--d-model",
            "24",
            "--n-heads",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    for key in ("config", "model_config"):
        config = captured[key]
        assert isinstance(config, (GenConfig, ModelConfig))
        assert (config.arch, config.n_layers, config.d_model, config.n_heads) == (
            arch,
            2,
            24,
            3,
        )


def test_legacy_configs_default_to_resnet() -> None:
    gen = GenConfig.model_validate_json((FIXTURES / "legacy_metadata.json").read_text())
    with (FIXTURES / "legacy_model.pkl").open("rb") as f:
        legacy = pickle.load(f)
    model = ModelConfig.model_validate(dict(legacy.hparams))
    for config in (gen, model):
        assert (config.arch, config.n_layers, config.d_model, config.n_heads) == (
            "resnet",
            6,
            128,
            4,
        )
    for config in (TINY_CONFIG, TINY_GEN_CONFIG):
        for name, value in (
            ("arch", "unknown"),
            ("n_layers", 0),
            ("d_model", 0),
            ("n_heads", 0),
        ):
            with pytest.raises(ValidationError):
                type(config).model_validate({**config.model_dump(), name: value})
    with pytest.raises(ValidationError, match="divisible"):
        ModelConfig.model_validate(
            {**TINY_CONFIG.model_dump(), "arch": "hybrid", "d_model": 15}
        )


@pytest.mark.parametrize(
    "change",
    [
        {"arch": "resnet_gpool"},
        {"n_layers": 3},
        {"d_model": 32},
        {"n_heads": 2},
        {"value_hidden": 18},
    ],
)
def test_resume_rejects_changed_token_architecture(
    tmp_path: Path, change: dict[str, object]
) -> None:
    model_config = tiny_config("transformer")
    config = TINY_GEN_CONFIG.model_copy(
        update={
            name: getattr(model_config, name)
            for name in (
                "arch",
                "n_layers",
                "d_model",
                "n_heads",
                "head",
                "value_hidden",
            )
        }
    )
    evaluation = EvalConfig(_tiny_suite())
    parent = TrainingGen.load_latest_with_default(
        str(tmp_path), config, model_config, evaluation
    )
    assert TrainingGen.load_latest(str(tmp_path)).config == config
    with pytest.raises(ValueError, match="Architecture options differ"):
        train_single_gen(
            str(tmp_path),
            torch.device("cpu"),
            parent,
            config.model_copy(update=change),
            evaluation,
        )


def test_print_default_parameter_counts() -> None:
    counts: dict[str, int] = {}
    for arch in ("resnet", *NEW_ARCHES):
        for head in ("fc", "conv"):
            config = ModelConfig(
                arch=arch,
                head=head,
                n_residual_blocks=4,
                conv_filter_size=64,
                n_policy_layers=4,
                n_value_layers=2,
                lr_schedule={0: 0.002},
                l2_reg=0.0004,
            )
            model = ConnectFourNet(config)
            counts[f"{arch}/{head}"] = sum(p.numel() for p in model.parameters())
    print(json.dumps(counts, indent=2))
    assert counts == {
        "resnet/fc": 29_254_985,
        "resnet/conv": 309_470,
        "resnet_gpool/fc": 29_288_009,
        "resnet_gpool/conv": 342_494,
        "transformer/fc": 1_225_603,
        "transformer/conv": 1_225_603,
        "hybrid/fc": 1_530_691,
        "hybrid/conv": 1_530_691,
    }
