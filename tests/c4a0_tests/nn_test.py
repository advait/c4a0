import copy
import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

import c4a0_rust
from c4a0.nn import ConnectFourNet, ModelConfig, NumpyEvaluator
from c4a0.training import parse_lr_schedule
from c4a0.utils import load_model
from c4a0_rust import BUF_N_CHANNELS, N_COLS, N_ROWS

FIXTURES = Path(__file__).parent / "fixtures"


def test_initialization_matches_pre_head_removal_bytes() -> None:
    # Captured from 515664f (conv heads, before FC heads were removed) with one torch
    # thread, including BatchNorm buffers. Never regenerate from new code.
    config = ModelConfig(
        n_residual_blocks=4,
        conv_filter_size=64,
        lr_schedule={0: 0.001},
        l2_reg=0.0,
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(1337)
            model = ConnectFourNet(config)
    finally:
        torch.set_num_threads(threads)
    actual = {
        name: {
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.detach().numpy().tobytes()).hexdigest(),
        }
        for name, value in model.state_dict().items()
    }
    assert actual == json.loads((FIXTURES / "conv_4x64_seed1337.json").read_text())


@pytest.mark.parametrize("batch_size", [1, 5])
def test_heads_preserve_forward_and_numpy_contract(batch_size: int) -> None:
    config = ModelConfig(
        n_residual_blocks=1,
        conv_filter_size=8,
        value_hidden=17,
        lr_schedule={0: 0.001},
        l2_reg=0.0,
    )
    model = ConnectFourNet(config)
    pos = torch.rand(batch_size, 2, N_ROWS, N_COLS)
    # The heads train with singleton batches too: their BatchNorm is spatial.
    policy, qp, qn = model(pos)
    (policy.sum() + qp.sum() + qn.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())
    copy.deepcopy(model)  # BestModelCheckpoint's publication path.
    model.eval()
    with torch.no_grad():
        outputs = model(pos)
    policy, qp, qn = outputs
    assert policy.shape == (batch_size, N_COLS)
    assert qp.shape == qn.shape == (batch_size,)
    assert all(torch.isfinite(t).all() for t in outputs)
    assert (policy <= 0).all()
    torch.testing.assert_close(policy.exp().sum(dim=1), torch.ones(batch_size))
    assert ((qp >= -1) & (qp <= 1)).all()
    assert ((qn >= -1) & (qn <= 1)).all()
    restored = pickle.loads(pickle.dumps(model))
    for actual, expected in zip(
        NumpyEvaluator(restored)(pos.numpy()), outputs, strict=True
    ):
        assert actual.shape == tuple(expected.shape)
        assert actual.tobytes() == expected.numpy().tobytes()


def test_4x64_has_309470_parameters() -> None:
    config = ModelConfig(
        n_residual_blocks=4,
        conv_filter_size=64,
        lr_schedule={0: 0.001},
        l2_reg=0.0,
    )
    model = ConnectFourNet(config)
    assert sum(p.numel() for p in model.parameters()) == 309470


def test_conv_pickle_saved_before_head_removal_predicts_identically() -> None:
    # Saved by 515664f with head="conv" and value_hidden=17, BatchNorm stats moved off
    # their initial values, rather than built by the code under test.
    with (FIXTURES / "conv_model_515664f.pkl").open("rb") as f:
        legacy = pickle.load(f)
    assert legacy.hparams["head"] == "conv"
    config = ModelConfig.model_validate(dict(legacy.hparams))
    assert config.value_hidden == 17
    current = ConnectFourNet(config)
    current.load_state_dict(legacy.state_dict())
    current.eval()
    positions = c4a0_rust.encode_positions(["", "3", "334", "0123456"])
    for actual, expected in zip(
        NumpyEvaluator(legacy)(positions), current.forward_numpy(positions), strict=True
    ):
        assert actual.tobytes() == expected.tobytes()


def test_fc_head_pickles_fail_to_load_with_a_clear_error(tmp_path: Path) -> None:
    model = make_model()
    del model.conv_policy  # What an FC-head pickle lacks.
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(model))
    with pytest.raises(ValueError, match="fully connected heads"):
        load_model(path)


def make_model() -> ConnectFourNet:
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=8,
            lr_schedule=parse_lr_schedule([0, 1e-3]),
            l2_reg=0.0,
        )
    )


def starting_pos() -> torch.Tensor:
    return torch.zeros((1, BUF_N_CHANNELS, N_ROWS, N_COLS), dtype=torch.float32)


def test_random_nn_works() -> None:
    model = make_model()
    pos = starting_pos()
    model.eval()
    with torch.no_grad():
        policy_logprobs, q_penalty, q_no_penalty = model(pos)

    policy = torch.exp(policy_logprobs).squeeze(0).numpy()
    q_penalty = q_penalty.squeeze(0).item()
    q_no_penalty = q_no_penalty.squeeze(0).item()

    assert len(policy) == N_COLS
    assert policy.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= q_penalty <= 1.0
    assert -1.0 <= q_no_penalty <= 1.0


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_loss_of_zero() -> None:
    """Using the model output as training labels should result in a loss of zero."""
    model = make_model()
    pos = starting_pos()
    model.eval()
    with torch.no_grad():
        policy_logprobs, q_penalty, q_no_penalty = model(pos)
        policy = torch.exp(policy_logprobs)

    assert pos.shape == (1, BUF_N_CHANNELS, N_ROWS, N_COLS)
    assert policy.shape == (1, N_COLS)
    assert q_penalty.shape == (1,)
    assert q_no_penalty.shape == (1,)

    training_batch = (
        pos,
        policy,
        q_penalty,
        q_no_penalty,
    )
    loss = model.training_step(training_batch, 0)
    loss = loss.detach().item()
    assert loss == pytest.approx(0.0, abs=1e-6)


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_loss_of_nonzero() -> None:
    """Using random label data should result in a > 0 loss."""
    model = make_model()
    pos = starting_pos()
    policy = torch.rand((1, N_COLS))
    policy = policy / policy.sum(dim=1, keepdim=True)
    q_penalty = torch.rand((1,))
    q_no_penalty = torch.rand((1,))

    training_batch = (
        pos,
        policy,
        q_penalty,
        q_no_penalty,
    )
    model.eval()
    loss = model.training_step(training_batch, 0)
    loss = loss.detach().item()
    assert loss > 0.0


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_loss_logs_both_value_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Value diagnostics should expose both value heads, not just q_penalty."""
    model = make_model()
    logged_names: list[str] = []

    def capture_log(name: str, *args: object, **kwargs: object) -> None:
        logged_names.append(name)

    monkeypatch.setattr(model, "log", capture_log)
    pos = starting_pos()
    policy = torch.ones((1, N_COLS)) / N_COLS
    q_penalty = torch.zeros((1,))
    q_no_penalty = torch.ones((1,))

    model.training_step((pos, policy, q_penalty, q_no_penalty), 0)

    assert "train_q_penalty_mse" in logged_names
    assert "train_q_no_penalty_mse" in logged_names
    assert "train_value_mse" in logged_names


def _tiny_net() -> ConnectFourNet:
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=8,
            lr_schedule={0: 1e-3},
            l2_reg=0.0,
        )
    )


def _batch(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return (
        torch.rand(n, 2, 6, 7, generator=generator),
        torch.softmax(torch.rand(n, 7, generator=generator), dim=1),
        torch.rand(n, generator=generator) * 2 - 1,
        torch.rand(n, generator=generator) * 2 - 1,
    )


def test_model_deepcopies_after_a_training_step() -> None:
    # BestModelCheckpoint deep-copies the model at validation end. A torchmetrics loss module
    # kept its last training result with an autograd graph, which made that copy fail.
    model = _tiny_net()
    model.train()
    model.step(_batch(4), "train")
    copy.deepcopy(model)


def test_policy_loss_is_batchmean_kl_of_target_to_prediction() -> None:
    model = _tiny_net()
    pos, policy, q_penalty, q_no_penalty = _batch(6)
    with torch.no_grad():
        loss = model.step((pos, policy, q_penalty, q_no_penalty), "val")
        logprob, qp, qn = model.forward(pos)
        target_log = torch.log(policy + ConnectFourNet.EPS)
        expected = (
            torch.nn.functional.kl_div(
                logprob, target_log, log_target=True, reduction="batchmean"
            )
            + torch.nn.functional.mse_loss(qp, q_penalty)
            + torch.nn.functional.mse_loss(qn, q_no_penalty)
        )
    assert torch.allclose(loss, expected, atol=1e-6)


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
@pytest.mark.parametrize("wide", [False, True])
def test_inference_callback_matches_legacy_across_batch_sizes(
    device: str, wide: bool
) -> None:
    import numpy as np

    import c4a0_rust
    from c4a0.nn import NumpyEvaluator
    from c4a0_tests.training_test import TINY_CONFIG

    config = (
        TINY_CONFIG.model_copy(
            update={
                "n_residual_blocks": 4,
                "conv_filter_size": 64,
            }
        )
        if wide
        else TINY_CONFIG
    )
    model = ConnectFourNet(config).to(device).eval()
    evaluator = NumpyEvaluator(model)
    for size in [1, 4, 32, 2, 17, 1]:
        moves = (["123", "33", "1234", ""] * ((size + 3) // 4))[:size]
        positions = c4a0_rust.encode_positions(moves)
        with torch.no_grad():
            expected = tuple(
                np.ascontiguousarray(t.cpu().numpy())
                for t in model.forward(torch.from_numpy(positions).to(device))
            )
        for output in (model.forward_numpy(positions), evaluator(positions)):
            for actual, legacy in zip(output, expected, strict=True):
                assert actual.shape == legacy.shape
                assert actual.tobytes() == legacy.tobytes()


@pytest.mark.parametrize("lazy", [False, True])
def test_compile_failures_fall_back_once(
    monkeypatch: pytest.MonkeyPatch, lazy: bool
) -> None:
    import c4a0_rust
    from c4a0.nn import NumpyEvaluator
    from c4a0_tests.training_test import TINY_CONFIG, TINY_GEN_CONFIG

    attempts: list[bool] = []

    def failing_compile(forward: object, *, dynamic: bool) -> object:
        attempts.append(dynamic)
        if not lazy:
            raise RuntimeError("unsupported backend")

        def failed_forward(pos: torch.Tensor) -> None:
            raise RuntimeError("lazy backend failure")

        return failed_forward

    monkeypatch.setattr(torch, "compile", failing_compile)
    model = ConnectFourNet(TINY_CONFIG).eval()
    evaluator = NumpyEvaluator(
        model, TINY_GEN_CONFIG.model_copy(update={"compile": True})
    )
    for moves in [[""], ["3", "33", "12"], ["1", "11"]]:
        pos = c4a0_rust.encode_positions(moves)
        expected = model.forward_numpy(pos)
        actual = evaluator(pos)
        assert all(
            a.tobytes() == b.tobytes() for a, b in zip(actual, expected, strict=True)
        )
    assert attempts == [True]
    assert not evaluator.compiled


def test_compiled_forward_handles_dynamic_batches_and_stays_out_of_pickle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pickle

    import c4a0_rust
    from c4a0.nn import NumpyEvaluator
    from c4a0_tests.training_test import TINY_CONFIG, TINY_GEN_CONFIG

    real_compile = torch.compile

    def compile_eager(
        forward: Callable[
            [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ],
        *,
        dynamic: bool,
    ) -> object:
        # Exercise real Dynamo graph capture without an expensive CPU Inductor build.
        return real_compile(forward, dynamic=dynamic, backend="eager")

    monkeypatch.setattr(torch, "compile", compile_eager)
    model = ConnectFourNet(TINY_CONFIG).eval()
    evaluator = NumpyEvaluator(
        model, TINY_GEN_CONFIG.model_copy(update={"compile": True})
    )
    for n in [1, 7, 3, 1]:
        pos = c4a0_rust.encode_positions(["123"] * n)
        assert all(
            a.tobytes() == b.tobytes()
            for a, b in zip(evaluator(pos), model.forward_numpy(pos), strict=True)
        )
    assert evaluator.compiled
    assert not hasattr(pickle.loads(pickle.dumps(model)), "_orig_mod")


def test_mirror_inference_is_symmetric_and_normalized() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        n_residual_blocks=1,
        conv_filter_size=8,
        lr_schedule={0: 1e-3},
        l2_reg=0.0,
        mirror_inference=True,
    )
    model = ConnectFourNet(config)
    x = torch.rand(5, 2, N_ROWS, N_COLS).round().numpy()
    policy, q_penalty, q_no_penalty = model.forward_numpy(x)
    m_policy, m_q_penalty, m_q_no_penalty = model.forward_numpy(
        np.ascontiguousarray(x[..., ::-1])
    )
    np.testing.assert_allclose(policy, m_policy[:, ::-1], atol=1e-5)
    np.testing.assert_allclose(q_penalty, m_q_penalty, atol=1e-6)
    np.testing.assert_allclose(q_no_penalty, m_q_no_penalty, atol=1e-6)
    np.testing.assert_allclose(np.exp(policy).sum(1), 1.0, atol=1e-5)
    # Self-play's evaluator searches with the same mirrored network.
    for a, b in zip(
        NumpyEvaluator(model)(x), (policy, q_penalty, q_no_penalty), strict=True
    ):
        np.testing.assert_allclose(a, b, atol=1e-5)
