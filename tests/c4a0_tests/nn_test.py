import copy
import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from c4a0.nn import (
    ConnectFourNet,
    Head,
    ModelConfig,
    NumpyEvaluator,
    two_hot_cross_entropy,
)
from c4a0.training import parse_lr_schedule
from c4a0_rust import BUF_N_CHANNELS, N_COLS, N_ROWS

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("explicit_fc", [False, True])
def test_fc_initialization_matches_pre_head_option_bytes(explicit_fc: bool) -> None:
    # Captured from 1bccc92 before editing nn.py with one torch thread, including
    # BatchNorm buffers from the dummy trunk pass. Never regenerate from new code.
    config = ModelConfig(
        n_residual_blocks=4,
        conv_filter_size=64,
        n_policy_layers=4,
        n_value_layers=2,
        lr_schedule={0: 0.001},
        l2_reg=0.0,
    )
    if explicit_fc:
        config = config.model_copy(update={"head": "fc", "value_hidden": 17})
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
    assert actual == json.loads((FIXTURES / "fc_4x64_seed1337.json").read_text())


@pytest.mark.parametrize("head", ["fc", "conv"])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_heads_preserve_forward_and_numpy_contract(head: Head, batch_size: int) -> None:
    config = ModelConfig(
        head=head,
        n_residual_blocks=1,
        conv_filter_size=8,
        n_policy_layers=2,
        n_value_layers=2,
        value_hidden=17,
        lr_schedule={0: 0.001},
        l2_reg=0.0,
    )
    model = ConnectFourNet(config)
    pos = torch.rand(batch_size, 2, N_ROWS, N_COLS)
    # Conv heads must also train with singleton batches: their BN is spatial.
    if batch_size > 1 or head == "conv":
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


def test_conv_4x64_has_309470_parameters_and_ignores_fc_depth() -> None:
    config = ModelConfig(
        head="conv",
        n_residual_blocks=4,
        conv_filter_size=64,
        n_policy_layers=4,
        n_value_layers=2,
        lr_schedule={0: 0.001},
        l2_reg=0.0,
    )
    torch.manual_seed(1337)
    model = ConnectFourNet(config)
    assert sum(p.numel() for p in model.parameters()) == 309470
    assert sum(p.numel() for p in model.parameters()) < 1_000_000
    torch.manual_seed(1337)
    ignored_depths = ConnectFourNet(
        config.model_copy(update={"n_policy_layers": 99, "n_value_layers": 0})
    )
    assert model.state_dict().keys() == ignored_depths.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, ignored_depths.state_dict()[name])


def test_legacy_model_pickle_without_head_still_predicts_identically() -> None:
    # This pickle was saved by 1bccc92, rather than stripping fields off a new model.
    with (FIXTURES / "legacy_model.pkl").open("rb") as f:
        legacy = pickle.load(f)
    assert not hasattr(legacy, "head")
    assert "head" not in legacy.hparams
    with torch.random.fork_rng():
        torch.manual_seed(1337)
        current = make_model().eval()
    positions = torch.zeros(3, 2, N_ROWS, N_COLS).numpy()
    for actual, expected in zip(
        NumpyEvaluator(legacy)(positions), current.forward_numpy(positions), strict=True
    ):
        assert actual.tobytes() == expected.tobytes()
    config = ModelConfig.model_validate(dict(legacy.hparams))
    assert config.head == "fc"
    assert config.value_hidden == 256


def make_model() -> ConnectFourNet:
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=8,
            n_policy_layers=1,
            n_value_layers=1,
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
            n_policy_layers=1,
            n_value_layers=1,
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
                "n_policy_layers": 4,
                "n_value_layers": 2,
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
        n_policy_layers=1,
        n_value_layers=1,
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


def test_two_hot_cross_entropy_is_minimized_by_the_two_hot_target() -> None:
    n_bins = 201
    target = torch.tensor([-1.0, -0.375, 0.0, 0.42, 1.0])
    centers = torch.linspace(-1.0, 1.0, n_bins)
    position = (target + 1) / 2 * (n_bins - 1)
    low = position.floor().clamp(max=n_bins - 2)
    two_hot = torch.zeros(len(target), n_bins)
    two_hot[torch.arange(len(target)), low.long()] = 1 - (position - low)
    two_hot[torch.arange(len(target)), low.long() + 1] += position - low
    # The encoding's mean is the target, and its own log-probabilities minimize the loss.
    torch.testing.assert_close(two_hot @ centers, target)
    best = two_hot_cross_entropy(torch.log(two_hot + 1e-12), target)
    worse = two_hot_cross_entropy(torch.zeros(len(target), n_bins), target)
    assert best < worse


def test_value_bins_head_returns_the_mean_and_learns_a_target() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        head="conv",
        value_hidden=16,
        value_bins=21,
        n_residual_blocks=1,
        conv_filter_size=8,
        n_policy_layers=1,
        n_value_layers=1,
        lr_schedule={0: 1e-2},
        l2_reg=0.0,
    )
    model = ConnectFourNet(config)
    pos = torch.rand(64, 2, N_ROWS, N_COLS).round()
    _, q_penalty, q_no_penalty, logits = model.forward_with_bins(pos)
    assert logits is not None and logits.shape == (64, 21)
    torch.testing.assert_close(q_penalty, logits.softmax(1) @ torch.linspace(-1, 1, 21))
    assert q_no_penalty.abs().max() <= 1
    target = torch.full((64,), 0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        optimizer.zero_grad()
        bins = model.forward_with_bins(pos)[3]
        assert bins is not None
        two_hot_cross_entropy(bins, target).backward()
        optimizer.step()
    model.eval()
    assert model.forward(pos)[1].sub(0.3).abs().max() < 0.02


def test_value_bins_require_conv_heads() -> None:
    with pytest.raises(ValueError, match="conv heads"):
        ModelConfig(
            value_bins=21,
            n_residual_blocks=1,
            conv_filter_size=8,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule={0: 1e-3},
            l2_reg=0.0,
        )
