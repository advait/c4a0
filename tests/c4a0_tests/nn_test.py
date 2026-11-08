import copy

import pytest
import torch

from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training import parse_lr_schedule
from c4a0_rust import BUF_N_CHANNELS, N_COLS, N_ROWS


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


def test_inference_callback_matches_legacy_across_batch_sizes() -> None:
    import numpy as np

    import c4a0_rust
    from c4a0.nn import NumpyEvaluator
    from c4a0_tests.training_test import TINY_CONFIG

    model = ConnectFourNet(TINY_CONFIG).eval()
    evaluator = NumpyEvaluator(model)
    for size in [1, 4, 32, 2, 17, 1]:
        positions = c4a0_rust.encode_positions(["123", "33", "1234", ""] * size)
        with torch.no_grad():
            expected = tuple(
                np.ascontiguousarray(t.cpu().numpy())
                for t in model(torch.from_numpy(positions))
            )
        for output in (model.forward_numpy(positions), evaluator(positions)):
            for actual, legacy in zip(output, expected, strict=True):
                assert actual.shape == legacy.shape
                assert actual.tobytes() == legacy.tobytes()
