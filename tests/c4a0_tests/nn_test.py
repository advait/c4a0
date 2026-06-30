import pytest
import torch

from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.training import parse_lr_schedule
from c4a0_rust import BUF_N_CHANNELS, N_COLS, N_ROWS


def make_model(
    policy_loss_weight: float = 1.0,
    q_penalty_loss_weight: float = 1.0,
    q_no_penalty_loss_weight: float = 1.0,
) -> ConnectFourNet:
    return ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=8,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule=parse_lr_schedule([0, 1e-3]),
            l2_reg=0.0,
            policy_loss_weight=policy_loss_weight,
            q_penalty_loss_weight=q_penalty_loss_weight,
            q_no_penalty_loss_weight=q_no_penalty_loss_weight,
        )
    )


def starting_pos() -> torch.Tensor:
    return torch.zeros((1, BUF_N_CHANNELS, N_ROWS, N_COLS), dtype=torch.float32)


def test_random_nn_works():
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
def test_loss_of_zero():
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
def test_loss_of_nonzero():
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
@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_loss_uses_configured_component_weights():
    model = make_model(
        policy_loss_weight=2.0,
        q_penalty_loss_weight=0.5,
        q_no_penalty_loss_weight=0.25,
    )
    pos = starting_pos()
    policy = torch.ones((1, N_COLS)) / N_COLS
    q_penalty = torch.ones((1,))
    q_no_penalty = -torch.ones((1,))

    model.eval()
    policy_logprob, q_penalty_pred, q_no_penalty_pred = model(pos)
    policy_loss = (
        (policy * (torch.log(policy + model.EPS) - policy_logprob)).sum(dim=1).mean()
    )
    q_penalty_loss = ((q_penalty_pred - q_penalty) ** 2).mean()
    q_no_penalty_loss = ((q_no_penalty_pred - q_no_penalty) ** 2).mean()
    expected = 2.0 * policy_loss + 0.5 * q_penalty_loss + 0.25 * q_no_penalty_loss

    actual = model.training_step((pos, policy, q_penalty, q_no_penalty), 0)

    assert actual.detach().item() == pytest.approx(expected.detach().item(), abs=1e-6)


def test_loss_logs_both_value_components(monkeypatch):
    """Value diagnostics should expose both value heads, not just q_penalty."""
    model = make_model()
    logged_names: list[str] = []

    def capture_log(name, *args, **kwargs):
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
