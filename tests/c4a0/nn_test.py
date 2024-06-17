import pytest
import torch

from c4a0.c4 import N_COLS, N_ROWS, STARTING_POS
from c4a0.nn import ConnectFourNet


def test_random_nn_works():
    model = ConnectFourNet()
    pos = torch.from_numpy(STARTING_POS).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        policy_logprobs, value = model(pos)

    policy = torch.exp(policy_logprobs).squeeze(0).numpy()
    value = value.squeeze(0).item()

    assert len(policy) == N_COLS
    assert policy.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= value <= 1.0


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_loss_of_zero():
    """Using the model output as training labels should result in a loss of zero."""
    model = ConnectFourNet()
    pos = torch.from_numpy(STARTING_POS).float().unsqueeze(0)
    model.eval()
    with torch.no_grad():
        policy_logprobs, value = model(pos)
        policy = torch.exp(policy_logprobs)
        value = value.squeeze(0)

    assert pos.shape == (1, N_ROWS, N_COLS)
    assert policy.shape == (1, N_COLS)
    assert value.shape == (1,)

    training_batch = (
        [0],
        pos,
        policy,
        value,
    )
    loss = model.training_step(training_batch, 0)
    loss = loss.detach().item()
    assert loss == pytest.approx(0.0, abs=1e-6)


@pytest.mark.filterwarnings("ignore:You are trying to `self.log()`*")
def test_loss_of_nonzero():
    """Using random label data should result in a > 0 loss."""
    model = ConnectFourNet()
    pos = torch.from_numpy(STARTING_POS).float().unsqueeze(0)
    policy = torch.rand((1, N_COLS))
    value = torch.rand((1,))

    training_batch = (
        [0],
        pos,
        policy,
        value,
    )
    model.eval()
    loss = model.training_step(training_batch, 0)
    loss = loss.detach().item()
    assert loss > 0.0
