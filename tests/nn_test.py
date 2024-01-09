import pytest
import torch

from c4 import N_COLS, N_ROWS, STARTING_POS
from nn import ConnectFourNet


def test_random_nn_works():
    model = ConnectFourNet()
    pos = torch.from_numpy(STARTING_POS).float().unsqueeze(0)
    with torch.no_grad():
        policy, value = model(pos)

    policy = policy.squeeze(0).detach().numpy()
    value = value.squeeze(0).item()

    assert len(policy) == N_COLS
    assert policy.sum().item() == pytest.approx(1.0)
    assert -1.0 <= value <= 1.0


def test_loss_of_zero():
    model = ConnectFourNet()
    pos = torch.from_numpy(STARTING_POS).float().unsqueeze(0)
    with torch.no_grad():
        policy, value = model(pos)
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
    assert loss == pytest.approx(0.0)
