import pytest
import torch

from c4 import N_COLS, STARTING_POS
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
