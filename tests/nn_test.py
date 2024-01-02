import pytest
import torch

from c4 import N_COLS, STARTING_POS
from nn import ConnectFourNet


def test_random_nn_works():
    model = ConnectFourNet()
    policy, value = model.single(STARTING_POS)

    assert len(policy) == N_COLS
    assert torch.sum(policy).item() == pytest.approx(1.0)
    assert -1.0 <= value <= 1.0
