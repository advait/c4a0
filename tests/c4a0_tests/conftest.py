import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from c4a0.nn import ConnectFourNet, ModelConfig

# Set the random seed for deterministic behavior
seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


@pytest.fixture
def legacy_cuda_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ConnectFourNet, Path]:
    """Real CPU storage bytes tagged CUDA by torch.save, as in a legacy CUDA pickle."""
    model = ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=8,
            n_policy_layers=1,
            n_value_layers=1,
            lr_schedule={0: 1e-3},
            l2_reg=0.0,
        )
    )
    path = tmp_path / "model.pkl"
    with monkeypatch.context() as patch:
        patch.setattr(torch.serialization, "location_tag", lambda _storage: "cuda:0")
        path.write_bytes(pickle.dumps(model))
    assert b"cuda:0" in path.read_bytes()
    return model, path
