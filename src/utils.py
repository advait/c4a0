from sys import platform

import torch


def get_torch_device() -> torch.device:
    """Tries to use cuda or mps, if available, otherwise falls back to cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if platform == "darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS unavailable because the current torch install was not built with MPS enabled."
            )
        else:
            raise RuntimeError(
                "MPS unavailable because the current MacOS version is not 12.3+ and/or you do not "
                "have an MPS-enabled device on this machine."
            )

    return torch.device("cpu")
