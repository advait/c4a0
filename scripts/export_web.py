"""Export a model for the browser player (web/): weights with BatchNorm folded into the
convolution before it, a layer manifest, and reference outputs the page checks itself against.

Usage: uv run --locked python scripts/export_web.py OUT_DIR [MODEL]

MODEL defaults to models/c4a0-final.pkl. Writes OUT_DIR/weights.bin (little-endian float32),
OUT_DIR/manifest.json (layers with offsets into weights.bin, in float32 elements) and
OUT_DIR/cases.json (64 encoded dev positions with the PyTorch outputs for them).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import c4a0_rust
from c4a0.labels import Split, load_suite
from c4a0.nn import ConnectFourNet, ResidualBlock
from c4a0.utils import load_model

ROOT = Path(__file__).resolve().parent.parent
N_CASES = 64

Layer = dict[str, int | str | bool]


class Weights:
    def __init__(self) -> None:
        self.arrays: list[np.ndarray] = []
        self.size = 0

    def put(self, t: torch.Tensor) -> int:
        a = t.detach().to(torch.float32).contiguous().numpy().ravel()
        self.arrays.append(a)
        self.size += a.size
        return self.size - a.size

    def conv(
        self, conv: nn.Conv2d, bn: nn.BatchNorm2d | None, *, relu: bool, residual: bool
    ) -> Layer:
        """A convolution with the BatchNorm after it folded into its weights and bias."""
        w, b = conv.weight, conv.bias
        assert b is not None and conv.kernel_size[0] == conv.kernel_size[1]
        if bn is not None:
            assert bn.running_var is not None and bn.running_mean is not None
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w, b = (
                w * scale[:, None, None, None],
                (b - bn.running_mean) * scale + bn.bias,
            )
        return {
            "kind": "conv",
            "cin": conv.in_channels,
            "cout": conv.out_channels,
            "k": conv.kernel_size[0],
            "w": self.put(w),
            "b": self.put(b),
            "act": "relu" if relu else "none",
            "residual": residual,
        }

    def linear(self, m: nn.Linear, act: str) -> Layer:
        return {
            "kind": "linear",
            "cin": m.in_features,
            "cout": m.out_features,
            "k": 1,
            "w": self.put(m.weight),
            "b": self.put(m.bias),
            "act": act,
            "residual": False,
        }


def layers(model: ConnectFourNet, w: Weights) -> dict[str, list[Layer]]:
    """The forward pass as three layer lists; see web/net.js for how they chain."""
    stem, *blocks = model.conv
    assert isinstance(stem, nn.Conv2d)
    trunk = [w.conv(stem, None, relu=False, residual=False)]
    for block in blocks:
        assert isinstance(block, ResidualBlock)
        conv1, conv2, bn, _relu = block.block
        trunk.append(w.conv(conv1, None, relu=False, residual=False))
        trunk.append(
            w.conv(conv2, bn, relu=True, residual=True)
        )  # x + relu(bn(conv(conv(x))))
    p, v = model.conv_policy, model.conv_value
    return {
        "trunk": trunk,
        "policy": [
            w.conv(p[0], p[1], relu=True, residual=False),
            w.linear(p[4], "none"),
        ],
        "value": [
            w.conv(v[0], v[1], relu=True, residual=False),
            w.linear(v[4], "relu"),
            w.linear(v[6], "tanh"),
        ],
    }


def main() -> None:
    out = Path(sys.argv[1])
    model_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models/c4a0-final.pkl"
    )
    model: ConnectFourNet = load_model(model_path)
    model.eval()
    assert not getattr(model, "mirror_inference", False), (
        "the page applies mirroring itself"
    )

    w = Weights()
    manifest = {
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        **layers(model, w),
    }
    moves = list(load_suite(Split.DEV).labels.moves[:N_CASES])
    enc = c4a0_rust.encode_positions(moves).astype(np.float32)
    logp, q_penalty, q_no_penalty = model.forward_numpy(enc)
    cases = {
        "moves": moves,
        "input": enc.ravel().tolist(),
        "policy": logp.ravel().tolist(),
        "q_penalty": q_penalty.tolist(),
        "q_no_penalty": q_no_penalty.tolist(),
    }

    out.mkdir(parents=True, exist_ok=True)
    np.concatenate(w.arrays).astype("<f4").tofile(out / "weights.bin")
    (out / "manifest.json").write_text(json.dumps(manifest))
    (out / "cases.json").write_text(json.dumps(cases))
    print(f"{out}: {w.size:,} floats for {manifest['parameters']:,} parameters")


if __name__ == "__main__":
    main()
