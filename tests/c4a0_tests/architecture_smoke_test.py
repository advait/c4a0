"""Real CLI fits on a few hundred committed solver labels, with CPU artifacts."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import c4a0_rust
from c4a0.nn import ConnectFourNet
from c4a0.supervised import CeilingReport
from c4a0.utils import load_model


def test_new_architectures_reach_real_ceiling_cli(tmp_path: Path) -> None:
    output = tmp_path / "smoke"
    result = subprocess.run(
        [sys.executable, "scripts/smoke_architectures.py", str(output)],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["smoke_only"] and not manifest["convergence_evidence"]
    assert manifest["device"] == "cpu"
    assert manifest["n_labels"] == len(set(manifest["source_row_indices"])) == 256
    labels_hash = hashlib.sha256(
        (output / "smoke-only-labels.tsv").read_bytes()
    ).hexdigest()
    assert manifest["labels_sha256"] == labels_hash
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for arch in ("resnet_gpool", "transformer", "hybrid"):
            folder = output / arch
            report = CeilingReport.model_validate_json(
                (folder / "report.json").read_text()
            )
            assert report.net.arch == arch
            assert (report.net.n_layers, report.net.d_model, report.net.n_heads) == (
                2,
                16,
                4,
            )
            assert (report.net.n_residual_blocks, report.net.conv_filter_size) == (2, 8)
            assert report.n_train == 256 and report.epochs == 2
            assert report.train_sha256 == labels_hash
            assert report.test is None and set(report.dev.mcts) == {4}
            model_path = folder / "model.pkl"
            assert (
                report.dev.model_sha256
                == hashlib.sha256(model_path.read_bytes()).hexdigest()
            )
            trained = load_model(model_path)
            assert trained.device.type == "cpu" and trained.arch == arch
            assert report.parameter_count == sum(
                p.numel() for p in trained.parameters()
            )
            with torch.random.fork_rng():
                torch.manual_seed(report.seed)
                initial = ConnectFourNet(report.net)
            assert any(
                not torch.equal(a, b)
                for a, b in zip(trained.parameters(), initial.parameters(), strict=True)
            )
            positions = c4a0_rust.encode_positions(["", "123", "1212"])
            policy, qp, qn = trained.forward_numpy(positions)
            assert policy.shape == (3, 7) and qp.shape == qn.shape == (3,)
            assert all(np.isfinite(t).all() for t in (policy, qp, qn))
    finally:
        torch.set_num_threads(previous)
