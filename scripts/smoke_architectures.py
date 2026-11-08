"""Run each new architecture through the CPU ceiling CLI on 256 suite rows.

These rows are repackaged as train labels only for a plumbing check. The report
is contaminated by dev overlap and must never be used as a convergence result.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from c4a0.labels import Labels, Recipe, Split, load_suite, write_labels


def smoke(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    source = load_suite(Split.DEV)
    indices = np.linspace(0, len(source.labels) - 1, 256, dtype=np.int64)
    rows = source.labels.take(indices)
    labels_path = out_dir / "smoke-only-labels.tsv"
    write_labels(
        labels_path,
        Labels(
            recipe=Recipe(
                version=1, split=Split.TRAIN, seed=0, per_ply=0, plies=(0, 41)
            ),
            moves=rows.moves,
            scores=rows.scores,
        ),
    )
    # All children run on CPU with the same seed, labels, optimizer and work limits.
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for arch in ("resnet_gpool", "transformer", "hybrid"):
        command = [
            sys.executable,
            "-m",
            "c4a0.main",
            "ceiling",
            str(labels_path),
            str(out_dir / arch),
            "--arch",
            arch,
            "--head",
            "conv",
            "--n-residual-blocks",
            "2",
            "--conv-filter-size",
            "8",
            "--n-layers",
            "2",
            "--d-model",
            "16",
            "--n-heads",
            "4",
            "--value-hidden",
            "17",
            "--batch-size",
            "64",
            "--max-epochs",
            "2",
            "--patience",
            "2",
            "--seed",
            "1337",
            "--mcts-iterations",
            "4",
        ]
        with (out_dir / f"{arch}.log").open("w") as log:
            subprocess.run(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        print(f"{arch}: saved model.pkl and report.json", flush=True)
    manifest = {
        "smoke_only": True,
        "convergence_evidence": False,
        "device": "cpu",
        "source_suite": source.name,
        "source_sha256": source.sha256,
        "source_row_indices": indices.tolist(),
        "n_labels": len(rows),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "warning": "Training rows overlap dev. Scores only establish CLI plumbing, not generalization.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path, help="New directory for smoke artifacts.")
    smoke(parser.parse_args().out_dir)
