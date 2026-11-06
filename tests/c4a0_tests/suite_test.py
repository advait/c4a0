"""Coverage of the committed v1 suites. Split purity and per-ply quotas are checked in Rust."""

from pathlib import Path

import numpy as np
import pytest

from c4a0.evaluation import tally_policy, top1_credit
from c4a0.labels import (
    BUCKET_NAMES,
    SUITE_V1,
    Split,
    format_labels,
    label,
    load_suite,
    suite_path,
)

SOLVER = Path(__file__).resolve().parents[2] / "solver" / "c4solver"
BOOK = SOLVER.with_name("7x6.book")
SPLITS = [Split.DEV, Split.TEST]


@pytest.mark.parametrize("split", SPLITS)
def test_every_ply_from_two_is_present(split):
    # Plies 0 and 1 have one and four mirror classes, and all of them hash to train.
    counts = np.bincount(load_suite(split).labels.ply, minlength=42)
    assert (counts[2:] > 0).all()
    assert (counts[5:] == 64).all()
    assert counts[:2].tolist() == [0, 0]


@pytest.mark.parametrize("split", SPLITS)
def test_both_players_move_in_every_bucket(split):
    labels = load_suite(split).labels
    for b, name in enumerate(BUCKET_NAMES):
        parities = set((labels.ply[labels.bucket == b] % 2).tolist())
        assert parities == {0, 1}, name


@pytest.mark.parametrize("split", SPLITS)
def test_every_column_is_the_unique_best_move_often(split):
    labels = load_suite(split).labels
    nontrivial = ~labels.trivial
    unique = nontrivial & (labels.best.sum(axis=1) == 1)
    share = labels.best[unique].sum(axis=0) / nontrivial.sum()
    assert (share >= 0.02).all(), share


@pytest.mark.parametrize("split", SPLITS)
def test_oracle_scores_one_and_uniform_scores_the_best_share(split):
    labels = load_suite(split).labels
    oracle = tally_policy(top1_credit(labels.best.astype(np.float32), labels), labels)
    assert oracle.top1 == 1.0
    zeros = np.zeros((len(labels), 7), dtype=np.float32)
    uniform = tally_policy(top1_credit(zeros, labels), labels)
    share = labels.best.sum(axis=1) / labels.legal.sum(axis=1)
    assert uniform.top1 == pytest.approx(share[~labels.trivial].mean())
    assert uniform.top1 is not None and 0.2 < uniform.top1 < 0.35


def test_dev_and_test_share_no_move_strings():
    dev, test = (set(load_suite(s).labels.moves) for s in SPLITS)
    assert not dev & test


@pytest.mark.skipif(not SOLVER.is_file(), reason="solver not built in solver/")
@pytest.mark.parametrize("split", SPLITS)
def test_suite_rebuilds_byte_for_byte(split):
    labels = label(SUITE_V1[split], str(SOLVER), str(BOOK), n_procs=16)
    assert format_labels(labels) == suite_path(split).read_text()
