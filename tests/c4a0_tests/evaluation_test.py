import hashlib
import io
import pickle
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import pytest
import torch

from c4a0 import evaluation
from c4a0.evaluation import (
    SuiteEval,
    evaluate_model,
    illegal_argmax,
    tally_policy,
    tally_value,
    top1_credit,
)
from c4a0.labels import Suite
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0_tests.labels_test import PONS_ROWS, X, make_labels


def one_hot_ties(*cols: int) -> list[float]:
    return [1.0 if c in cols else 0.0 for c in range(7)]


def test_ties_split_credit_evenly() -> None:
    labels = make_labels({"3344": PONS_ROWS["3344"]})  # best {2, 5}
    policies = np.array(
        [
            one_hot_ties(2, 3),  # half of the tie is best
            one_hot_ties(2, 5),  # both tied moves are best
            one_hot_ties(5),
            one_hot_ties(3),  # a slower win gets nothing
            one_hot_ties(1, 2, 3, 4),
        ],
        dtype=np.float32,
    )
    for i, want in enumerate([0.5, 1.0, 1.0, 0.0, 0.25]):
        credit = top1_credit(policies[i : i + 1], labels)
        assert credit.tolist() == [want]


def test_illegal_argmax_is_masked_and_counted() -> None:
    labels = make_labels({"000000": PONS_ROWS["000000"], "": PONS_ROWS[""]})  # best {3}
    policies = np.array(
        [[9.0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 5, 0, 0, 0]], dtype=np.float32
    )
    assert top1_credit(policies, labels).tolist() == [1.0, 1.0]
    assert illegal_argmax(policies, labels) == 1


def test_log_probs_with_masked_minus_inf_columns() -> None:
    labels = make_labels({"000000": PONS_ROWS["000000"]})
    policies = np.array([[-np.inf, -2.3, -2.3, -0.5, -2.3, -2.3, -np.inf]], np.float32)
    assert top1_credit(policies, labels).tolist() == [1.0]


def test_nan_and_all_minus_inf_rows_are_rejected() -> None:
    labels = make_labels({"": PONS_ROWS[""]})
    with pytest.raises(ValueError):
        top1_credit(np.full((1, 7), np.nan, dtype=np.float32), labels)
    with pytest.raises(ValueError):
        top1_credit(np.full((1, 7), -np.inf, dtype=np.float32), labels)


def test_trivial_rows_leave_policy_denominators() -> None:
    labels = make_labels({"0": [-2] * 7, "3344": PONS_ROWS["3344"], "": PONS_ROWS[""]})
    credit = top1_credit(np.zeros((3, 7), dtype=np.float32), labels)
    tally = tally_policy(credit, labels)
    assert tally.by_bucket["00-05"].n == 2
    assert sum(t.n for t in tally.by_bucket.values()) == 2
    assert tally.by_bucket["06-11"].n == 0


def test_oracle_scores_one_and_uniform_scores_best_share() -> None:
    labels = make_labels(PONS_ROWS)
    oracle = labels.best.astype(np.float32)
    assert tally_policy(top1_credit(oracle, labels), labels).top1 == 1.0
    uniform = tally_policy(top1_credit(np.zeros((4, 7), np.float32), labels), labels)
    share = labels.best.sum(1) / labels.legal.sum(1)
    assert uniform.top1 == pytest.approx(share.mean())
    assert uniform.top1 == pytest.approx((1 / 7 + 2 / 7 + 1 / 6 + 1 / 7) / 4)


def test_value_confusion_uses_fixed_cutoffs() -> None:
    labels = make_labels(
        {
            "": PONS_ROWS[""],  # win
            "1": [0, 0, 0, -1, 0, 0, 0],  # draw
            "0": [-2] * 7,  # loss
            "2": [-3, -3, -3, -3, -3, -3, -3],  # loss
            "3": [X, 0, -1, -1, -1, -1, -1],  # draw
        }
    )
    q = np.array([0.9, 0.5, -0.51, -0.5, 0.0], dtype=np.float32)
    value = tally_value(q, labels)
    assert value.confusion == [[1, 1, 0], [0, 2, 0], [0, 0, 1]]
    assert value.accuracy == pytest.approx(4 / 5)


def test_evaluate_model_scores_the_pickled_bytes(tmp_path: Path) -> None:
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
    path.write_bytes(pickle.dumps(model))
    labels = make_labels({**PONS_ROWS, "0": [-2] * 7})
    suite = Suite(labels=labels, name="test/tiny", sha256="abc")

    result = evaluate_model(path, suite, (2, 8), 2.0, 0.01)

    assert result.model_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (result.suite, result.suite_sha256) == ("test/tiny", "abc")
    assert (result.n_positions, result.n_trivial) == (5, 1)
    assert sorted(result.mcts) == [2, 8]
    for policy_eval in [result.raw, *result.mcts.values()]:
        assert sum(t.n for t in policy_eval.by_bucket.values()) == 4
        assert policy_eval.top1 is not None and 0.0 <= policy_eval.top1 <= 1.0
    assert sum(map(sum, result.value.confusion)) == 5
    assert SuiteEval.model_validate_json(result.model_dump_json()) == result


def test_cpu_evaluation_matches_legacy_loading_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The committed pre-head-change model fixes weights and BatchNorm buffers.
    path = tmp_path / "model.pkl"
    path.write_bytes(
        (Path(__file__).parent / "fixtures" / "legacy_model.pkl").read_bytes()
    )
    suite = Suite(labels=make_labels(PONS_ROWS), name="test/tiny", sha256="abc")
    result = evaluate_model(path, suite, (2, 8), 2.0, 0.01, device="cpu")

    def legacy_load(
        source: str | Path | BinaryIO, device: str | torch.device
    ) -> ConnectFourNet:
        assert isinstance(source, io.BytesIO)
        assert device == "cpu"
        return cast(ConnectFourNet, pickle.load(source))

    monkeypatch.setattr(evaluation, "load_model", legacy_load)
    assert result == evaluate_model(path, suite, (2, 8), 2.0, 0.01)


def test_evaluation_scores_legacy_cuda_bytes_on_cpu(
    legacy_cuda_model: tuple[ConnectFourNet, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    model, path = legacy_cuda_model
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    suite = Suite(labels=make_labels(PONS_ROWS), name="test/tiny", sha256="abc")
    result = evaluate_model(path, suite, (4,), 2.0, 0.01)
    assert result.model_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    cpu_path = path.with_name("cpu-model.pkl")
    cpu_path.write_bytes(pickle.dumps(model))
    cpu_result = evaluate_model(cpu_path, suite, (4,), 2.0, 0.01)
    assert result.model_dump(exclude={"model_sha256"}) == cpu_result.model_dump(
        exclude={"model_sha256"}
    )


def test_non_finite_value_is_rejected_not_scored_as_draw() -> None:
    labels = make_labels({"3344": PONS_ROWS["3344"]})
    with pytest.raises(ValueError, match="non-finite"):
        tally_value(np.array([np.nan], dtype=np.float32), labels)


def test_openings_oracle_scores_full_and_column_zero_first_move_scores_zero() -> None:
    from c4a0.evaluation import OPENINGS

    oracle = OPENINGS.best.astype(np.float32)
    assert top1_credit(oracle, OPENINGS).sum() == len(OPENINGS)
    empty_board_col0 = top1_credit(
        np.array([one_hot_ties(0)], dtype=np.float32),
        OPENINGS.take(np.arange(len(OPENINGS)) == 0),
    )
    assert empty_board_col0.tolist() == [0.0]
    assert OPENINGS.best[0].tolist() == [c == 3 for c in range(7)]


SOLVER_BIN = (
    __import__("pathlib").Path(__file__).resolve().parents[2] / "solver" / "c4solver"
)


@pytest.mark.skipif(not SOLVER_BIN.is_file(), reason="solver not built in solver/")
def test_openings_scores_match_the_solver() -> None:
    import subprocess

    from c4a0.evaluation import OPENINGS

    book = SOLVER_BIN.parent / "7x6.book"
    lines = (
        "\n".join("".join(str(int(c) + 1) for c in m) for m in OPENINGS.moves) + "\n"
    )
    out = subprocess.run(
        [str(SOLVER_BIN), "-b", str(book), "-a"],
        input=lines,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = [
        [int(x) for x in line.split()[-7:]] for line in out.splitlines() if line.strip()
    ]
    assert rows == OPENINGS.scores.tolist()


def test_keeps_credits_a_slower_win_but_not_a_loss() -> None:
    labels = make_labels(
        {"3344": PONS_ROWS["3344"]}
    )  # wins: 2, 5 (fastest), 3, 4; losses: 0, 1, 6
    slower_win = np.array([one_hot_ties(3)], dtype=np.float32)
    losing = np.array([one_hot_ties(0)], dtype=np.float32)
    assert top1_credit(slower_win, labels).tolist() == [0.0]
    assert top1_credit(slower_win, labels, labels.keeps).tolist() == [1.0]
    assert top1_credit(losing, labels, labels.keeps).tolist() == [0.0]


def test_keeps_tally_skips_positions_where_every_move_keeps_the_result() -> None:
    # "01010": only column 0 wins, so a losing move throws the win away: counted.
    # A lost position (every move loses) cannot be thrown away: not counted.
    labels = make_labels(
        {"01010": PONS_ROWS["01010"], "6666": [-3, -2, -5, -4, -2, -6, -3]}
    )
    assert labels.keeps_trivial.tolist() == [False, True]
    credit = top1_credit(
        np.array([one_hot_ties(0), one_hot_ties(1)], dtype=np.float32),
        labels,
        labels.keeps,
    )
    tally = tally_policy(credit, labels, ~labels.keeps_trivial)
    assert sum(t.n for t in tally.by_bucket.values()) == 1
    assert tally.top1 == 1.0
