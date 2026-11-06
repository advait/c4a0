import hashlib
import pickle

import numpy as np
import pytest

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


def test_ties_split_credit_evenly():
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


def test_illegal_argmax_is_masked_and_counted():
    labels = make_labels({"000000": PONS_ROWS["000000"], "": PONS_ROWS[""]})  # best {3}
    policies = np.array(
        [[9.0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 5, 0, 0, 0]], dtype=np.float32
    )
    assert top1_credit(policies, labels).tolist() == [1.0, 1.0]
    assert illegal_argmax(policies, labels) == 1


def test_log_probs_with_masked_minus_inf_columns():
    labels = make_labels({"000000": PONS_ROWS["000000"]})
    policies = np.array([[-np.inf, -2.3, -2.3, -0.5, -2.3, -2.3, -np.inf]], np.float32)
    assert top1_credit(policies, labels).tolist() == [1.0]


def test_nan_and_all_minus_inf_rows_are_rejected():
    labels = make_labels({"": PONS_ROWS[""]})
    with pytest.raises(ValueError):
        top1_credit(np.full((1, 7), np.nan, dtype=np.float32), labels)
    with pytest.raises(ValueError):
        top1_credit(np.full((1, 7), -np.inf, dtype=np.float32), labels)


def test_trivial_rows_leave_policy_denominators():
    labels = make_labels({"0": [-2] * 7, "3344": PONS_ROWS["3344"], "": PONS_ROWS[""]})
    credit = top1_credit(np.zeros((3, 7), dtype=np.float32), labels)
    tally = tally_policy(credit, labels)
    assert tally.by_bucket["00-05"].n == 2
    assert sum(t.n for t in tally.by_bucket.values()) == 2
    assert tally.by_bucket["06-11"].n == 0


def test_oracle_scores_one_and_uniform_scores_best_share():
    labels = make_labels(PONS_ROWS)
    oracle = labels.best.astype(np.float32)
    assert tally_policy(top1_credit(oracle, labels), labels).top1 == 1.0
    uniform = tally_policy(top1_credit(np.zeros((4, 7), np.float32), labels), labels)
    share = labels.best.sum(1) / labels.legal.sum(1)
    assert uniform.top1 == pytest.approx(share.mean())
    assert uniform.top1 == pytest.approx((1 / 7 + 2 / 7 + 1 / 6 + 1 / 7) / 4)


def test_value_confusion_uses_fixed_cutoffs():
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


def test_evaluate_model_scores_the_pickled_bytes(tmp_path):
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

    result = evaluate_model(path, suite, (2, 8), 6.6, 0.01)

    assert result.model_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (result.suite, result.suite_sha256) == ("test/tiny", "abc")
    assert (result.n_positions, result.n_trivial) == (5, 1)
    assert sorted(result.mcts) == [2, 8]
    for policy_eval in [result.raw, *result.mcts.values()]:
        assert sum(t.n for t in policy_eval.by_bucket.values()) == 4
        assert policy_eval.top1 is not None and 0.0 <= policy_eval.top1 <= 1.0
    assert sum(map(sum, result.value.confusion)) == 5
    assert SuiteEval.model_validate_json(result.model_dump_json()) == result
