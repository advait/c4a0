import numpy as np
import pytest
import torch

from c4a0.evaluation import tally_policy, top1_credit
from c4a0.labels import Split, load_suite
from c4a0.nn import ConnectFourNet, ModelConfig
from c4a0.supervised import fit_supervised, solver_targets, terminal_ply
from c4a0_tests.labels_test import make_labels


@pytest.mark.parametrize(
    "moves, score, ply, want",
    [
        # Empty board, Pons best 1: the first player wins with their last stone, ply 41.
        ("", 1, 0, 41),
        # "010101", best 18: the mover completes four in column 0 now.
        ("010101", 18, 6, 7),
        # "01010", best 1: the second player blocks and wins with their 21st stone.
        ("01010", 1, 5, 42),
        # "01010", -18 elsewhere: the first player wins next move.
        ("01010", -18, 5, 7),
        # "101020", -18 for non-blocking moves: the second player wins next move.
        ("101020", -18, 6, 8),
        # "1020406", best 18: the second player completes column 0 now.
        ("1020406", 18, 7, 8),
        # "1020406", -17 except column 3: the first player completes the bottom row next.
        ("1020406", -17, 7, 9),
    ],
)
def test_terminal_ply_matches_golden_pons_rows(
    moves: str, score: int, ply: int, want: int
) -> None:
    assert len(moves) == ply
    assert terminal_ply(np.array([score]), np.array([ply])).tolist() == [want]


def test_solver_targets() -> None:
    labels = make_labels(
        {
            "3344": [-4, -3, 18, 2, 2, 18, -3],  # double threat, win at ply 7
            "101020": [0, -18, -18, -18, -18, -18, -18],  # draw
            "01010": [1, -18, -18, -18, -18, -18, -18],  # win at ply 42
            "1020406": [-17, -17, -17, -17, -17, -17, -17],  # synthetic: loss at ply 9
        }
    )
    pos, policy, q_penalty, q_no_penalty = solver_targets(labels, 0.01)
    assert pos.shape == (4, 2, 6, 7)
    assert policy[0].tolist() == [0, 0, 0.5, 0, 0, 0.5, 0]
    assert policy[1].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert policy[3].tolist() == pytest.approx([1 / 7] * 7)
    assert q_no_penalty.tolist() == [1, 0, 1, -1]
    assert q_penalty.tolist() == pytest.approx([0.93, 0.0, 0.58, -0.91])


def test_tiny_net_overfits_solver_labels() -> None:
    torch.manual_seed(0)
    dev = load_suite(Split.DEV).labels
    rows = dev.take(np.flatnonzero(~dev.trivial)[:256])
    model = ConnectFourNet(
        ModelConfig(
            n_residual_blocks=1,
            conv_filter_size=16,
            n_policy_layers=2,
            n_value_layers=2,
            lr_schedule={0: 3e-3},
            l2_reg=0.0,
        )
    )
    fitted, epochs = fit_supervised(
        model, rows, rows, 0.01, batch_size=64, max_epochs=15, patience=15
    )
    assert epochs > 0
    policy, _, q = fitted.forward_numpy(solver_targets(rows, 0.01)[0].numpy())
    trained = tally_policy(top1_credit(policy, rows), rows).top1
    uniform = tally_policy(top1_credit(np.zeros_like(policy), rows), rows).top1
    assert uniform is not None and trained is not None
    assert trained > uniform + 0.5, (trained, uniform)
