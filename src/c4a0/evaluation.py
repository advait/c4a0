"""
Scores a saved model against a solver-labeled suite: how often its move is solver-optimal.

A row's credit is |A & B| / |A|, where A is the set of legal moves tied for the policy's maximum
and B is the set of solver-best moves. Slower wins get no credit. Trivial rows (every legal move
is best) are left out of policy tallies. Tallies are stored, not rates, so runs can be merged and
intervals derived.
"""

import hashlib
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, computed_field

import c4a0_rust
from c4a0.labels import BUCKET_NAMES, Labels, Moves, Recipe, Split, Suite
from c4a0.utils import load_model


class Tally(BaseModel):
    hits: float
    """Sum of per-row credit; ties make it fractional."""
    n: int


class PolicyEval(BaseModel):
    by_bucket: dict[str, Tally]
    """Ply bucket name -> tally over non-trivial rows."""

    @computed_field
    @property
    def top1(self) -> float | None:
        n = sum(t.n for t in self.by_bucket.values())
        return sum(t.hits for t in self.by_bucket.values()) / n if n else None


ValueDecoder = Literal["cutoff_0.5_v1"]


class ValueEval(BaseModel):
    decoder: ValueDecoder = "cutoff_0.5_v1"
    """q_no_penalty < -0.5 is a loss, > 0.5 a win, otherwise a draw."""
    confusion: list[list[int]]
    """[solver loss, draw, win][predicted loss, draw, win] counts over all rows."""

    @computed_field
    @property
    def accuracy(self) -> float | None:
        n = sum(map(sum, self.confusion))
        return sum(self.confusion[i][i] for i in range(3)) / n if n else None


class SuiteEval(BaseModel):
    """One saved model scored on one suite."""

    suite: str
    suite_sha256: str
    model_sha256: str
    """Of the model.pkl bytes that were scored."""
    n_positions: int
    n_trivial: int
    c_exploration: float
    c_ply_penalty: float
    raw: PolicyEval
    """Raw network policy, illegal moves masked as in play."""
    illegal_argmax: int
    """Rows whose unmasked raw policy maximum includes a full column."""
    mcts: dict[int, PolicyEval] = {}
    raw_keeps: PolicyEval | None = None
    """Raw policy, credit for any move that keeps the game result, over rows where some legal
    move would throw it (`Labels.keeps_trivial` rows are excluded)."""
    mcts_keeps: dict[int, PolicyEval] = {}
    """MCTS iterations -> keeps-the-result eval of the root visit policy (the convergence
    predicate)."""
    """MCTS iterations -> eval of the root visit policy."""
    value: ValueEval
    """Raw network q_no_penalty."""
    openings_raw: Tally | None = None
    """Raw policy on OPENINGS (plies 0-1, which hash to train and so are absent from dev/test)."""
    openings_mcts: dict[int, Tally] = {}


# Plies 0 and 1 have 5 mirror classes, all in the train split, so no held-out suite can contain
# them. They are checked here separately. Scores are `c4solver -a` output for "" and "1".."7".
OPENINGS = Labels(
    recipe=Recipe(version=1, split=Split.TRAIN, seed=0, per_ply=0, plies=(0, 1)),
    moves=tuple(Moves(m) for m in ("", "0", "1", "2", "3", "4", "5", "6")),
    scores=np.array(
        [
            [-2, -1, 0, 1, 0, -1, -2],
            [-1, 2, 1, 2, -1, 1, -2],
            [-2, 0, 1, 0, -2, -2, -3],
            [-2, -2, 0, 0, 0, 0, -3],
            [-4, -2, -2, -1, -2, -2, -4],
            [-3, 0, 0, 0, 0, -2, -2],
            [-3, -2, -2, 0, 1, 0, -2],
            [-2, 1, -1, 2, 1, 2, -1],
        ],
        dtype=np.int8,
    ),
)


def top1_credit(
    policies: NDArray[np.floating],
    labels: Labels,
    target: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Per-row credit |A & target| / |A|, A = legal moves tied for the policy maximum.
    `target` defaults to the solver-best moves (strict); pass `labels.keeps` for the
    keeps-the-result metric."""
    target = labels.best if target is None else target
    if np.isnan(policies).any():
        raise ValueError("policy contains NaN")
    masked = np.where(labels.legal, policies, -np.inf)
    top = masked.max(axis=1, keepdims=True)
    if np.isneginf(top).any():
        raise ValueError("policy is -inf on every legal move")
    argmax = labels.legal & (masked == top)
    return (argmax & target).sum(axis=1) / argmax.sum(axis=1)


def illegal_argmax(policies: NDArray[np.floating], labels: Labels) -> int:
    top = policies.max(axis=1, keepdims=True)
    return int(((policies == top) & ~labels.legal).any(axis=1).sum())


def tally_policy(
    credit: NDArray[np.float64],
    labels: Labels,
    counted: NDArray[np.bool_] | None = None,
) -> PolicyEval:
    """Per-bucket tallies over `counted` rows (default: rows where the strict metric can miss)."""
    counted = ~labels.trivial if counted is None else counted
    return PolicyEval(
        by_bucket={
            name: Tally(
                hits=float(credit[counted & (labels.bucket == b)].sum()),
                n=int((counted & (labels.bucket == b)).sum()),
            )
            for b, name in enumerate(BUCKET_NAMES)
        }
    )


def tally_value(q_no_penalty: NDArray[np.floating], labels: Labels) -> ValueEval:
    if not np.isfinite(q_no_penalty).all():
        raise ValueError("q_no_penalty contains a non-finite value")
    pred = np.where(q_no_penalty < -0.5, 0, np.where(q_no_penalty > 0.5, 2, 1))
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels.value.astype(np.int64) + 1, pred), 1)
    return ValueEval(confusion=confusion.tolist())


def evaluate_model(
    model_path: Path,
    suite: Suite,
    mcts_iterations: Sequence[int],
    c_exploration: float,
    c_ply_penalty: float,
    device: str | torch.device = "cpu",
) -> SuiteEval:
    """Scores the model pickled at `model_path` on `suite`: raw policy and value on every row,
    plus root-visit MCTS policies at each budget in `mcts_iterations` on non-trivial rows."""
    model_bytes = model_path.read_bytes()
    model = load_model(io.BytesIO(model_bytes), device)
    labels = suite.labels

    policy, _q_penalty, q_no_penalty = model.forward_numpy(
        c4a0_rust.encode_positions(labels.moves)
    )

    nontrivial = labels.take(~labels.trivial)
    mcts = {}
    mcts_keeps: dict[int, PolicyEval] = {}
    for n in mcts_iterations:
        visits = c4a0_rust.mcts_policies(
            nontrivial.moves,
            n,
            c_exploration,
            c_ply_penalty,
            lambda _model_id, pos: model.forward_numpy(pos),
        )
        mcts[n] = tally_policy(top1_credit(visits, nontrivial), nontrivial)
        mcts_keeps[n] = tally_policy(
            top1_credit(visits, nontrivial, nontrivial.keeps),
            nontrivial,
            ~nontrivial.keeps_trivial,
        )

    def tally_openings(policies: NDArray[np.floating]) -> Tally:
        return Tally(hits=float(top1_credit(policies, OPENINGS).sum()), n=len(OPENINGS))

    openings_input = c4a0_rust.encode_positions(OPENINGS.moves)
    openings_raw = tally_openings(model.forward_numpy(openings_input)[0])
    openings_mcts = {
        n: tally_openings(
            c4a0_rust.mcts_policies(
                OPENINGS.moves,
                n,
                c_exploration,
                c_ply_penalty,
                lambda _model_id, pos: model.forward_numpy(pos),
            )
        )
        for n in mcts_iterations
    }

    return SuiteEval(
        suite=suite.name,
        suite_sha256=suite.sha256,
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        n_positions=len(labels),
        n_trivial=int(labels.trivial.sum()),
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        raw=tally_policy(top1_credit(policy, labels), labels),
        raw_keeps=tally_policy(
            top1_credit(policy, labels, labels.keeps), labels, ~labels.keeps_trivial
        ),
        mcts_keeps=mcts_keeps,
        illegal_argmax=illegal_argmax(policy, labels),
        mcts=mcts,
        value=tally_value(q_no_penalty, labels),
        openings_raw=openings_raw,
        openings_mcts=openings_mcts,
    )
