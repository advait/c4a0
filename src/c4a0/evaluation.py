"""
Scores a saved model against a solver-labeled suite: how often its move is solver-optimal.

A row's credit is |A & B| / |A|, where A is the set of legal moves tied for the policy's maximum
and B is the set of solver-best moves. Slower wins get no credit. Trivial rows (every legal move
is best) are left out of policy tallies. Tallies are stored, not rates, so runs can be merged and
intervals derived.
"""

import hashlib
from pathlib import Path
import pickle
from typing import Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, computed_field

from c4a0.labels import BUCKET_NAMES, Labels, Suite
from c4a0.nn import ConnectFourNet

import c4a0_rust


class Tally(BaseModel):
    hits: float
    """Sum of per-row credit; ties make it fractional."""
    n: int


class PolicyEval(BaseModel):
    by_bucket: dict[str, Tally]
    """Ply bucket name -> tally over non-trivial rows."""

    @computed_field
    @property
    def top1(self) -> Optional[float]:
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
    def accuracy(self) -> Optional[float]:
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
    """MCTS iterations -> eval of the root visit policy."""
    value: ValueEval
    """Raw network q_no_penalty."""


def top1_credit(policies: NDArray[np.floating], labels: Labels) -> NDArray[np.float64]:
    """Per-row credit |A & best| / |A|, A = legal moves tied for the policy maximum."""
    if np.isnan(policies).any():
        raise ValueError("policy contains NaN")
    masked = np.where(labels.legal, policies, -np.inf)
    top = masked.max(axis=1, keepdims=True)
    if np.isneginf(top).any():
        raise ValueError("policy is -inf on every legal move")
    argmax = labels.legal & (masked == top)
    return (argmax & labels.best).sum(axis=1) / argmax.sum(axis=1)


def illegal_argmax(policies: NDArray[np.floating], labels: Labels) -> int:
    top = policies.max(axis=1, keepdims=True)
    return int(((policies == top) & ~labels.legal).any(axis=1).sum())


def tally_policy(credit: NDArray[np.float64], labels: Labels) -> PolicyEval:
    counted = ~labels.trivial
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
) -> SuiteEval:
    """Scores the model pickled at `model_path` on `suite`: raw policy and value on every row,
    plus root-visit MCTS policies at each budget in `mcts_iterations` on non-trivial rows."""
    model_bytes = model_path.read_bytes()
    model: ConnectFourNet = pickle.loads(model_bytes)
    labels = suite.labels

    policy, _q_penalty, q_no_penalty = model.forward_numpy(
        c4a0_rust.encode_positions(labels.moves)
    )

    nontrivial = labels.take(~labels.trivial)
    mcts = {}
    for n in mcts_iterations:
        visits = c4a0_rust.mcts_policies(
            nontrivial.moves,
            n,
            c_exploration,
            c_ply_penalty,
            lambda _model_id, pos: model.forward_numpy(pos),
        )
        mcts[n] = tally_policy(top1_credit(visits, nontrivial), nontrivial)

    return SuiteEval(
        suite=suite.name,
        suite_sha256=suite.sha256,
        model_sha256=hashlib.sha256(model_bytes).hexdigest(),
        n_positions=len(labels),
        n_trivial=int(labels.trivial.sum()),
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        raw=tally_policy(top1_credit(policy, labels), labels),
        illegal_argmax=illegal_argmax(policy, labels),
        mcts=mcts,
        value=tally_value(q_no_penalty, labels),
    )
