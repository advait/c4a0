"""
Solver-labeled positions: the labels file format, the committed evaluation suites, and labeling
with Pascal Pons's solver. The (N, 7) Pons score matrix is the only stored solver fact; legal,
best, trivial, value and ply bucket are derived from it.
"""

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
import hashlib
import os
from pathlib import Path
from typing import NewType

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

import c4a0_rust

Moves = NewType("Moves", str)
"""0-indexed column digits from the empty board. Only Rust interprets them."""


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class Recipe(BaseModel, frozen=True):
    """How a labels file was sampled. Membership in `split` is decided in Rust (`Split::of`)."""

    version: int
    split: Split
    seed: int
    per_ply: int
    plies: tuple[int, int]


PLY_BUCKETS = (0, 6, 12, 18, 24, 30, 36, 42)
BUCKET_NAMES = tuple(
    f"{lo:02d}-{hi - 1:02d}" for lo, hi in zip(PLY_BUCKETS, PLY_BUCKETS[1:])
)
ILLEGAL = np.int8(-128)
N_COLS = 7

_HEADER = "# c4a0-labels "
_COLUMNS = "moves\tscores"


@dataclass(frozen=True)
class Labels:
    """Solver-labeled positions, columnar. `scores` is the only stored fact."""

    recipe: Recipe
    moves: tuple[Moves, ...]
    scores: NDArray[np.int8]
    """(N, 7) Pons scores for playing each column, from the mover's view. ILLEGAL = full column."""

    def __post_init__(self):
        n = len(self.moves)
        if self.scores.dtype != np.int8 or self.scores.shape != (n, N_COLS):
            raise ValueError(
                f"scores must be int8 ({n}, {N_COLS}), got {self.scores.dtype} {self.scores.shape}"
            )
        if n and not self.legal.any(axis=1).all():
            raise ValueError("every row needs a legal move")

    def __len__(self) -> int:
        return len(self.moves)

    @cached_property
    def legal(self) -> NDArray[np.bool_]:
        return self.scores != ILLEGAL

    @cached_property
    def best_score(self) -> NDArray[np.int8]:
        """The position's Pons score: the best legal move's score (ILLEGAL is below all)."""
        return self.scores.max(axis=1)

    @cached_property
    def best(self) -> NDArray[np.bool_]:
        return self.legal & (self.scores == self.best_score[:, None])

    @cached_property
    def value(self) -> NDArray[np.int8]:
        """-1 loss, 0 draw, +1 win for the player to move."""
        return np.sign(self.best_score).astype(np.int8)

    @cached_property
    def trivial(self) -> NDArray[np.bool_]:
        """Every legal move is best, so the policy cannot be wrong."""
        return (self.best == self.legal).all(axis=1)

    @cached_property
    def ply(self) -> NDArray[np.int64]:
        return np.array([len(m) for m in self.moves], dtype=np.int64)

    @cached_property
    def bucket(self) -> NDArray[np.int64]:
        """Index into BUCKET_NAMES."""
        return np.searchsorted(PLY_BUCKETS, self.ply, side="right") - 1

    def take(self, idx: NDArray[np.int64] | NDArray[np.bool_]) -> "Labels":
        rows = np.arange(len(self))[idx]
        return Labels(
            recipe=self.recipe,
            moves=tuple(self.moves[i] for i in rows),
            scores=self.scores[rows],
        )


@dataclass(frozen=True)
class Suite:
    labels: Labels
    name: str
    """e.g. "v1/dev"."""
    sha256: str
    """Of the file bytes."""


SUITE_DIR = Path(__file__).resolve().parent / "suite"
SUITE_V1 = {
    Split.DEV: Recipe(version=1, split=Split.DEV, seed=1, per_ply=64, plies=(0, 41)),
    Split.TEST: Recipe(version=1, split=Split.TEST, seed=2, per_ply=64, plies=(0, 41)),
}


def suite_path(split: Split, version: int = 1) -> Path:
    return SUITE_DIR / f"v{version}" / f"{split}.tsv"


def load_suite(split: Split, version: int = 1) -> Suite:
    """Loads a committed evaluation suite and checks it matches its recipe."""
    if version != 1:
        raise ValueError(f"unknown suite version {version}")
    data = suite_path(split, version).read_bytes()
    labels = parse_labels(data.decode())
    if labels.recipe != SUITE_V1[split]:
        raise ValueError(f"suite v{version}/{split} recipe mismatch: {labels.recipe}")
    return Suite(
        labels=labels,
        name=f"v{version}/{split}",
        sha256=hashlib.sha256(data).hexdigest(),
    )


def format_labels(labels: Labels) -> str:
    lines = [_HEADER + labels.recipe.model_dump_json(), _COLUMNS]
    for moves, row in zip(labels.moves, labels.scores):
        scores = ",".join("x" if s == ILLEGAL else str(s) for s in row)
        lines.append(f"{moves}\t{scores}")
    return "\n".join(lines) + "\n"


def parse_labels(text: str) -> Labels:
    lines = text.split("\n")
    if not lines[0].startswith(_HEADER) or lines[1] != _COLUMNS or lines[-1] != "":
        raise ValueError("not a c4a0 labels file")
    recipe = Recipe.model_validate_json(lines[0][len(_HEADER) :])
    moves: list[Moves] = []
    scores: list[list[int]] = []
    for i, line in enumerate(lines[2:-1], start=3):
        m, _, s = line.partition("\t")
        row = s.split(",")
        if not set(m) <= set("0123456") or len(row) != N_COLS:
            raise ValueError(f"line {i}: malformed row {line!r}")
        moves.append(Moves(m))
        try:
            scores.append([int(ILLEGAL) if v == "x" else int(v) for v in row])
        except ValueError:
            raise ValueError(f"line {i}: malformed scores {s!r}") from None
    if any(not -21 <= v <= 21 for row in scores for v in row if v != ILLEGAL):
        raise ValueError("score out of range")
    return Labels(
        recipe=recipe,
        moves=tuple(moves),
        scores=np.array(scores, dtype=np.int8).reshape(-1, N_COLS),
    )


def read_labels(path: Path) -> Labels:
    return parse_labels(path.read_text())


def write_labels(path: Path, labels: Labels) -> None:
    """Writes atomically: a temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(format_labels(labels))
    os.replace(tmp, path)


def label(recipe: Recipe, solver_path: str, book_path: str, n_procs: int) -> Labels:
    """Samples `recipe` positions in Rust and labels them with Pons's solver."""
    first_ply, last_ply = recipe.plies
    rows = c4a0_rust.label_positions(
        recipe.split.value,
        first_ply,
        last_ply,
        recipe.per_ply,
        recipe.seed,
        solver_path,
        book_path,
        n_procs,
    )
    scores = [[int(ILLEGAL) if v is None else v for v in s] for _, s in rows]
    return Labels(
        recipe=recipe,
        moves=tuple(Moves(m) for m, _ in rows),
        scores=np.array(scores, dtype=np.int8).reshape(-1, N_COLS),
    )
