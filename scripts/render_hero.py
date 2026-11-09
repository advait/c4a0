"""Render the README hero GIF: real MCTS overturning the network's first instinct, walk by walk.

Usage: uv run --locked python scripts/render_hero.py [MODEL] [--out PATH]

MODEL defaults to models/c4a0-final.pkl and the GIF goes to images/hero.gif. Every number drawn
comes from the network (`forward_numpy`), the Rust search (`c4a0_rust.mcts_policies`, called
once per budget n = 2..N_ITERS) or the committed dev suite's solver scores. The script asserts
each claim its captions make before drawing anything. Text uses only the DejaVu fonts bundled
with matplotlib, so the output is byte-identical wherever the locked environment runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import c4a0_rust
from c4a0.labels import Split, load_suite
from c4a0.nn import ConnectFourNet
from c4a0.utils import load_model

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "c4a0-final.pkl"
DEFAULT_OUT = ROOT / "images" / "hero.gif"
MOVES = "0042021361344511"  # dev suite, ply 16, first player (Red) to move
N_ITERS = 256  # the README's MCTS-256 budget
C_EXPLORATION, C_PLY_PENALTY, LEAF_MINIMAX = (
    2.0,
    0.01,
    0,
)  # eval defaults; plain NN-guided MCTS

FloatArray = NDArray[np.float32]
EvalResult = tuple[FloatArray, FloatArray, FloatArray]
RGB = tuple[int, int, int]

# ----------------------------------------------------------------------------------------------
# Data: the real network and the real search, verified
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Search:
    prior: NDArray[np.float64]  # network policy at the root
    params: int
    counts: NDArray[np.int64]  # (N_ITERS + 1, 7): root child visits after n iterations
    visited: NDArray[
        np.int64
    ]  # (N_ITERS + 1,): the root child walk n went down (n >= 2)
    scores: NDArray[np.int64]  # solver score per column, mover's view
    legal: NDArray[np.bool_]
    board: NDArray[np.int64]  # (6, 7), row 0 = bottom, 1 = first player, 2 = second
    mover: int
    best: int  # the solver's unique winning column
    fav: int  # the prior's favourite
    cross: int  # first iteration from which `best` has the most visits through N_ITERS
    evidence: dict[str, object]

    def share(self, n: int) -> NDArray[np.float64]:
        return self.counts[n] / (n - 1)  # iteration 1 evaluates the root itself


def replay(moves: str) -> NDArray[np.int64]:
    board = np.zeros((6, 7), np.int64)
    for ply, ch in enumerate(moves):
        col = int(ch)
        board[int(np.count_nonzero(board[:, col])), col] = 1 + ply % 2
    return board


def compute(model_path: Path) -> Search:
    torch.manual_seed(0)
    torch.set_num_threads(1)  # batch-1 inference; one thread keeps float sums portable
    torch.use_deterministic_algorithms(True)
    model: ConnectFourNet = load_model(model_path)
    params = sum(p.numel() for p in model.parameters())

    enc = c4a0_rust.encode_positions([MOVES])
    board = replay(MOVES)
    mover = 1 + len(MOVES) % 2
    assert np.array_equal(enc[0, 0] == 1, board == mover), (
        "encoder channel 0 is the mover"
    )
    assert np.array_equal(enc[0, 1] == 1, board == 3 - mover), (
        "encoder channel 1 is the opponent"
    )
    logp, _, _ = model.forward_numpy(enc)
    prior = np.exp(logp[0].astype(np.float64))
    assert abs(prior.sum() - 1) < 1e-5

    # Memoise network outputs per position (the network is deterministic; a plain rerun below
    # proves the cache changes nothing) and record every leaf the search sends to the network.
    cache: dict[bytes, tuple[FloatArray, np.float32, np.float32]] = {}
    trace: list[bytes] = []

    def cached(_model_id: int, pos: FloatArray) -> EvalResult:
        out = []
        for row in pos:
            key = row.tobytes()
            trace.append(key)
            if key not in cache:
                p, q, q2 = model.forward_numpy(row[None])
                cache[key] = (p[0], q[0], q2[0])
            out.append(cache[key])
        return (
            np.stack([o[0] for o in out]),
            np.array([o[1] for o in out], np.float32),
            np.array([o[2] for o in out], np.float32),
        )

    def run(n: int) -> tuple[FloatArray, list[bytes]]:
        trace.clear()
        policy = c4a0_rust.mcts_policies(
            [MOVES], n, C_EXPLORATION, C_PLY_PENALTY, cached, LEAF_MINIMAX
        )[0]
        return policy, list(trace)

    final, full_trace = run(N_ITERS)
    assert len(full_trace) == N_ITERS, (
        "each iteration sends exactly one leaf to the network"
    )
    counts = np.zeros((N_ITERS + 1, 7), np.int64)
    for n in range(2, N_ITERS + 1):
        policy, leaves = run(n) if n < N_ITERS else (final, full_trace)
        # Snapshots of one growing search: every budget evaluates a prefix of the same leaves.
        assert leaves == full_trace[:n], f"budget {n} diverges from the full search"
        raw = policy.astype(np.float64) * (n - 1)
        counts[n] = np.rint(raw)
        assert np.abs(raw - counts[n]).max() < 1e-3, (
            f"non-integral visits at {n} (proven root?)"
        )
        assert counts[n].sum() == n - 1
    step = np.diff(counts[1:], axis=0)
    assert (step >= 0).all() and (step.sum(axis=1) == 1).all(), (
        "one root visit per iteration"
    )
    visited = np.zeros(N_ITERS + 1, np.int64)
    visited[2:] = step.argmax(axis=1)

    # The cache must be invisible: rerun the full search with the plain, uncached callback.
    plain = c4a0_rust.mcts_policies(
        [MOVES],
        N_ITERS,
        C_EXPLORATION,
        C_PLY_PENALTY,
        lambda _model_id, pos: model.forward_numpy(pos),
        LEAF_MINIMAX,
    )[0]
    assert np.array_equal(plain, final), "uncached search differs from the cached one"
    assert np.array_equal(
        np.rint(plain.astype(np.float64) * (N_ITERS - 1)), counts[N_ITERS]
    )

    suite = load_suite(Split.DEV)
    labels = suite.labels
    row = labels.moves.index(MOVES)
    scores = labels.scores[row].astype(np.int64)
    legal = labels.legal[row]
    fav = int(np.argmax(np.where(legal, prior, -1)))
    best = int(np.argmax(np.where(legal, scores, -999)))
    assert scores[fav] < 0, "caption: the prior's favourite loses"
    assert (scores[legal] > 0).sum() == 1 and scores[best] > 0, (
        "caption: the only winning move"
    )
    leads = np.array(
        [
            n >= 2 and counts[n, best] > np.delete(counts[n], best).max()
            for n in range(N_ITERS + 1)
        ]
    )
    assert leads[N_ITERS], "search must end on the solver's move"
    cross = int(np.flatnonzero(~leads)[-1]) + 1
    assert leads[cross:].all() and not leads[cross - 1]

    evidence: dict[str, object] = {
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "parameters": params,
        "suite": suite.name,
        "suite_sha256": suite.sha256,
        "moves_zero_indexed": MOVES,
        "c_exploration": C_EXPLORATION,
        "c_ply_penalty": C_PLY_PENALTY,
        "leaf_minimax_depth": LEAF_MINIMAX,
        "solver_scores": scores.tolist(),
        "network_prior": [round(float(p), 6) for p in prior],
        "visits_at": {str(N_ITERS): counts[N_ITERS].tolist()},
        "network_calls": len(full_trace),
        "unique_leaves": len(set(full_trace)),
        "best_leads_from_iteration": cross,
        "checks": [
            "every budget's leaves are a prefix of the full search's",
            "integral visits summing to n - 1, one new root visit per iteration",
            "uncached rerun bit-identical to the cached search",
            "board matches the Rust encoder",
        ],
    }
    return Search(
        prior,
        params,
        counts,
        visited,
        scores,
        legal,
        board,
        mover,
        best,
        fav,
        cross,
        evidence,
    )


# ----------------------------------------------------------------------------------------------
# Drawing (Pillow at 2x, downsampled)
# ----------------------------------------------------------------------------------------------

S = 2  # supersampling factor
W, H = 800, 684

BG: RGB = (13, 17, 32)
SLAB: RGB = (30, 40, 84)
SLAB_HI: RGB = (46, 60, 120)
HOLE: RGB = (8, 10, 20)
RED: RGB = (255, 92, 92)
GOLD: RGB = (255, 200, 74)
CYAN: RGB = (46, 230, 200)
LAV: RGB = (150, 160, 214)
INK: RGB = (232, 235, 250)
MUTED: RGB = (120, 128, 170)
DIM: RGB = (62, 70, 110)
LOSS: RGB = (196, 112, 136)
BLACK: RGB = (0, 0, 0)
WHITE: RGB = (255, 255, 255)
EMPTY_CELL: RGB = (22, 28, 50)
OTHER: RGB = (66, 76, 122)

# Only matplotlib's bundled DejaVu files: no ambient font lookup, same glyphs on every machine.
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(name: str, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / f"{name}.ttf"), int(size * S))


F_KICK = font("DejaVuSansMono-Bold", 11)
F_MONO = font("DejaVuSansMono", 13)
F_MONO_S = font("DejaVuSansMono", 11)
F_BIG = font("DejaVuSans-Bold", 88)  # tabular digits, undotted zero
F_HEAD = font("DejaVuSans-Bold", 30)
F_CAP = font("DejaVuSans-Bold", 17)
F_CAP2 = font("DejaVuSans", 13.5)
F_COL = font("DejaVuSansMono-Bold", 15)
F_TAG = font("DejaVuSansMono-Bold", 10.5)

CELL = 52
BX0 = 44  # board column 0 left edge
BAR_BASE, BAR_H = 304, 168
BOARD_Y = 332  # top of the slab
PANEL_X = 474
CAPTION_W = 304
WALK_COLS, WALK_CELL = 17, 14  # N_ITERS - 1 = 255 walks fill 17 x 15 exactly
WALK_X, WALK_Y = PANEL_X, 284


def col_cx(c: int) -> float:
    return BX0 + CELL * (c + 0.5)


def hole_cy(r: int) -> float:  # r = 0 is the bottom row
    return BOARD_Y + 12 + CELL * (5 - r + 0.5)


def mix(a: RGB, b: RGB, t: float) -> RGB:
    r, g, bl = (int(round(x + (y - x) * t)) for x, y in zip(a, b, strict=True))
    return (r, g, bl)


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


class Canvas:
    """A 2x canvas plus an additive glow layer, blurred and merged in `finish`."""

    def __init__(self) -> None:
        self.im = Image.new("RGB", (W * S, H * S), BG)
        self.d = ImageDraw.Draw(self.im)
        self.glow = Image.new("RGB", (W * S, H * S), BLACK)
        self.gd = ImageDraw.Draw(self.glow)

    def rect(
        self,
        box: tuple[float, float, float, float],
        fill: RGB | None = None,
        outline: RGB | None = None,
        width: float = 1,
        r: float = 0,
        glow: bool = False,
    ) -> None:
        d = self.gd if glow else self.d
        xy = [v * S for v in box]
        d.rounded_rectangle(xy, r * S, fill=fill, outline=outline, width=int(width * S))

    def circle(
        self,
        cx: float,
        cy: float,
        rad: float,
        fill: RGB | None = None,
        outline: RGB | None = None,
        width: float = 1,
        glow: bool = False,
    ) -> None:
        d = self.gd if glow else self.d
        xy = [(cx - rad) * S, (cy - rad) * S, (cx + rad) * S, (cy + rad) * S]
        d.ellipse(xy, fill=fill, outline=outline, width=int(width * S))

    def text(
        self,
        x: float,
        y: float,
        s: str,
        f: ImageFont.FreeTypeFont,
        fill: RGB,
        anchor: str = "la",
        spacing: float = 0,
    ) -> float:
        """Draw `s`; `spacing` letter-spaces it. Returns the x after the text."""
        if not spacing:
            self.d.text((x * S, y * S), s, font=f, fill=fill, anchor=anchor)
            return x + f.getlength(s) / S
        for ch in s:
            self.d.text((x * S, y * S), ch, font=f, fill=fill, anchor=anchor)
            x += f.getlength(ch) / S + spacing
        return x

    def finish(self) -> Image.Image:
        g = self.glow.filter(ImageFilter.GaussianBlur(7 * S))
        merged = np.asarray(self.im, np.int16) + np.asarray(g, np.int16)
        im = Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8))
        return im.resize((W, H), Image.Resampling.LANCZOS)


def stone(cv: Canvas, cx: float, cy: float, color: RGB) -> None:
    cv.circle(cx, cy, 19.5, fill=mix(color, BLACK, 0.25))
    cv.circle(cx, cy - 1, 18, fill=color)
    cv.circle(cx, cy - 1, 11, outline=mix(color, WHITE, 0.35), width=1.5)


def draw_board(
    cv: Canvas, D: Search, drop: float | None, hot: NDArray[np.float64]
) -> None:
    """The slab and stones. `drop` in [0, 2] drops the mover's stone into the solver's column."""
    x0, x1 = BX0 - 12, BX0 + 7 * CELL + 12
    y0, y1 = BOARD_Y, BOARD_Y + 6 * CELL + 24
    cv.rect((x0, y0 + 4, x1, y1 + 4), fill=(6, 8, 16), r=16)
    cv.rect((x0, y0, x1, y1), fill=SLAB, r=16)
    cv.rect((x0 + 2, y0 + 2, x1 - 2, y0 + 5), fill=SLAB_HI, r=2)
    for c in range(7):  # a wash over the columns the latest walks went down
        if hot[c] > 0.01:
            box = (BX0 + c * CELL + 4, y0 + 8, BX0 + (c + 1) * CELL - 4, y1 - 8)
            cv.rect(box, fill=mix(SLAB, CYAN, 0.22 * hot[c]), r=10)
    for c in range(7):
        for r in range(6):
            v = int(D.board[r, c])
            if v == 0:
                cv.circle(col_cx(c), hole_cy(r), 19.5, fill=HOLE)
            else:
                stone(cv, col_cx(c), hole_cy(r), RED if v == 1 else GOLD)
    if drop is not None:
        c = D.best
        top, land = BOARD_Y - 30, hole_cy(int(np.count_nonzero(D.board[:, c])))
        cy = top + (land - top) * min(drop, 1.0) ** 2
        if drop > 1.0:  # a small settle bounce
            b = min(drop - 1.0, 1.0)
            cy = land - 7 * math.sin(b * math.pi) * (1 - b)
        color = RED if D.mover == 1 else GOLD
        cv.circle(col_cx(c), cy, 21, fill=mix(BLACK, color, 0.5), glow=True)
        stone(cv, col_cx(c), cy, color)


def draw_bars(
    cv: Canvas, D: Search, n: int, hot: NDArray[np.float64], reveal: float
) -> None:
    """Hollow lavender outline = prior; cyan fill = share of search visits after n iterations."""
    share = D.share(n) if n >= 2 else None
    bw = CELL - 16
    for c in range(7):
        cx = col_cx(c)
        x0, x1 = cx - bw / 2, cx + bw / 2
        cv.rect((x0 - 3, BAR_BASE, x1 + 3, BAR_BASE + 2), fill=DIM)
        sh = 0.0 if share is None else float(share[c])
        lead = share is not None and sh == share.max() and sh > 0
        if sh * BAR_H > 0.5:
            fill = CYAN if lead else mix(CYAN, BG, 0.45)
            cv.rect((x0, BAR_BASE - BAR_H * sh, x1, BAR_BASE), fill=fill, r=3)
            if lead:
                glow = mix(BLACK, CYAN, 0.3)
                cv.rect(
                    (x0, BAR_BASE - BAR_H * sh, x1, BAR_BASE), fill=glow, r=3, glow=True
                )
        ph = BAR_H * float(D.prior[c])
        if ph > 1.5:
            box = (x0 - 2, BAR_BASE - ph, x1 + 2, BAR_BASE + 1)
            cv.rect(box, outline=LAV, width=1.6, r=4)
        if hot[c] > 0.01:  # spark on the child the latest walk went down
            top = BAR_BASE - BAR_H * sh - 12
            cv.circle(cx, top, 4.5, fill=mix(BG, WHITE, hot[c]))
            cv.circle(cx, top, 9, fill=mix(BLACK, CYAN, 0.6 * hot[c]), glow=True)
        if share is None and D.prior[c] >= 0.005:
            label = f"{round(D.prior[c] * 100)}%"
            cv.text(cx, BAR_BASE - ph - 8, label, F_MONO_S, LAV, anchor="ms")
        elif share is not None and sh >= 0.005:
            ytop = BAR_BASE - max(BAR_H * sh, ph) - 16 - (10 if hot[c] > 0.01 else 0)
            label = f"{round(sh * 100)}%"
            cv.text(cx, ytop, label, F_MONO_S, INK if lead else MUTED, anchor="ms")
        key_col = c in (D.best, D.fav)
        cv.text(
            cx, BAR_BASE + 20, str(c + 1), F_COL, INK if key_col else MUTED, anchor="ms"
        )
        if reveal > 0 and D.legal[c]:
            a = ease(reveal)
            s = int(D.scores[c])
            y = 120
            if s > 0:
                cv.rect((cx - 20, y - 11, cx + 20, y + 6), fill=mix(BG, CYAN, a), r=4)
                glow = mix(BLACK, CYAN, 0.35 * a)
                cv.rect((cx - 20, y - 11, cx + 20, y + 6), fill=glow, r=4, glow=True)
                cv.text(cx, y + 1, "WIN", F_TAG, BG, anchor="ms")
            else:
                tag, color = ("DRAW", LAV) if s == 0 else ("LOSS", LOSS)
                cv.text(cx, y + 1, tag, F_TAG, mix(BG, color, 0.9 * a), anchor="ms")


def walk_color(D: Search, c: int) -> RGB:
    return CYAN if c == D.best else LAV if c == D.fav else OTHER


def draw_walks(cv: Canvas, D: Search, n: int) -> None:
    """One square per iteration 2..N_ITERS, in order, coloured by the root move it went down."""
    cv.text(WALK_X, WALK_Y - 22, "EVERY WALK, IN ORDER", F_KICK, MUTED, spacing=1.6)
    rows = math.ceil((N_ITERS - 1) / WALK_COLS)
    for it in range(2, N_ITERS + 1):
        r, c = divmod(it - 2, WALK_COLS)
        x, y = WALK_X + c * WALK_CELL, WALK_Y + r * WALK_CELL
        fill = walk_color(D, int(D.visited[it])) if it <= n else EMPTY_CELL
        cv.rect(
            (x + 1.5, y + 1.5, x + WALK_CELL - 1.5, y + WALK_CELL - 1.5),
            fill=fill,
            r=1.5,
        )
        ring = (x - 0.5, y - 0.5, x + WALK_CELL + 0.5, y + WALK_CELL + 0.5)
        if (
            it == D.cross and n >= D.cross
        ):  # the walk from which column `best` leads for good
            cv.rect(ring, outline=WHITE, width=2, r=3)
            cv.rect(ring, fill=mix(BLACK, CYAN, 0.6), r=3, glow=True)
        elif it == n and n < N_ITERS:
            cv.rect(ring, outline=INK, width=1.4, r=2.5)
    ky = WALK_Y + rows * WALK_CELL + 16
    x = WALK_X
    for c, label in (
        (D.best, f"col {D.best + 1}"),
        (D.fav, f"col {D.fav + 1}"),
        (-1, "other"),
    ):
        cv.rect((x, ky - 5, x + 10, ky + 5), fill=walk_color(D, c), r=1.5)
        color = walk_color(D, c) if c >= 0 else MUTED
        x = cv.text(x + 15, ky, label, F_MONO_S, color, anchor="lm") + 14
    cv.rect((x + 4, ky - 6, x + 16, ky + 6), outline=LAV, width=1.6, r=2)
    cv.text(x + 22, ky, "prior", F_MONO_S, LAV, anchor="lm")


def wrap(s: str, f: ImageFont.FreeTypeFont, width: float) -> list[str]:
    out: list[str] = []
    cur = ""
    for word in s.split():
        t = f"{cur} {word}".strip()
        if f.getlength(t) / S <= width:
            cur = t
        else:
            out.append(cur)
            cur = word
    return [*out, cur] if cur else out


def frame(
    D: Search,
    *,
    n: int,
    hot: NDArray[np.float64],
    caption: tuple[str, str],
    reveal: float = 0.0,
    drop: float | None = None,
) -> Image.Image:
    cv = Canvas()
    x = cv.text(BX0 - 12, 26, "C4A0", F_KICK, CYAN, spacing=3)
    tagline = (
        f"·  {round(D.params / 1000)}K-PARAMETER NETWORK  ·  TRAINED BY SELF-PLAY ONLY"
    )
    cv.text(x + 8, 26, tagline, F_KICK, MUTED, spacing=1.2)
    cv.text(BX0 - 13, 88, "Watch it change its mind.", F_HEAD, INK, anchor="ls")

    draw_bars(cv, D, n, hot, reveal)
    draw_board(cv, D, drop, hot)

    cv.text(PANEL_X, 128, "MCTS ITERATION", F_KICK, MUTED, spacing=1.6)
    cv.text(PANEL_X + 184, 216, str(n), F_BIG, CYAN if n else DIM, anchor="rs")
    cv.text(PANEL_X + 194, 212, f"/ {N_ITERS}", F_MONO, MUTED, anchor="ls")
    if n >= D.cross:
        mark = f"COLUMN {D.best + 1} LEADS FROM ITERATION {D.cross}"
        cv.text(PANEL_X, 234, mark, F_KICK, CYAN, spacing=1.2)
    draw_walks(cv, D, n)
    cv.rect((0, 0, W - 1, H - 1), outline=(30, 37, 66), width=1)

    head, body = caption
    y = 540
    for ln in wrap(head, F_CAP, CAPTION_W):
        cv.text(PANEL_X, y, ln, F_CAP, INK)
        y += 23
    y += 6
    for ln in wrap(body, F_CAP2, CAPTION_W):
        cv.text(PANEL_X, y, ln, F_CAP2, MUTED)
        y += 19
    assert y <= H - 8, f"caption overflows: {caption}"
    return cv.finish()


# ----------------------------------------------------------------------------------------------
# Timeline and encoding
# ----------------------------------------------------------------------------------------------


def schedule(cross: int) -> list[int]:
    """Budgets to show: every iteration at first, then every other, then accelerating."""
    ns = [*range(2, 31), *range(32, 121, 2), cross - 1, cross]
    x = 120.0
    while x < N_ITERS:
        x = min(N_ITERS, x * 1.07 + 2)
        ns.append(int(round(x)))
    return sorted(set(ns))


def save_gif(
    frames: Sequence[Image.Image], durs: Sequence[int], out: Path, comment: bytes
) -> None:
    """One shared palette (no flicker, no dithering), unchanged pixels as transparency."""
    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    sample = Image.new("RGB", (W, H * len(picks)))
    for j, k in enumerate(picks):
        sample.paste(frames[k], (0, H * j))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    palette = (pal.getpalette() or [])[: 255 * 3] + [0, 0, 0]
    idx = [np.asarray(f) for f in q]
    deltas = []
    for k, a in enumerate(idx):
        d = a.copy()
        if k:
            d[a == idx[k - 1]] = 255  # index 255 = transparent: keep the previous pixel
        im = Image.fromarray(d.astype(np.uint8), "P")
        im.putpalette(palette)
        deltas.append(im)
    out.parent.mkdir(parents=True, exist_ok=True)
    deltas[0].save(
        out,
        save_all=True,
        append_images=deltas[1:],
        duration=list(durs),
        loop=0,
        optimize=False,
        disposal=1,
        transparency=255,
        comment=comment,
    )
    with Image.open(out) as g:  # decode the file back and compare every frame
        assert g.n_frames == len(q), (g.n_frames, len(q))
        for k in range(g.n_frames):
            g.seek(k)
            want = np.asarray(q[k].convert("RGB"))
            assert np.array_equal(np.asarray(g.convert("RGB")), want), (
                f"frame {k} differs"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", nargs="?", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    D = compute(args.model)
    best, fav, cross = D.best + 1, D.fav + 1, D.cross
    p_fav, p_best = round(D.prior[D.fav] * 100), round(D.prior[D.best] * 100)
    final = D.share(N_ITERS)
    who = "Red" if D.mover == 1 else "Yellow"
    print(
        f"{D.params} parameters; prior col {fav} {p_fav}%, col {best} {p_best}%; "
        f"col {best} leads from iteration {cross}; visits at {N_ITERS}: "
        f"{D.counts[N_ITERS].tolist()}; {D.evidence['unique_leaves']} unique leaves"
    )

    intro = (
        f"{who} to move. The network's instinct: column {fav}.",
        f"Its policy head puts {p_fav}% on column {fav} and {p_best}% on column {best} "
        "(hollow bars).",
    )
    searching = (
        "Now search: one walk per iteration.",
        "Each walk descends the tree by prior and backed-up value, sends one leaf position "
        "to the network and backs its value up to the root.",
    )
    turned = (
        f"Column {best} takes the lead at iteration {cross} and keeps it.",
        f"The network gave it {p_best}%. From here to iteration {N_ITERS} it has more "
        "visits than any other move.",
    )
    solved = (
        f"The solver agrees: column {best} is the only win.",
        f"Perfect play scores column {fav}, the network's {p_fav}% favourite, as a loss. "
        f"After {N_ITERS} iterations search puts {round(final[D.best] * 100)}% of its visits "
        f"on column {best}.",
    )

    frames: list[Image.Image] = []
    durs: list[int] = []
    zero = np.zeros(7)
    frames.append(
        frame(D, n=0, hot=zero, caption=intro)
    )  # the prior, from the first frame
    durs.append(2400)
    hot = np.zeros(7)
    prev = 1
    for n in schedule(cross):
        hot *= 0.55
        for it in range(prev + 1, n + 1):
            hot[D.visited[it]] = 1.0
        prev = n
        frames.append(
            frame(D, n=n, hot=hot.copy(), caption=turned if n >= cross else searching)
        )
        durs.append(1200 if n == cross else 70)  # GIF delays are whole centiseconds
    for i in range(16):  # play the solver's move, then reveal every column's verdict
        drop = i / 7 if i <= 7 else 1 + (i - 7) / 4
        reveal = max(0.0, (i - 8) / 6)
        frames.append(
            frame(
                D,
                n=N_ITERS,
                hot=zero,
                caption=solved,
                reveal=reveal,
                drop=min(drop, 2.0),
            )
        )
        durs.append(50)
    durs[-1] = 3600

    comment = json.dumps(D.evidence, sort_keys=True, separators=(",", ":")).encode()
    save_gif(frames, durs, args.out, comment)
    size, total = args.out.stat().st_size, sum(durs)
    assert 8000 <= total <= 15000, f"loop of {total} ms"
    assert size < 3_000_000, f"GIF of {size} bytes"
    print(
        f"wrote {args.out}: {len(frames)} frames, {total / 1000:.2f} s/loop, {size / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
