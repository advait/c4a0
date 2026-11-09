"""Render the README training-progress figure from docs/data/training-progress.json.

Usage: uv run python scripts/plot_progress.py

Writes images/training-progress-light.svg and images/training-progress-dark.svg: the best
self-play run's dev-suite accuracy per generation (left axis), over the self-play games it has
played so far (right axis, a recessive context area rather than a second measure to compare).
Line colors are the dataviz reference categorical slots in fixed order, stepped separately for
the light and dark surfaces.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs/data/training-progress.json"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#8a897f",
        "grid": "#e7e6e1",
        "area": "#ecebe6",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#8f8e86",
        "grid": "#2e2e2c",
        "area": "#262624",
        "series": ["#3987e5", "#d95926", "#199e70"],
    },
}

# (data key, legend label), in the fixed categorical order.
LINES = [
    ("mcts256_strict_leaf10", "Network + MCTS-256 + 10-ply exact leaf search"),
    ("mcts256_strict", "Network + MCTS-256"),
    ("raw_strict", "Network alone (policy argmax)"),
]


def render(theme_name: str) -> Path:
    t = THEMES[theme_name]
    data = json.loads(DATA.read_text())
    rows = data["rows"]
    gens = np.array([r["gen"] for r in rows])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": t["grid"],
            "axes.labelcolor": t["ink2"],
            "xtick.color": t["ink2"],
            "ytick.color": t["ink2"],
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=t["surface"])
    fig.subplots_adjust(left=0.08, right=0.84, top=0.8, bottom=0.12)

    # Context layer on the right axis: experience accumulated by self-play.
    games = ax.twinx()
    games.fill_between(
        gens, (gens * data["games_per_gen"]) / 1000, color=t["area"], linewidth=0
    )
    games.set_ylim(0, 100)
    games.set_ylabel("Self-play games played (thousands)", color=t["muted"])
    games.tick_params(length=0, colors=t["muted"], pad=44)
    games.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
    ax.set_zorder(games.get_zorder() + 1)
    ax.patch.set_visible(False)

    for axis in (ax, games):
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(t["grid"])
    games.set_facecolor(t["surface"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.tick_params(length=0)
    ax.set_ylim(80, 100)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.set_xlim(0, gens.max())
    ax.set_xlabel("Generation")

    for color, (key, label) in zip(t["series"], LINES, strict=True):
        points = [(r["gen"], r[key] * 100) for r in rows if r.get(key) is not None]
        x, y = np.array(points).T
        ax.plot(x, y, color=color, linewidth=2, label=label, solid_capstyle="round")
        ax.annotate(
            f"{y[-1]:.1f}%",
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=t["ink"],
            fontsize=9,
            annotation_clip=False,
        )

    fig.text(
        0.08,
        0.93,
        "Learning Connect Four from nothing but self-play",
        color=t["ink"],
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.875,
        "Share of held-out positions where the move is solver-optimal (fastest win, "
        f"slowest loss). {rows[-1]['hours']:.1f} hours on one shared RTX 3090.",
        color=t["ink2"],
        fontsize=9.5,
    )
    legend = ax.legend(
        loc="lower right", frameon=False, fontsize=9, labelcolor=t["ink2"]
    )
    legend.set_zorder(5)

    out = ROOT / f"images/training-progress-{theme_name}.svg"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    for theme in THEMES:
        print(render(theme))
