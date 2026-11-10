"""Render the README training-progress figure from docs/data/training-progress.json.

Usage: uv run python scripts/plot_progress.py

Writes images/training-progress-light.svg and images/training-progress-dark.svg: the best
self-play run's dev-suite accuracy per generation (left axis), over the self-play games it has
played so far (right axis, a recessive context area rather than a second measure to compare).
Colors and fonts come from figure_theme, shared with the hero GIF and the network diagram.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from figure_theme import THEMES, font_file
from matplotlib.font_manager import fontManager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs/data/training-progress.json"

# (theme color, data key, legend label), in the fixed categorical order.
LINES = [
    (
        "violet",
        "mcts256_strict_leaf10",
        "Network + MCTS-256 + 10-ply exact leaf search",
    ),
    ("mint", "mcts256_strict", "Network + MCTS-256"),
    ("ink2", "raw_strict", "Network alone (policy argmax)"),
]


for bold in (
    False,
    True,
):  # the exact bundled files figure_theme names, not a system lookup
    fontManager.addfont(str(font_file(bold=bold)))


def render(theme_name: str) -> Path:
    t = THEMES[theme_name]
    data = json.loads(DATA.read_text())
    rows = data["rows"]
    gens = np.array([r["gen"] for r in rows])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.edgecolor": t["grid"],
            "axes.labelcolor": t["ink2"],
            "xtick.color": t["ink2"],
            "ytick.color": t["ink2"],
            "svg.fonttype": "path",
        }
    )
    # 980 x 540 units, the network diagram's canvas, so both scale identically in the README.
    fig, ax = plt.subplots(figsize=(980 / 72, 540 / 72), facecolor=t["surface"])
    fig.subplots_adjust(left=0.07, right=0.86, top=0.8, bottom=0.11)

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

    for color, key, label in LINES:
        points = [(r["gen"], r[key] * 100) for r in rows if r.get(key) is not None]
        x, y = np.array(points).T
        ax.plot(x, y, color=t[color], linewidth=2, label=label, solid_capstyle="round")
        ax.annotate(
            f"{y[-1]:.1f}%",
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=t["ink"],
            fontsize=12,
            annotation_clip=False,
        )

    fig.text(
        24 / 980,
        1 - 36 / 540,
        "Learning Connect Four from nothing but self-play",
        color=t["ink"],
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        24 / 980,
        1 - 58 / 540,
        "Share of held-out positions where the move is solver-optimal (fastest win, "
        f"slowest loss). {rows[-1]['hours']:.1f} hours on one shared RTX 3090.",
        color=t["ink2"],
        fontsize=12.5,
    )
    area = Patch(
        facecolor=t["area"],
        edgecolor=t["muted"],
        linewidth=0.8,
        label="Self-play games played (right axis)",
    )
    handles = [*ax.get_legend_handles_labels()[0], area]
    legend = ax.legend(
        handles=handles, loc="lower right", frameon=False, labelcolor=t["ink2"]
    )
    legend.set_zorder(5)

    out = ROOT / f"images/training-progress-{theme_name}.svg"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    for theme in THEMES:
        print(render(theme))
