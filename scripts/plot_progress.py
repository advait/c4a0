"""Render the README training-progress figure from docs/data/training-progress.json.

Usage: uv run python scripts/plot_progress.py

Writes images/training-progress-light.svg and images/training-progress-dark.svg. Each panel
plots one metric on one axis; a line is the mean of an arm's two runs and the band spans them.
Colors are the dataviz reference categorical slots in fixed order, stepped separately for the
light and dark surfaces (validated with the dataviz palette validator).
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
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#8f8e86",
        "grid": "#2e2e2c",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}

PANELS = [
    {
        "key": "search",
        "metric": "strict64",
        "title": "Fixing the self-play loop",
        "subtitle": "MCTS-64 picks the solver's fastest-winning move (dev suite)",
        "ylim": (50, 100),
        "target": None,
        "legend": {"loc": "upper left", "ncol": 2},
    },
    {
        "key": "target",
        "metric": "keeps256",
        "title": "Approaching the target",
        "subtitle": "MCTS-256 move keeps the game result: win stays win, draw stays draw (dev suite)",
        "ylim": (85, 100),
        "target": 98.0,
        "legend": {"loc": "lower right", "ncol": 1},
    },
]


def arm_curve(runs: dict, names: list[str], metric: str) -> tuple[np.ndarray, ...]:
    by_gen: dict[int, list[float]] = {}
    for name in names:
        for row in runs.get(name, []):
            if row.get(metric) is not None:
                by_gen.setdefault(row["gen"], []).append(row[metric] * 100)
    gens = np.array(sorted(g for g, v in by_gen.items() if len(v) == len(names)))
    vals = [by_gen[g] for g in gens]
    return (
        gens,
        np.array([np.mean(v) for v in vals]),
        np.array([min(v) for v in vals]),
        np.array([max(v) for v in vals]),
    )


def render(theme_name: str) -> Path:
    t = THEMES[theme_name]
    data = json.loads(DATA.read_text())
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
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.4), facecolor=t["surface"])
    fig.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.07, hspace=0.62)
    for ax, panel in zip(axes, PANELS, strict=True):
        ax.set_facecolor(t["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])
        ax.grid(axis="y", color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.set_ylim(*panel["ylim"])
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )
        ax.set_xlabel("Generation")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ends = []
        for i, arm in enumerate(data["series"][panel["key"]]):
            gens, mean, lo, hi = arm_curve(data["runs"], arm["runs"], panel["metric"])
            if len(gens) == 0:
                continue
            color = t["series"][i]
            ax.fill_between(gens, lo, hi, color=color, alpha=0.16, linewidth=0)
            ax.plot(
                gens,
                mean,
                color=color,
                linewidth=2,
                solid_capstyle="round",
                label=arm["label"],
            )
            ax.plot(
                gens[-1],
                mean[-1],
                "o",
                color=color,
                markersize=8,
                markeredgecolor=t["surface"],
                markeredgewidth=2,
            )
            ends.append([mean[-1], f"{mean[-1]:.1f}%  {arm['short']}", gens[-1]])
        # Direct labels at line ends, nudged apart so they never overlap.
        ends.sort(key=lambda e: e[0])
        min_gap = (panel["ylim"][1] - panel["ylim"][0]) * 0.055
        for j in range(1, len(ends)):
            ends[j][0] = max(ends[j][0], ends[j - 1][0] + min_gap)
        x_label = ax.get_xlim()[1]
        for y, text, _ in ends:
            ax.annotate(
                text,
                xy=(x_label, y),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color=t["ink"],
                fontsize=9,
                annotation_clip=False,
            )
        if panel["target"] is not None:
            ax.axhline(
                panel["target"], color=t["muted"], linewidth=1.2, linestyle=(0, (4, 3))
            )
            ax.annotate(
                f"target {panel['target']:.0f}%",
                xy=(0, panel["target"]),
                xytext=(4, 4),
                textcoords="offset points",
                color=t["ink2"],
                fontsize=9,
            )
        ax.text(
            0,
            1.17,
            panel["title"],
            transform=ax.transAxes,
            color=t["ink"],
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0,
            1.06,
            panel["subtitle"],
            transform=ax.transAxes,
            color=t["ink2"],
            fontsize=9.5,
        )
        legend = ax.legend(
            **panel["legend"], frameon=False, fontsize=8.5, labelcolor=t["ink2"]
        )
        for line in legend.get_lines():
            line.set_linewidth(2)
    fig.text(
        0.08,
        0.015,
        "Line = mean of two independent runs; band = their range.",
        color=t["muted"],
        fontsize=8.5,
    )
    out = ROOT / f"images/training-progress-{theme_name}.svg"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


if __name__ == "__main__":
    for name in THEMES:
        print(render(name))
