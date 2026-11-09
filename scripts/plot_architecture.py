"""Render the README network diagram for the final configuration (8x64 residual trunk, conv
policy and value heads).

Usage: uv run python scripts/plot_architecture.py

Writes images/architecture-light.svg and images/architecture-dark.svg. Parameter counts come
from instantiating the default network, so the figure cannot drift from the code.
"""

from dataclasses import dataclass
from pathlib import Path

from c4a0.nn import ConnectFourNet, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "card": "#f3f2ee",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#8a897f",
        "line": "#d6d5ce",
        "policy": "#2a78d6",
        "value": "#eb6834",
        "stone_a": "#0b0b0b",
        "stone_b": "#fcfcfb",
    },
    "dark": {
        "surface": "#1a1a19",
        "card": "#242423",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#8f8e86",
        "line": "#3a3a37",
        "policy": "#3987e5",
        "value": "#d95926",
        "stone_a": "#f0efec",
        "stone_b": "#1a1a19",
    },
}

# A mid-game position for the input sketch: (row from bottom, column, mover's stone?).
STONES = [
    (0, 3, True),
    (0, 2, False),
    (1, 3, False),
    (0, 4, True),
    (1, 2, True),
    (2, 3, True),
    (0, 1, False),
    (1, 4, False),
    (3, 3, False),
    (0, 5, True),
]


@dataclass(frozen=True)
class Layer:
    text: str
    note: str = ""


@dataclass(frozen=True)
class Block:
    title: str
    layers: tuple[Layer, ...]
    params: int
    accent: str  # theme key for the block's accent bar


def blocks(net: ConnectFourNet, n_blocks: int, filters: int) -> dict[str, Block]:
    count = lambda module: sum(p.numel() for p in module.parameters())  # noqa: E731
    return {
        "stem": Block(
            "Stem",
            (Layer(f"Conv 3×3 · {filters}"),),
            count(net.conv[0]),
            "muted",
        ),
        "trunk": Block(
            f"Residual block ×{n_blocks}",
            (
                Layer(f"Conv 3×3 · {filters}"),
                Layer(f"Conv 3×3 · {filters}"),
                Layer("BatchNorm · ReLU"),
                Layer("+ skip connection", "x + f(x)"),
            ),
            sum(count(b) for b in net.conv[1:]),
            "ink2",
        ),
        "policy": Block(
            "Policy head",
            (
                Layer("Conv 1×1 · 2"),
                Layer("BatchNorm · ReLU"),
                Layer("Linear 84 → 7"),
                Layer("log-softmax"),
            ),
            count(net.conv_policy),
            "policy",
        ),
        "value": Block(
            "Value head",
            (
                Layer("Conv 1×1 · 1"),
                Layer("BatchNorm · ReLU"),
                Layer("Linear 42 → 256 · ReLU"),
                Layer("Linear 256 → 2 · tanh"),
            ),
            count(net.conv_value),
            "value",
        ),
    }


def text(
    x: float,
    y: float,
    s: str,
    fill: str,
    size: float = 13,
    weight: int = 400,
    anchor: str = "start",
    family: str = FONT,
) -> str:
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" font-family="{family}">{s}</text>'
    )


def output(x: float, y: float, label: str, accent: str, t: dict[str, str]) -> str:
    """An output caption: the arrow carries the head's accent, the words stay in ink."""
    return (
        f'<text x="{x}" y="{y}" font-size="12.5" font-weight="600" font-family="{FONT}">'
        f'<tspan fill="{t[accent]}">→ </tspan><tspan fill="{t["ink2"]}">{label}</tspan></text>'
    )


def arrow(points: list[tuple[float, float]], color: str) -> str:
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points)
    )
    return (
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6" '
        f'marker-end="url(#head)"/>'
    )


def block_svg(
    b: Block, x: float, y: float, w: float, t: dict[str, str], stacked: int = 0
) -> tuple[str, float]:
    """Draws a block card; returns its SVG and height. `stacked` draws repeat cards behind."""
    row = 24
    h = 44 + row * len(b.layers) + 26
    out = []
    for i in range(stacked, 0, -1):
        out.append(
            f'<rect x="{x + 4 * i}" y="{y - 4 * i}" width="{w}" height="{h}" rx="10" '
            f'fill="{t["card"]}" stroke="{t["line"]}"/>'
        )
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{t["card"]}" '
        f'stroke="{t["line"]}"/>'
    )
    out.append(
        f'<rect x="{x}" y="{y + 12}" width="4" height="{h - 24}" rx="2" '
        f'fill="{t[b.accent]}"/>'
    )
    out.append(text(x + 16, y + 26, b.title, t["ink"], 14, 700))
    for i, layer in enumerate(b.layers):
        ly = y + 50 + row * i
        out.append(text(x + 16, ly, layer.text, t["ink2"], 12.5, family=MONO))
        if layer.note:
            out.append(
                text(
                    x + w - 14,
                    ly,
                    layer.note,
                    t["muted"],
                    11.5,
                    anchor="end",
                    family=MONO,
                )
            )
    out.append(text(x + 16, y + h - 12, f"{b.params:,} parameters", t["muted"], 11.5))
    return "\n".join(out), h


def board_svg(x: float, y: float, cell: float, t: dict[str, str]) -> str:
    out = [
        f'<rect x="{x}" y="{y}" width="{cell * 7}" height="{cell * 6}" rx="8" '
        f'fill="{t["card"]}" stroke="{t["line"]}"/>'
    ]
    for r in range(6):
        for c in range(7):
            cx, cy = x + cell * (c + 0.5), y + cell * (5 - r + 0.5)
            out.append(
                f'<circle cx="{cx}" cy="{cy}" r="{cell * 0.36}" fill="{t["surface"]}" '
                f'stroke="{t["line"]}"/>'
            )
    for r, c, mover in STONES:
        cx, cy = x + cell * (c + 0.5), y + cell * (5 - r + 0.5)
        out.append(
            f'<circle cx="{cx}" cy="{cy}" r="{cell * 0.36}" '
            f'fill="{t["stone_a"] if mover else t["stone_b"]}" stroke="{t["ink2"]}" '
            f'stroke-width="1.4"/>'
        )
    return "\n".join(out)


def render(theme_name: str, net: ConnectFourNet, config: ModelConfig) -> Path:
    t = THEMES[theme_name]
    bs = blocks(net, config.n_residual_blocks, config.conv_filter_size)
    total = sum(p.numel() for p in net.parameters())
    W, H = 980, 540
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-labelledby="title">',
        f'<title id="title">c4a0 network: a 2-plane 6x7 board feeds a 3x3 convolution stem and '
        f"{config.n_residual_blocks} residual blocks of {config.conv_filter_size} filters, which "
        f"split into a policy head (7 move probabilities) and a value head (two values, with "
        f"and without a ply penalty). {total:,} parameters.</title>",
        f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{t["muted"]}"/></marker></defs>',
        f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>',
        text(24, 36, "The network", t["ink"], 18, 700),
        text(
            24,
            58,
            f"{total:,} parameters · one forward pass per MCTS leaf · "
            "same weights for both players (the board is always seen from the mover's side)",
            t["ink2"],
            12.5,
        ),
    ]
    # Input.
    cell = 22
    bx, by = 24, 150
    parts.append(text(bx, by - 30, "Input · 2×6×7", t["ink"], 14, 700))
    parts.append(
        text(
            bx, by - 12, "plane 0: stones to move  ·  plane 1: opponent", t["muted"], 11
        )
    )
    parts.append(board_svg(bx, by, cell, t))
    # Stem and trunk.
    stem, stem_h = block_svg(bs["stem"], 214, 170, 170, t)
    trunk, trunk_h = block_svg(bs["trunk"], 412, 126, 220, t, stacked=3)
    parts += [stem, trunk]
    mid = 126 + trunk_h / 2
    parts.append(
        arrow([(bx + cell * 7 + 6, by + cell * 3), (210, by + cell * 3)], t["muted"])
    )
    parts.append(arrow([(384, 170 + stem_h / 2), (408, 170 + stem_h / 2)], t["muted"]))
    # Heads.
    policy, policy_h = block_svg(bs["policy"], 690, 86, 270, t)
    value, value_h = block_svg(bs["value"], 690, 106 + policy_h + 34, 270, t)
    parts += [policy, value]
    py, vy = 86 + policy_h / 2, 106 + policy_h + 34 + value_h / 2
    parts.append(arrow([(645, mid), (664, mid), (664, py), (686, py)], t["muted"]))
    parts.append(arrow([(645, mid), (664, mid), (664, vy), (686, vy)], t["muted"]))
    # Outputs.
    parts.append(
        output(690, 86 + policy_h + 22, "prior: where search looks first", "policy", t)
    )
    oy = 106 + policy_h + 34 + value_h + 22
    parts.append(output(690, oy, "q_penalty: faster wins worth more", "value", t))
    parts.append(output(690, oy + 19, "q_no_penalty: outcome alone", "value", t))
    parts.append("</svg>")
    out = ROOT / f"images/architecture-{theme_name}.svg"
    out.write_text("\n".join(parts) + "\n")
    return out


if __name__ == "__main__":
    config = ModelConfig(
        n_residual_blocks=8, conv_filter_size=64, lr_schedule={0: 2e-3}, l2_reg=4e-4
    )
    net = ConnectFourNet(config)
    for theme in THEMES:
        print(render(theme, net, config))
