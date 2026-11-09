"""The one look shared by every README figure: palette, fonts, and text drawn as glyph outlines.

Each figure renders a light and a dark variant from THEMES. Text uses only the DejaVu fonts
bundled with matplotlib, drawn as paths in the SVGs, so the hero GIF, the network diagram and
the training graph show the same glyphs on every machine and in every browser.
"""

from pathlib import Path

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, text_to_path

# `blue` marks what search found (and the final agent), `orange` the network's instinct, in
# the dataviz reference's categorical order, stepped separately for each surface.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "card": "#f3f2ee",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#8a897f",
        "line": "#d6d5ce",
        "grid": "#e7e6e1",
        "area": "#ecebe6",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "green": "#1baf7a",
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
        "grid": "#2e2e2c",
        "area": "#262624",
        "blue": "#3987e5",
        "orange": "#d95926",
        "green": "#199e70",
        "stone_a": "#f0efec",
        "stone_b": "#1a1a19",
    },
}

SVG_OPS = {
    MplPath.MOVETO: "M",
    MplPath.LINETO: "L",
    MplPath.CURVE3: "Q",
    MplPath.CURVE4: "C",
}

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font_file(*, bold: bool = False, mono: bool = False) -> Path:
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "")
    return FONT_DIR / f"{name}.ttf"


def svg_text(
    x: float,
    y: float,
    s: str,
    fill: str,
    size: float = 13,
    *,
    bold: bool = False,
    mono: bool = False,
    anchor: str = "start",
) -> tuple[str, float]:
    """`s` as an SVG path with its baseline at y. Returns the path and its advance width."""
    prop = FontProperties(fname=font_file(bold=bold, mono=mono), size=size)
    width = text_to_path.get_text_width_height_descent(s, prop, ismath=False)[0]
    x0 = x - {"start": 0.0, "middle": width / 2, "end": width}[anchor]
    d: list[str] = []
    for points, code in TextPath((0, 0), s, prop=prop).iter_segments():
        if code == MplPath.CLOSEPOLY:
            d.append("Z")
            continue
        xy = " ".join(
            f"{x0 + px:.2f},{y - py:.2f}"
            for px, py in zip(points[::2], points[1::2], strict=True)
        )
        d.append(SVG_OPS[code] + xy)
    return f'<path fill="{fill}" d="{"".join(d)}"/>', width
