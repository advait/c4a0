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

# The browser player's palette (web/style.css). Violet is c4a0 and its search, mint the other
# side; they differ in lightness as well as hue. `*_glow` and `*_deep` shade the glossy stones.
THEMES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#f6f5f1",
        "card": "#ffffff",
        "ink": "#121211",
        "ink2": "#5b5a55",
        "muted": "#96958c",
        "line": "#e4e2db",
        "grid": "#e9e7e0",
        "area": "#ebe9e2",
        "hole": "#ebe9e2",
        "violet": "#6c5ce7",
        "violet_deep": "#4a3bc2",
        "violet_glow": "#c3baff",
        "mint": "#14b896",
        "mint_deep": "#0a8a70",
        "mint_glow": "#8ff0d6",
    },
    "dark": {
        "surface": "#141413",
        "card": "#1d1d1b",
        "ink": "#f2f1ed",
        "ink2": "#b9b8ae",
        "muted": "#7d7c74",
        "line": "#2f2f2c",
        "grid": "#262624",
        "area": "#222220",
        "hole": "#121211",
        "violet": "#8c7dff",
        "violet_deep": "#5f4fe0",
        "violet_glow": "#d4ceff",
        "mint": "#2fd3ae",
        "mint_deep": "#15997d",
        "mint_glow": "#a5f5e0",
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
