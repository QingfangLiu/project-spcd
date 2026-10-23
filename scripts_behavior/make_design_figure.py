#!/usr/bin/env python3
"""Draw a separate Python version of Figure 1, preserving the Adobe originals.

Run from any directory: python3 /path/to/scripts_behavior/make_design_figure.py
Requires matplotlib, numpy, and Pillow. Image paths and layout use project-relative
paths and inches, respectively. The two food illustrations were extracted from
the original PDF; the six cues use the higher-resolution repository originals.

The default 4 s conditioning delay is the effective interval from the end of
prediction to odor arrival at the nose: 2 s programmed delay plus approximately
2 s transit through the delivery tubing. It is not the software delay alone.
Trial displays are schematic; their widths do not encode duration.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_font_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_style import (
    DIRECT_COLOR, GRID_COLOR, INFERRED_COLOR, MUTED_TEXT_COLOR, NOTE_SIZE,
    PANEL_SIZE, REFERENCE_COLOR, SPINE_COLOR, TEXT_COLOR, TITLE_SIZE,
    configure_matplotlib,
)

WIDTH = 13.4
YMIN, YMAX = 1.15, 8.6
CUES = {
    "A1": ROOT / "cues/characters_gray/charmander.jpeg",
    "A2": ROOT / "cues/characters_gray/stitch.jpg",
    "C": ROOT / "cues/characters_gray/tigger.jpg",
    "B1": ROOT / "cues/places_gray/waterfall_small.jpg",
    "B2": ROOT / "cues/places_gray/museum.jpg",
    "D": ROOT / "cues/places_gray/city_road_night.jpg",
}
OUTCOME_COLORS = (INFERRED_COLOR, DIRECT_COLOR, SPINE_COLOR)


@lru_cache(maxsize=None)
def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGBA"))


def text(ax, x, y, label, *, size=NOTE_SIZE, color=TEXT_COLOR,
         weight="normal", ha="center", va="center", **kwargs):
    return ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, linespacing=1.2, **kwargs)


def image(ax, path: Path, x, y, width, height=None):
    """Fit each image in a fixed box without changing its aspect ratio."""
    pixels = load_image(path)
    height = width if height is None else height
    ratio = pixels.shape[1] / pixels.shape[0]
    actual_width, actual_height = min(width, height * ratio), min(height, width / ratio)
    ax.imshow(pixels, extent=(x - actual_width / 2, x + actual_width / 2,
                             y - actual_height / 2, y + actual_height / 2),
              interpolation="lanczos", zorder=3)


def arrow(ax, x1, y1, x2, y2, *, color=REFERENCE_COLOR):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                mutation_scale=13, linewidth=1.25,
                                color=color, zorder=2))


def cue(ax, code, x, y, *, width=.64, label=True):
    image(ax, CUES[code], x, y, width)
    if label:
        text(ax, x, y - width / 2 - .17, code, size=14, weight="bold")


def panel_header(ax, x, letter, title):
    text(ax, x, 8.22, letter, size=PANEL_SIZE, weight="bold", ha="left")
    text(ax, x + .38, 8.22, title, size=TITLE_SIZE, weight="bold", ha="left")


def association_panels(ax):
    # Rounded stage outlines echo the Adobe layout; titles sit above each box.
    for left, right in ((.28, 3.02), (3.34, 6.38), (6.76, 13.12)):
        ax.add_patch(FancyBboxPatch(
            (left, 4.27), right - left, 3.73,
            boxstyle="round,pad=0,rounding_size=0.18",
            edgecolor=SPINE_COLOR, facecolor="white", linewidth=1.15,
            zorder=0,
        ))
    panel_header(ax, .28, "A", "Preconditioning")
    panel_header(ax, 3.40, "B", "Conditioning")
    panel_header(ax, 6.76, "C", "Probe tests")
    text(ax, 1.57, 7.78, "Learn cue pairs", color=MUTED_TEXT_COLOR)
    text(ax, 4.86, 7.78, "Learn cue-outcome associations", size=12, color=MUTED_TEXT_COLOR)
    text(ax, 9.97, 7.78, "Choices before and after selective satiation", color=MUTED_TEXT_COLOR)

    for y, first, second, outcome, color in zip(
            (7.12, 6.10, 5.08), ("A1", "A2", "C"), ("B1", "B2", "D"),
            ("Sweet odor", "Savory odor", "Clean air"), OUTCOME_COLORS):
        cue(ax, first, .94, y)
        arrow(ax, 1.40, y, 1.97, y)
        cue(ax, second, 2.40, y)
        cue(ax, second, 3.93, y)
        arrow(ax, 4.38, y, 4.91, y)
        if second == "D":
            # Clean air is an actual outcome, rather than a missing image.
            text(ax, 5.55, y, "Air", size=19, color=REFERENCE_COLOR)
        else:
            asset = "sweet_odor.png" if second == "B1" else "savory_odor.png"
            image(ax, ROOT / "fig/design_assets" / asset, 5.55, y, .65)
        text(ax, 5.55, y - .49, outcome, size=13, color=color)

    text(ax, 8.26, 7.42, "Preconditioned cues", color=MUTED_TEXT_COLOR)
    text(ax, 11.39, 7.42, "Conditioned cues", color=MUTED_TEXT_COLOR)
    comparisons = (("A1", "C", "B1", "D"),
                   ("A2", "C", "B2", "D"),
                   ("A1", "A2", "B1", "B2"))
    for y, codes in zip((6.92, 5.92, 4.92), comparisons):
        for center, pair in ((8.26, codes[:2]), (11.39, codes[2:])):
            cue(ax, pair[0], center - .66, y, width=.58)
            text(ax, center, y, "vs", size=13, color=MUTED_TEXT_COLOR)
            cue(ax, pair[1], center + .66, y, width=.58)


def screen(ax, x, y, label, duration, *, content=(), fixation=False,
           prediction=False, choice=False, odor=False, feedback=False):
    """Draw one trial display; common geometry keeps all timelines aligned."""
    width, height = .80, .70
    cue_size = width * .425
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                boxstyle="round,pad=0.018,rounding_size=0.035",
                                linewidth=.85, edgecolor=SPINE_COLOR,
                                facecolor="#F2F3F4", zorder=1))
    if feedback:
        # Alternative accuracy messages; feedback does not show a cue image.
        text(ax, x, y + .19, "Correct", size=10)
        text(ax, x, y, "or", size=8, color=MUTED_TEXT_COLOR)
        text(ax, x, y - .19, "Incorrect", size=10)
    elif prediction:
        # The cue remains visible during prediction, as described in Methods.
        image(ax, CUES["B1"], x, y, cue_size)
        text(ax, x, y - .27, "clea  stra  pota", size=8)
    elif len(content) == 2:
        for dx, code in zip((-.20, .20), content):
            image(ax, CUES[code], x + dx, y, cue_size)
    elif content:
        image(ax, CUES[content[0]], x, y, cue_size)
    elif fixation:
        text(ax, x, y, "+", size=23,
             color=INFERRED_COLOR if odor else TEXT_COLOR)
    elif choice:
        for dx, number in ((-.16, "1"), (.16, "2")):
            text(ax, x + dx, y, number, size=17, weight="bold")
    text(ax, x, y - .57, label, size=13, va="top")
    text(ax, x, y - .83, duration, size=13, color=MUTED_TEXT_COLOR, va="top")


def trial_timelines(ax, conditioning_delay):
    ax.plot([.28, 13.12], [4.12, 4.12], color=GRID_COLOR, linewidth=1)
    sections = (
        (.28, "Preconditioning trial", (.70, 1.70, 2.70, 3.70), (
            dict(label="First cue", duration="3 s", content=("A1",)),
            dict(label="Gap", duration="0.3 s", fixation=True),
            dict(label="Choice", duration="3 s", content=("B1", "B2")),
            dict(label="Feedback", duration="2 s", feedback=True),
        )),
        (4.63, "Conditioning trial", (5.05, 6.05, 7.05, 8.05), (
            dict(label="Cue", duration="2 s", content=("B1",)),
            dict(label="Prediction", duration="Max 3 s", prediction=True),
            dict(label="Delay", duration=f"{conditioning_delay:g} s", fixation=True),
            dict(label="Odor", duration="2 s", fixation=True, odor=True),
        )),
        (9.02, "Probe trial", (9.43, 10.43, 11.43, 12.43), (
            dict(label="Cue 1", duration="3 s", content=("B1",)),
            dict(label="Gap", duration="4-10 s", fixation=True),
            dict(label="Cue 2", duration="3 s", content=("B2",)),
            dict(label="Choice", duration="Max 2 s", choice=True),
        )),
    )
    for start, title, xs, steps in sections:
        text(ax, start, 3.73, title, size=TITLE_SIZE, weight="bold", ha="left")
        for i, (x, step) in enumerate(zip(xs, steps)):
            screen(ax, x, 3.05, **step)
            if i + 1 < len(xs):
                arrow(ax, x + .43, 3.05, xs[i + 1] - .43, 3.05)

    text(ax, 2.20, 1.60, "Choose the associated second cue",
         size=13, color=MUTED_TEXT_COLOR)
    text(ax, 6.55, 1.60, "Predict the associated odor", size=13, color=MUTED_TEXT_COLOR)
    text(ax, 10.93, 1.60, "Choose the preferred cue", size=13, color=MUTED_TEXT_COLOR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioning-delay", type=float, default=4,
                        help="Effective delay to odor arrival in seconds (default 4: 2 programmed + 2 transit).")
    parser.add_argument("--output-stem", type=Path, default=ROOT / "fig/fig_design_python",
                        help="Output path without extension; existing Adobe paths are protected.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.conditioning_delay <= 0 or args.dpi <= 0:
        parser.error("Delay and DPI must be positive.")
    stem = args.output_stem.expanduser().resolve()
    if stem.suffix:
        parser.error("The output stem must have no file extension.")
    if stem == (ROOT / "fig/fig_design").resolve():
        parser.error("The original Adobe figure is protected. Choose a different output stem.")
    configure_matplotlib()
    plt.rcParams["svg.fonttype"] = "none"
    fig = plt.figure(figsize=(WIDTH, YMAX - YMIN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(YMIN, YMAX), aspect="equal")
    ax.axis("off")
    association_panels(ax)
    trial_timelines(ax, args.conditioning_delay)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png", "svg"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
