#!/usr/bin/env python3
"""Draw manuscript Figure 1 using the shared style and checked-in image assets.

Run from any directory: python3 /path/to/scripts_behavior/make_design_figure.py
Requires matplotlib, numpy, and Pillow. Image paths and layout use project-relative
paths and inches, respectively. The two food illustrations were extracted from
the retired Adobe figure and are checked in under fig/design_assets; the six
cues use the higher-resolution repository originals.

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
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_style import (
    DIRECT_COLOR, GRID_COLOR, INFERRED_COLOR, NOTE_SIZE,
    PANEL_SIZE, REFERENCE_COLOR, SPINE_COLOR, TEXT_COLOR, TITLE_SIZE,
    configure_matplotlib,
)

WIDTH = 13.4
YMIN, YMAX = 1.95, 11.22
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


def airflow(ax, x, y, *, width=.65):
    """Draw the clean-air outcome as three vector airflow strokes."""
    strokes = (
        ((.23, .69), (.41, .69), (.61, .69), (.65, .69),
         (.76, .69), (.76, .85), (.65, .85),
         (.61, .85), (.59, .82), (.59, .79)),
        ((.18, .59), (.39, .59), (.68, .59), (.79, .59)),
        ((.27, .49), (.42, .49), (.58, .49), (.62, .49),
         (.72, .49), (.72, .35), (.62, .35),
         (.58, .35), (.56, .38), (.56, .40)),
    )
    for stroke in strokes:
        vertices = (np.asarray(stroke) - (.485, .60)) * width / .61 + (x, y)
        codes = [MplPath.MOVETO] + [MplPath.CURVE4] * (len(stroke) - 1)
        ax.add_patch(PathPatch(
            MplPath(vertices, codes), fill=False, edgecolor=REFERENCE_COLOR,
            linewidth=1.6, capstyle="round", joinstyle="round", zorder=3,
        ))


def cue(ax, code, x, y, *, width=.64, label=True):
    image(ax, CUES[code], x, y, width)
    if label:
        text(ax, x, y - width / 2 - .17, code, size=14, weight="bold")


def panel_header(ax, x, letter, title, *, y=8.22):
    text(ax, x, y, letter, size=PANEL_SIZE, weight="bold", ha="left")
    text(ax, x + .38, y, title, size=TITLE_SIZE, weight="bold", ha="left")


def eating_meal(ax, x, y, *, scale=.9):
    """Draw a person lifting a bite from the example meal toward their mouth."""
    def point(dx, dy):
        return (x + scale * dx, y + scale * dy)

    head = (
        (-.65, -.04), (-.65, .06),
        (-.83, .16), (-.83, .39), (-.66, .44),
        (-.49, .50), (-.34, .38), (-.34, .27),
        (-.25, .16), (-.34, .13), (-.31, .09),
        (-.37, .07), (-.31, .04),
        (-.31, -.04), (-.41, -.08), (-.45, -.08),
        (-.45, -.15),
    )
    codes = (
        [MplPath.MOVETO, MplPath.LINETO] + [MplPath.CURVE4] * 6
        + [MplPath.LINETO] * 5 + [MplPath.CURVE4] * 3
        + [MplPath.LINETO]
    )
    ax.add_patch(PathPatch(
        MplPath([point(*p) for p in head], codes), fill=False,
        edgecolor=TEXT_COLOR, linewidth=1.45, capstyle="round",
        joinstyle="round", zorder=4,
    ))
    ax.add_patch(Circle(point(-.40, .23), .017 * scale,
                        facecolor=TEXT_COLOR, edgecolor="none", zorder=4))
    # Neck, shoulder, and torso, with the elbow bent toward the raised hand.
    for stroke in (
        ((-.65, -.04), (-.81, -.15), (-.86, -.34)),
        ((-.45, -.15), (-.27, -.24), (-.23, -.34)),
        ((-.63, -.15), (-.39, -.33), (-.05, .01), (-.13, .06)),
    ):
        ax.plot(*zip(*(point(*p) for p in stroke)), color=TEXT_COLOR,
                linewidth=1.45, solid_capstyle="round",
                solid_joinstyle="round", zorder=4)
    ax.add_patch(Ellipse(point(-.18, .09), .13 * scale, .075 * scale,
                         angle=-15, facecolor="#E8BB5C", edgecolor="#B78A35",
                         linewidth=.65, zorder=5))
    image(ax, ROOT / "fig/design_assets/savory_odor.png",
          *point(.47, -.10), .75 * scale)
    ax.add_patch(FancyArrowPatch(
        point(.29, .05), point(-.06, .19),
        connectionstyle="arc3,rad=.3", arrowstyle="-|>",
        mutation_scale=8, linewidth=1, color=REFERENCE_COLOR, zorder=4,
    ))


def session_timeline(ax):
    """Separate the two probe sessions around the meal, above shared details."""
    panel_header(ax, .28, "A", "Experimental sequence", y=10.84)
    stages = (
        (.28, 2.35, "Preconditioning", "Learn cue pairs"),
        (2.60, 4.67, "Conditioning", "Learn cue-outcome\nassociations"),
        (4.92, 7.34, "Pre-meal probe", "Choose the cue predicting\nthe preferred odor\n\nNo odors delivered"),
        (7.59, 10.19, "Selective satiation", None),
        (10.44, 13.12, "Post-meal probe", "Repeat the same\nprobe procedure"),
    )
    for index, (left, right, title, detail) in enumerate(stages):
        center = (left + right) / 2
        meal = detail is None
        ax.add_patch(FancyBboxPatch(
            (left, 8.95), right - left, 1.65,
            boxstyle="round,pad=0,rounding_size=0.12",
            edgecolor=SPINE_COLOR, facecolor="#F7F8F9",
            linewidth=1.15, zorder=0,
        ))
        text(ax, center, 10.30, title, size=14, weight="bold")
        if meal:
            eating_meal(ax, center, 9.75)
            text(ax, center, 9.31, "Eat matching meal until full", size=11)
            text(ax, center, 9.08, "Reduces pleasantness of odor 2", size=11)
        else:
            text(ax, center, 9.57, detail, size=12)
        if index + 1 < len(stages):
            arrow(ax, right + .025, 9.78, stages[index + 1][0] - .025, 9.78)
    ax.plot([.28, 13.12], [8.64, 8.64], color=GRID_COLOR, linewidth=1)


def association_panels(ax):
    # Match the outer margins of A/E, with equal gutters and shared headings.
    widths = (2.90, 3.55, 5.18)
    b_left, row_right = .28, 13.12
    gap = (row_right - b_left - sum(widths)) / 2
    c_left = b_left + widths[0] + gap
    d_left = c_left + widths[1] + gap
    b_content, c_content = b_left + .40, c_left + .40
    bottom, top = 5.00, 8.08
    rows = (7.56, 6.64, 5.72)
    for left, width in zip((b_left, c_left, d_left), widths):
        ax.add_patch(FancyBboxPatch(
            (left, bottom), width, top - bottom,
            boxstyle="round,pad=0,rounding_size=0.18",
            edgecolor=SPINE_COLOR, facecolor="white", linewidth=1.15,
            zorder=0,
        ))
    panel_header(ax, b_left, "B", "Preconditioning", y=top + .22)
    panel_header(ax, c_left, "C", "Conditioning", y=top + .22)
    panel_header(ax, d_left, "D", "Probe task (both sessions)", y=top + .22)

    for y, first, second, outcome, color in zip(
            rows, ("A1", "A2", "C"), ("B1", "B2", "D"),
            ("Food odor 1", "Food odor 2", "Clean air"), OUTCOME_COLORS):
        cue(ax, first, b_content + .57, y)
        arrow(ax, b_content + 1.02, y, b_content + 1.45, y)
        cue(ax, second, b_content + 1.90, y)
        cue(ax, second, c_content + .54, y)
        arrow(ax, c_content + .98, y, c_content + 1.30, y)
        if second == "D":
            airflow(ax, c_content + 1.80, y)
        else:
            asset = "sweet_odor.png" if second == "B1" else "savory_odor.png"
            image(ax, ROOT / "fig/design_assets" / asset, c_content + 1.80, y, .65)
        text(ax, c_content + 1.80, y - .49, outcome, size=12, color=color)

    # The bracket groups only the two food odors, excluding clean air.
    ax.plot([c_content + dx for dx in (2.37, 2.45, 2.45, 2.37)],
            [rows[0], rows[0], rows[1], rows[1]],
            color=REFERENCE_COLOR, linewidth=1.1)
    text(ax, c_content + 2.76, 7.10, "Initially\nsimilarly preferred", size=10.5,
         color="black", rotation=90)

    text(ax, d_left + .24, 6.56, "Preconditioned cues", color="black", rotation=90)
    text(ax, d_left + 2.91, 6.56, "Conditioned cues", color="black", rotation=90)
    comparisons = (("A1", "C", "B1", "D"),
                   ("A2", "C", "B2", "D"),
                   ("A1", "A2", "B1", "B2"))
    for y, codes in zip(rows, comparisons):
        for center, pair in ((d_left + 1.29, codes[:2]), (d_left + 3.96, codes[2:])):
            cue(ax, pair[0], center - .58, y, width=.58)
            text(ax, center, y, "vs", size=13, color="black")
            cue(ax, pair[1], center + .58, y, width=.58)

    # Counterbalancing concerns cue roles and odor assignments, not screen positions.
    for x, note in (
        (b_left + .27, "Character/place assignment\nto cue roles counterbalanced"),
        (c_left + .27, "Sweet/savory assignment\ncounterbalanced"),
    ):
        text(ax, x, 6.56, note, size=10.5, color="black", rotation=90)


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
        text(ax, x, y, "or", size=8, color="black")
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
    text(ax, x, y - .83, duration, size=13, color="black", va="top")


def trial_timelines(ax, conditioning_delay):
    ax.plot([.28, 13.12], [4.70, 4.70], color=GRID_COLOR, linewidth=1)
    panel_header(ax, .28, "E", "Trial structure", y=4.36)
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
        text(ax, start, 3.86, title, size=TITLE_SIZE, weight="bold", ha="left")
        for i, (x, step) in enumerate(zip(xs, steps)):
            screen(ax, x, 3.18, **step)
            if i + 1 < len(xs):
                arrow(ax, x + .43, 3.18, xs[i + 1] - .43, 3.18)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioning-delay", type=float, default=4,
                        help="Effective delay to odor arrival in seconds (default 4: 2 programmed + 2 transit).")
    parser.add_argument("--output-stem", type=Path, default=ROOT / "fig/fig1_design",
                        help="Output path without a file extension.")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.conditioning_delay <= 0 or args.dpi <= 0:
        parser.error("Delay and DPI must be positive.")
    stem = args.output_stem.expanduser().resolve()
    if stem.suffix:
        parser.error("The output stem must have no file extension.")
    configure_matplotlib()
    fig = plt.figure(figsize=(WIDTH, YMAX - YMIN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(YMIN, YMAX), aspect="equal")
    ax.axis("off")
    session_timeline(ax)
    association_panels(ax)
    trial_timelines(ax, args.conditioning_delay)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=args.dpi, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
