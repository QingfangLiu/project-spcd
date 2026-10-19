"""Shared manuscript style for the behavior and odor-identity MRI figures.

Based on the reward_identity_allocation_simulation figure's DejaVu Sans font,
blue/orange cue colors, and restrained axes. Text is scaled for these figures'
13.4-inch canvases so it remains legible at the same displayed journal width.
"""

import matplotlib.pyplot as plt


TEXT_COLOR = "#20252B"
MUTED_TEXT_COLOR = "#5D6871"
GRID_COLOR = "#E5E7E9"
SPINE_COLOR = "#9AA3AA"
INFERRED_COLOR = "#2878B5"
DIRECT_COLOR = "#D97721"
SUMMARY_COLOR = "#29343D"
REFERENCE_COLOR = "#7A858D"

LABEL_SIZE = 16
TICK_SIZE = 14
TITLE_SIZE = 17
PANEL_SIZE = 20
LEGEND_SIZE = 13
NOTE_SIZE = 13
INSET_LABEL_SIZE = 13
INSET_TICK_SIZE = 12


def configure_matplotlib() -> None:
    """Configure both generators without changing analytical conventions."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": LABEL_SIZE,
        "text.color": TEXT_COLOR,
        "axes.labelsize": LABEL_SIZE,
        "axes.labelcolor": TEXT_COLOR,
        "axes.labelpad": 9,
        "axes.titlesize": TITLE_SIZE,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 16,
        "axes.edgecolor": SPINE_COLOR,
        "axes.linewidth": 1.0,
        "axes.unicode_minus": False,
        "axes.grid": False,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "legend.handlelength": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def style_axis(axis, *, inset: bool = False) -> None:
    """Use open gray axes and faint horizontal guides, as in the model figure."""
    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axis.tick_params(
        axis="both", direction="out", length=4, width=0.9,
        labelsize=INSET_TICK_SIZE if inset else TICK_SIZE,
    )
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(SPINE_COLOR)
        axis.spines[name].set_linewidth(1.0)
    if inset:
        axis.yaxis.label.set_size(INSET_LABEL_SIZE)
        axis.yaxis.labelpad = 5


def add_panel_header(axis, letter: str, title: str, *, title_lines: int | None = None) -> None:
    """Align panel letters with single- or multi-line left-aligned titles."""
    axis.set_title(title, loc="left", pad=16)
    lines = title_lines if title_lines is not None else title.count("\n") + 1
    extra_line_height = TITLE_SIZE * 1.2 * (lines - 1)
    axis.annotate(
        letter, xy=(-0.12, 1), xycoords="axes fraction",
        xytext=(0, 16 + extra_line_height), textcoords="offset points",
        fontsize=PANEL_SIZE, fontweight="bold", ha="left", va="bottom",
        annotation_clip=False,
    )
