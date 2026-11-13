#!/usr/bin/env python3
"""Build manuscript Figure 6, showing conditioning odor-identity RSA.

Run from any directory:
    python /path/to/project-spcd/fig/make_fig6_conditioning_rsa.py

Outputs: fig/fig6_conditioning_rsa.pdf, fig/fig6_conditioning_rsa.png.
The PDF is 11.4 cm wide and the PNG is rendered from it at 600 dpi. Validate
the RSA data against the synced group summaries before plotting. Preserve
panel dimensions on the working canvas, then scale the complete figure
uniformly to its final width; never stretch either axis. Keep labels at
least 6 pt at final size. Analysis tables and manuscript sources are not
modified. Requires Matplotlib, NumPy, pandas, SciPy, nibabel, and PyMuPDF.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_font_cache"))

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import fitz
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from scipy import stats

from figure_style import (
    DIRECT_COLOR, INFERRED_COLOR, LABEL_SIZE, NOTE_SIZE,
    PANEL_SIZE, REFERENCE_COLOR, SUMMARY_COLOR, TITLE_SIZE,
    configure_matplotlib, style_axis,
)
from scripts_MRI.RSA_plot import make_pnas_odor_identity_convergence_figure as manuscript
from scripts_behavior import make_design_figure as design


FINAL_WIDTH_CM = 11.4
WORKING_WIDTH_IN = 9.0
PANEL_REFERENCE_WIDTH_IN = 13.4
MIN_FINAL_FONT_PT = 6.0
PNG_DPI = 600
RSA_SELECTION = "sensitivity_runs1-3"
RSA_WINDOW = "cueEXT5s"


def load_rsa_panel(project: Path):
    root = project / "rsa_roi" / "conditioning_odor_identity" / RSA_WINDOW / RSA_SELECTION
    metric = manuscript.RSA_METRIC
    rows, summaries = [], {}
    for roi in manuscript.ROI_ORDER:
        folder = root / manuscript.ROI_FILE_KEYS[roi]
        data = pd.read_csv(folder / "subject_results.csv")
        saved = pd.read_csv(folder / "group_summary.csv")
        data = data.loc[data["pair_scope"].eq("cross_run")].copy()
        saved = saved.loc[saved["pair_scope"].eq("cross_run")]
        if not data["analysis"].eq(RSA_SELECTION).all():
            raise ValueError(f"Unexpected RSA run selection: {folder}")
        if (data.empty or data["sub"].duplicated().any() or set(data["roi"]) != {roi}
                or not data["window"].eq(RSA_WINDOW).all()):
            raise ValueError(f"Invalid participant/ROI/window rows: {folder}")
        values = data[metric].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Insufficient or nonfinite participant scores: {folder}")
        test = stats.ttest_1samp(values, 0.0, alternative="greater")
        summary = saved.loc[saved["metric"].eq(metric) & saved["roi"].eq(roi)]
        if len(summary) != 1:
            raise ValueError(f"Expected one matching group-summary row: {folder}")
        reference = summary.iloc[0]
        observed = [len(values), values.mean(), test.statistic, test.pvalue]
        expected = [reference["n"], reference["mean"], reference["t"], reference["p_greater_than_zero"]]
        if not np.allclose(observed, expected, atol=1e-10, rtol=1e-8):
            raise ValueError(f"Participant scores do not reproduce saved group tests: {folder}")
        summaries[roi] = {"n": len(values), "mean": values.mean(), "t": test.statistic, "p": test.pvalue}
        rows.append(data)
    frame = pd.concat(rows, ignore_index=True)
    wide = frame.pivot(index="sub", columns="roi", values=metric).reindex(columns=manuscript.ROI_ORDER).sort_index()
    if wide.isna().any().any():
        raise ValueError(f"ROI samples do not match: {root}")
    return frame, wide, summaries


def draw_conditioning_schematic(axis, project):
    """Illustrate the analyzed food-odor trials, with delivery modeled separately."""
    axis.set_axis_off()
    axis.set_xlim(0, 4.7)
    axis.set_ylim(-.40, 4.25)
    axis.set_aspect("equal")
    cue_x, prediction_x, delivery_x = .50, 2.0, 4.0
    branch_slope = np.tan(np.deg2rad(30))
    prediction_offset = (prediction_x - cue_x) * branch_slope
    for x, y, heading in [(cue_x, 3.30, "Conditioned\ncue"),
                          (prediction_x, 4.08, "Expected\nodors"),
                          (delivery_x, 3.30, "Delivered\nodors")]:
        design.text(axis, x, y, heading, size=12.5, weight="bold")
    for y, code, odor, asset, color in [
            (2.35, "B1", "Odor 1", "sweet_odor.png", INFERRED_COLOR),
            (.75, "B2", "Odor 2", "savory_odor.png", DIRECT_COLOR)]:
        prediction_y = y + prediction_offset
        cue_path = project / design.CUES[code].relative_to(design.ROOT)
        design.image(axis, cue_path, cue_x, y + .04, .66)
        design.text(axis, cue_x, y - .45, code, size=13, weight="bold")
        # Shallower expectation branches keep the schematic compact; contents stay upright.
        axis.add_patch(FancyArrowPatch(
            (cue_x + .42, y + .42 * branch_slope),
            (prediction_x - .59, prediction_y - .59 * branch_slope),
            arrowstyle="-|>",
            mutation_scale=13, linewidth=1.25, color=REFERENCE_COLOR,
            linestyle=(0, (3, 3)), zorder=2,
        ))
        design.arrow(axis, cue_x + .45, y, delivery_x - .62, y)
        for x, odor_y, expected in [(prediction_x, prediction_y, True),
                                    (delivery_x, y, False)]:
            axis.add_patch(FancyBboxPatch(
                (x - .53, odor_y - .51), 1.06, 1.02,
                boxstyle="round,pad=0,rounding_size=0.08",
                edgecolor=color, facecolor="white", linewidth=1.15,
                linestyle=(0, (3, 3)) if expected else "solid", zorder=1,
            ))
            design.image(axis, project / "fig" / "design_assets" / asset, x, odor_y + .09, .66)
            design.text(axis, x, odor_y - .36, odor, size=12, color=color)
    # Group cue and prediction responses without extending to odor delivery.
    axis.plot([.12, .12, 2.55, 2.55], [.22, .08, .08, .22],
              color="black", linewidth=.9)
    annotation_y = -.14
    annotation_style = {"size": 11, "color": "black"}
    design.text(axis, 1.335, annotation_y,
                "Cue-evoked patterns",
                **annotation_style)
    # Keep the delivery note local to the delivery column.
    design.text(axis, delivery_x, annotation_y, "Delivery modeled\nseparately.",
                **annotation_style)


def draw_rsa_panel(axis, wide, summaries, offsets):
    baseline = 0.0
    values = wide.to_numpy()
    intervals = {roi: manuscript._mean_ci(wide[roi].to_numpy()) for roi in manuscript.ROI_ORDER}
    low = min(baseline, float(values.min()), *(ci[1] for ci in intervals.values()))
    high = max(baseline, float(values.max()), *(ci[2] for ci in intervals.values()))
    span = high - low
    if span <= 0:
        raise ValueError("Cannot plot a panel without score variation.")
    axis.set_ylim(low - .12 * span, high + .25 * span)
    jitter = np.array([offsets[sub] for sub in wide.index])
    for x, roi in enumerate(manuscript.ROI_ORDER):
        axis.scatter(x + jitter, wide[roi], s=28, color=manuscript.INDIVIDUAL_COLOR,
                     alpha=.4, edgecolor="white", linewidth=.3, zorder=2)
        mean, ci_low, ci_high = intervals[roi]
        axis.errorbar(x, mean, yerr=[[mean - ci_low], [ci_high - mean]], fmt="D",
                      markersize=6.5, markerfacecolor="white", markeredgecolor=SUMMARY_COLOR,
                      markeredgewidth=1.6, color=SUMMARY_COLOR, linewidth=2.2,
                      capsize=5, capthick=2.2, zorder=4)
        annotation = manuscript._significance_label(float(summaries[roi]["p"]))
        axis.text(x, .985, annotation, ha="center", va="top",
                  transform=axis.get_xaxis_transform(),
                  fontsize=20 if annotation == "*" else NOTE_SIZE, color=SUMMARY_COLOR)
    axis.axhline(baseline, color=REFERENCE_COLOR, linewidth=1.2, linestyle=(0, (3, 3)))
    axis.set_xlim(-.48, 1.48)
    axis.set_xticks([0, 1], [manuscript.ROI_SHORT[roi].replace(" OFC", "\nOFC") for roi in manuscript.ROI_ORDER])
    axis.set_ylabel("Spearman ρ", fontsize=LABEL_SIZE, labelpad=6)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    style_axis(axis)
    axis.tick_params(axis="x", labelsize=12, pad=9)
    for label, roi in zip(axis.get_xticklabels(), manuscript.ROI_ORDER):
        label.set_color(manuscript.ROI_COLORS[roi])
        label.set_fontweight("bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output_prefix", type=Path,
                        help="Default: <project_root>/fig/fig6_conditioning_rsa")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    prefix = (args.output_prefix or project / "fig" / "fig6_conditioning_rsa").expanduser().resolve()

    rsa, wide, summaries = load_rsa_panel(project)
    if not rsa["runs"].isin(["1,2,3", "1,2"]).all():
        raise ValueError("Unexpected available-run pattern in the conditioning inputs.")
    n = len(wide)
    offsets = manuscript._participant_offsets(wide.index.tolist(), width=.12)
    configure_matplotlib()

    # Convert only normalized horizontal positions; panel sizes stay unchanged
    # in inches relative to the reference working canvas.
    horizontal = PANEL_REFERENCE_WIDTH_IN / WORKING_WIDTH_IN
    fig = plt.figure(figsize=(WORKING_WIDTH_IN, 5.0))
    schematic = fig.add_axes([.015 * horizontal, .18, .345 * horizontal, .65])
    draw_conditioning_schematic(schematic, project)
    axis = fig.add_axes([.418 * horizontal, .28, .239 * horizontal, .55])
    draw_rsa_panel(axis, wide, summaries, offsets)
    for x, letter, title in [(.02, "A", "Odor expectation\nduring conditioning"),
                             (.388, "B", "Odor-identity RSA")]:
        fig.text(x * horizontal, .965, letter, fontsize=PANEL_SIZE,
                 fontweight="bold", va="top")
        fig.text((x + .03) * horizontal, .965, title,
                 fontsize=TITLE_SIZE, fontweight="bold",
                 va="top", linespacing=1.2)

    scale = FINAL_WIDTH_CM / (WORKING_WIDTH_IN * 2.54)
    # Drawing creates the tick labels before enforcing their final font size.
    fig.canvas.draw()
    for label in fig.findobj(match=Text):
        label.set_fontsize(max(label.get_fontsize(), MIN_FINAL_FONT_PT / scale))
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    lower = max(0.0, bounds.y0 - .08)
    export_bounds = Bbox.from_extents(0, lower, WORKING_WIDTH_IN, fig.get_figheight())

    metadata = {
        "Title": "Figure 6. Conditioning odor-identity RSA",
        "Subject": f"EXT5s all-available-run cross-run RSA; n={n}; width=11.4 cm",
        "Creator": "project-spcd/fig/make_fig6_conditioning_rsa.py",
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with BytesIO() as buffer:
        fig.savefig(buffer, format="pdf", dpi=PNG_DPI * scale,
                    bbox_inches=export_bounds, pad_inches=0, metadata=metadata)
        with fitz.open(stream=buffer.getvalue(), filetype="pdf") as source:
            width_pt = FINAL_WIDTH_CM / 2.54 * 72
            height_pt = source[0].rect.height * scale
            with fitz.open() as output:
                page = output.new_page(width=width_pt, height=height_pt)
                page.show_pdf_page(page.rect, source, 0, keep_proportion=True)
                output.set_metadata(source.metadata)
                output.save(prefix.with_suffix(".pdf"), garbage=4, deflate=True)
                page.get_pixmap(dpi=PNG_DPI, alpha=False).save(prefix.with_suffix(".png"))
    plt.close(fig)

    for roi, summary in summaries.items():
        print("RSA", manuscript.ROI_SHORT[roi], summary)
    print(f"Final size: {FINAL_WIDTH_CM:.2f} x {height_pt / 72 * 2.54:.2f} cm; PNG: {PNG_DPI} dpi")
    print(f"Saved {prefix}.pdf and {prefix}.png")


if __name__ == "__main__":
    main()
