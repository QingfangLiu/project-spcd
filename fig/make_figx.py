#!/usr/bin/env python3
"""Build the conditioning identity figure and caption from synced ROI results.

Run from any directory:
    python /path/to/project-spcd/fig/make_figx.py

Outputs: fig/figx.pdf, fig/figx.png, fig/figx_caption.txt.
Display choices: EXT5s all-available-run cross-run Spearman RSA and EARLY
passive-to-conditioning balanced accuracy. Window details belong in the
caption. This does not modify analysis tables or the manuscript.

Panel A reuses Figure 1's actual cue/outcome assets and image/arrow helpers.
Expected odor images are conceptual, not visual stimuli shown in the task.
The delivery note describes the pooled nuisance regressor, not proof that
all delivery-related information was removed from the cue estimates.
Panels B/C reuse the Figure 3/4 generator's ROI order/colors, participant jitter,
and mean/CI helper, together with the shared manuscript style. Validate plotted
means and one-sided t tests against every saved group summary before drawing.
"""

from __future__ import annotations

import argparse
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
from matplotlib.transforms import Bbox
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from figure_style import (
    DIRECT_COLOR, INFERRED_COLOR, LABEL_SIZE, NOTE_SIZE,
    PANEL_SIZE, REFERENCE_COLOR, SUMMARY_COLOR, TITLE_SIZE,
    configure_matplotlib, style_axis,
)
from scripts_MRI.RSA_plot import make_pnas_odor_identity_convergence_figure as manuscript
from scripts_behavior import make_design_figure as design

RSA_SELECTION = "sensitivity_runs1-3"
RSA_WINDOW = "cueEXT5s"
DECODING_WINDOW = "cueEARLY"


def load_panel(project: Path, analysis: str):
    is_rsa = analysis == "rsa"
    root = (project / "rsa_roi" / "conditioning_odor_identity" / RSA_WINDOW / RSA_SELECTION
            if is_rsa else project / "roi_decoding" / "conditioning_crosstask_odor" / DECODING_WINDOW)
    metric = manuscript.RSA_METRIC if is_rsa else manuscript.DECODING_METRIC
    baseline = 0.0 if is_rsa else 0.5
    rows, summaries = [], {}
    for roi in manuscript.ROI_ORDER:
        folder = root / manuscript.ROI_FILE_KEYS[roi]
        data = pd.read_csv(folder / "subject_results.csv")
        saved = pd.read_csv(folder / "group_summary.csv")
        if is_rsa:
            data = data.loc[data["pair_scope"].eq("cross_run")].copy()
            saved = saved.loc[saved["pair_scope"].eq("cross_run")]
            expected_window = RSA_WINDOW
            p_column = "p_greater_than_zero"
            if not data["analysis"].eq(RSA_SELECTION).all():
                raise ValueError(f"Unexpected RSA run selection: {folder}")
        else:
            expected_window = DECODING_WINDOW
            p_column = "p_greater_than_null"
            if not data["phase"].eq("conditioning").all() or not data["cue_subset"].eq("conditioned_odor_only").all():
                raise ValueError(f"Unexpected test phase/cue class: {folder}")
        if (data.empty or data["sub"].duplicated().any() or set(data["roi"]) != {roi}
                or not data["window"].eq(expected_window).all()):
            raise ValueError(f"Invalid participant/ROI/window rows: {folder}")
        values = data[metric].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Insufficient or nonfinite participant scores: {folder}")
        test = stats.ttest_1samp(values, baseline, alternative="greater")
        summary = saved.loc[saved["metric"].eq(metric) & saved["roi"].eq(roi)]
        if len(summary) != 1:
            raise ValueError(f"Expected one matching group-summary row: {folder}")
        reference = summary.iloc[0]
        observed = [len(values), values.mean(), test.statistic, test.pvalue]
        expected = [reference["n"], reference["mean"], reference["t"], reference[p_column]]
        if not np.allclose(observed, expected, atol=1e-10, rtol=1e-8):
            raise ValueError(f"Participant scores do not reproduce saved group tests: {folder}")
        summaries[roi] = {"n": len(values), "mean": values.mean(), "t": test.statistic, "p": test.pvalue}
        rows.append(data)
    frame = pd.concat(rows, ignore_index=True)
    wide = frame.pivot(index="sub", columns="roi", values=metric).reindex(columns=manuscript.ROI_ORDER).sort_index()
    if wide.isna().any().any():
        raise ValueError(f"ROI samples do not match: {root}")
    return frame, wide, metric, baseline, summaries


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


def draw_panel(axis, wide, baseline, summaries, offsets, *, ylabel):
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
    axis.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=6)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f") if baseline == 0 else PercentFormatter(xmax=1, decimals=0))
    style_axis(axis)
    axis.tick_params(axis="x", labelsize=12, pad=9)
    for label, roi in zip(axis.get_xticklabels(), manuscript.ROI_ORDER):
        label.set_color(manuscript.ROI_COLORS[roi])
        label.set_fontweight("bold")


def caption(n):
    return (
        "Figure X. Lateral OFC represents reward identity during conditioning.\n\n"
        "(A) Odor expectation during conditioning. Dashed branches link conditioned "
        "cues to their expected odors, shown in dashed boxes; solid boxes indicate "
        "delivered odors. Odor images are illustrative and were not displayed during "
        "the task. Odor delivery was modeled separately in the single-trial GLMs. "
        "(B) Odor-identity RSA. Spearman correlations between neural and expected "
        "odor-identity RDMs were computed from cue and prediction-period estimates "
        "(0-5 s; EXT5s). "
        "(C) Cross-task decoding. A classifier trained on passive odor responses was "
        "tested on early conditioning cue responses (0-2 s; EARLY). "
        "In B and C, analyses used the same bilateral OFC regions as Figure 3. Points represent "
        f"individual participants (n = {n}), and open diamonds indicate means with "
        "t-based 95% confidence intervals. Dashed lines indicate zero RSA correlation "
        "(B) and 50% chance accuracy (C). Asterisks denote uncorrected p < 0.05 "
        "from one-sided one-sample t tests against these baselines; n.s., not significant.\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output_prefix", type=Path, help="Default: <project_root>/fig/figx")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    prefix = (args.output_prefix or project / "fig" / "figx").expanduser().resolve()
    rsa, rsa_wide, _, rsa_baseline, rsa_stats = load_panel(project, "rsa")
    decoding, decoding_wide, _, decoding_baseline, decoding_stats = load_panel(project, "decoding")
    if not rsa_wide.index.equals(decoding_wide.index):
        raise ValueError("RSA and decoding participant samples differ.")
    # Confirm that both analyses use matching trials and available conditioning runs.
    reference = rsa.loc[rsa["roi"].eq(manuscript.LATERAL_ROI)].set_index("sub")
    comparison = decoding.loc[decoding["roi"].eq(manuscript.LATERAL_ROI)].set_index("sub").reindex(reference.index)
    if not np.array_equal(reference["n_samples"], comparison["n_cue_samples"]):
        raise ValueError("RSA and decoding selected trial counts differ.")
    if not reference["runs"].eq(comparison["cue_runs"]).all():
        raise ValueError("RSA and decoding available conditioning runs differ.")
    if not reference["runs"].isin(["1,2,3", "1,2"]).all():
        raise ValueError("Unexpected available-run pattern in the conditioning inputs.")
    n = len(rsa_wide)
    offsets = manuscript._participant_offsets(rsa_wide.index.tolist(), width=.12)
    configure_matplotlib()
    fig = plt.figure(figsize=(13.4, 5.0))
    schematic = fig.add_axes([.015, .18, .345, .65])
    draw_conditioning_schematic(schematic, project)
    plot_lefts = (.418, .746)
    axes = [fig.add_axes([left, .28, .239, .55]) for left in plot_lefts]
    # Reserve the same two-line heading area for all panels.
    for x, letter, title in [(.02, "A", "Odor expectation\nduring conditioning"),
                             (plot_lefts[0] - .03, "B", "Odor-identity RSA"),
                             (plot_lefts[1] - .03, "C", "Cross-task decoding")]:
        fig.text(x, .965, letter, fontsize=PANEL_SIZE, fontweight="bold", va="top")
        fig.text(x + .03, .965, title, fontsize=TITLE_SIZE, fontweight="bold",
                 va="top", linespacing=1.2)
    draw_panel(axes[0], rsa_wide, rsa_baseline, rsa_stats, offsets,
               ylabel="Spearman ρ")
    draw_panel(axes[1], decoding_wide, decoding_baseline, decoding_stats, offsets,
               ylabel="Decoding accuracy (%)")
    # Trim only the unused bottom margin, retaining a small pad below the labels.
    fig.canvas.draw()
    ink_bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    lower = max(0.0, ink_bounds.y0 - .08)
    export_bounds = Bbox.from_extents(0, lower, fig.get_figwidth(), fig.get_figheight())
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches=export_bounds, pad_inches=0, metadata={
        "Title": "Conditioning odor-identity RSA and cross-task decoding",
        "Subject": "EXT5s all-available-run cross-run RSA; EARLY decoding accuracy; n=" + str(n),
        "Creator": "project-spcd/fig/make_figx.py",
    })
    fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches=export_bounds, pad_inches=0)
    plt.close(fig)
    caption_path = prefix.parent / f"{prefix.name}_caption.txt"
    caption_path.write_text(caption(n))
    for analysis, summaries in [("RSA", rsa_stats), ("Decoding", decoding_stats)]:
        for roi, summary in summaries.items():
            print(analysis, manuscript.ROI_SHORT[roi], summary)
    print(f"Saved {prefix}.pdf, {prefix}.png, and {caption_path}")


if __name__ == "__main__":
    main()
