#!/usr/bin/env python3
"""Rebuild the two supplementary choice/pleasantness correlation figures.

Run from any directory:
    python3 /path/to/project-spcd/scripts_behavior/make_probe_pleasantness_figures.py

Source: corr_probe_beh_odor_rating.ipynb, plotting cells 11 and 20 (zero-based).
The original outputs are plots/probe_change_corr.pdf and
plots/probe_corr_pre_post.pdf. This generator reads subject-level pre/post
meal trials and pleasantness_summary.csv, rather than the cached group CSV.
Choice filtering/orientation matches make_behavior_figure.py: drop trials
with missing choice_class, acc, or rt; average within subject/comparison;
relabel A3/B3 as C/D; reverse A1>A2 and B1>B2. Default exclusions are subjects
19, 31, and 33. Missing pleasantness data are reported and omitted from both
figures. All six comparisons use the same complete participant sample.

Each point is one participant. Rating differences are first-cue minus
second-cue odor pleasantness; choices refer to the first cue. Changes are
post minus pre. Correlations are two-sided Pearson tests, uncorrected.
Lines are OLS fits. As in the original seaborn plots, shading is a pointwise
95% percentile interval for the fitted mean from resampling participant
(x, y) pairs; 10,000 resamples and fixed seeds make the bands reproducible.
Fits and intervals are restricted to the observed x range and are not
clipped to [0, 1]. Points are not jittered. Neutral phase colors avoid
repurposing the main behavior figure's odor-outcome colors.

Outputs: two vector PDFs and two 300-dpi PNGs under figures/supplementary/;
source observations and full precision statistics under beh_processed_subs/.
Existing notebook outputs and the manuscript are left untouched.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import (
    MUTED_TEXT_COLOR, NOTE_SIZE, REFERENCE_COLOR, SUMMARY_COLOR,
    add_panel_header, configure_matplotlib, style_axis,
)
from scripts_behavior.make_behavior_figure import DEFAULT_SKIP_SUBS


COMPARISONS = ("A1>C", "A2>A1", "A2>C", "B1>D", "B2>B1", "B2>D")
DESCRIPTIONS = ("Non-sated > air", "Sated > non-sated", "Sated > air") * 2
CUE_TO_ODOR = {"A1": 1, "A2": 2, "C": 0, "B1": 1, "B2": 2, "D": 0}
PHASE_COLORS = {"pre": "#929CA5", "post": SUMMARY_COLOR, "change": MUTED_TEXT_COLOR}
PHASE_MARKERS = {"pre": "o", "post": "D", "change": "o"}
BOOTSTRAP_SEED = 20261023


def build_source_data(project_root: Path, skip_subs: tuple[str, ...]) -> pd.DataFrame:
    """Reconstruct choices, join both odors, and require complete cells."""
    root = project_root / "beh_processed_subs"
    subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir() and p.name not in skip_subs)
    if len(subjects) < 3:
        raise ValueError(f"Fewer than three retained subject directories in {root}")

    phase_tables = []
    for phase in ("pre", "post"):
        trials = pd.concat(
            [pd.read_csv(root / subject / f"{phase}_meal_dat.csv").assign(subject=subject)
             for subject in subjects], ignore_index=True,
        ).dropna(subset=["choice_class", "acc", "rt"])
        means = trials.groupby(["subject", "choice_class"], as_index=False)["acc"].mean()
        means["choice_class"] = (
            means["choice_class"].str.replace("A3", "C", regex=False)
            .str.replace("B3", "D", regex=False)
        )
        reverse = means["choice_class"].isin(["A1>A2", "B1>B2"])
        means.loc[reverse, "acc"] = 1 - means.loc[reverse, "acc"]
        means["choice_class"] = means["choice_class"].replace({"A1>A2": "A2>A1", "B1>B2": "B2>B1"})
        phase_tables.append(means.assign(phase=phase))
    choices = pd.concat(phase_tables).pivot(
        index=["subject", "choice_class"], columns="phase", values="acc",
    ).reset_index()
    expected = {(s, c) for s in subjects for c in COMPARISONS}
    observed = set(zip(choices["subject"], choices["choice_class"]))
    if observed != expected:
        raise ValueError(f"Choice cells disagree: missing={sorted(expected - observed)}, extra={sorted(observed - expected)}")
    if not np.isfinite(choices[["pre", "post"]].to_numpy()).all():
        raise ValueError("Nonfinite pre/post choice proportions")
    if not choices[["pre", "post"]].apply(lambda col: col.between(0, 1)).all().all():
        raise ValueError("Choice proportions outside [0, 1]")
    choices["change"] = choices["post"] - choices["pre"]

    rating_tables = []
    for subject in subjects:
        path = root / subject / "pleasantness_summary.csv"
        if not path.is_file():
            print(f"Missing pleasantness data: {subject} (omitted from both figures)")
            continue
        frame = pd.read_csv(path).assign(subject=subject)
        if frame["odor_num"].duplicated().any() or set(frame["odor_num"]) != {0, 1, 2}:
            raise ValueError(f"Expected exactly three unique odor rows in {path}")
        if not np.isfinite(frame[["pre_mean", "post_mean", "change"]].to_numpy()).all():
            raise ValueError(f"Nonfinite pleasantness values in {path}")
        if not np.allclose(frame["post_mean"] - frame["pre_mean"], frame["change"], rtol=0, atol=1e-10):
            raise ValueError(f"Stored rating change disagrees with post - pre in {path}")
        rating_tables.append(frame)
    if not rating_tables:
        raise ValueError("No pleasantness summaries for retained subjects")
    ratings = pd.concat(rating_tables, ignore_index=True)
    choices = choices.loc[choices["subject"].isin(ratings["subject"])].copy()
    choices[["first_cue", "second_cue"]] = choices["choice_class"].str.split(">", expand=True)
    for side in ("first", "second"):
        choices[f"{side}_odor"] = choices[f"{side}_cue"].map(CUE_TO_ODOR)
        choices = choices.merge(
            ratings[["subject", "odor_num", "pre_mean", "post_mean"]].rename(
                columns={"pre_mean": f"{side}_pre", "post_mean": f"{side}_post"},
            ), left_on=["subject", f"{side}_odor"], right_on=["subject", "odor_num"],
            how="left", validate="many_to_one",
        ).drop(columns="odor_num")

    tables = []
    for phase in ("pre", "post", "change"):
        if phase == "change":
            first = choices["first_post"] - choices["first_pre"]
            second = choices["second_post"] - choices["second_pre"]
        else:
            first, second = choices[f"first_{phase}"], choices[f"second_{phase}"]
        tables.append(choices[["subject", "choice_class", "first_cue", "second_cue", "first_odor", "second_odor"]].assign(
            phase=phase, first_odor_pleasantness=first, second_odor_pleasantness=second,
            rating_difference=first - second, choice_proportion=choices[phase],
        ))
    source = pd.concat(tables, ignore_index=True)
    if not np.isfinite(source[["rating_difference", "choice_proportion"]].to_numpy()).all():
        raise ValueError("Nonfinite values after joining choices and ratings")
    if source["subject"].nunique() < 3:
        raise ValueError("At least three participants are required for regression")
    return source.sort_values(["choice_class", "phase", "subject"]).reset_index(drop=True)


def compute_statistics(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison in COMPARISONS:
        for phase in ("pre", "post", "change"):
            group = source.loc[source["choice_class"].eq(comparison) & source["phase"].eq(phase)]
            x, y = group["rating_difference"].to_numpy(), group["choice_proportion"].to_numpy()
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                raise ValueError(f"Cannot correlate constant values: {comparison}, {phase}")
            correlation = stats.pearsonr(x, y)
            fit = stats.linregress(x, y)
            rows.append({
                "choice_class": comparison, "phase": phase, "n": len(group),
                "pearson_r": float(correlation.statistic), "p_two_sided_uncorrected": float(correlation.pvalue),
                "ols_slope": float(fit.slope), "ols_intercept": float(fit.intercept),
                "ols_slope_se": float(fit.stderr), "bootstrap_resamples": 10000,
                "bootstrap_seed": BOOTSTRAP_SEED + COMPARISONS.index(comparison) * 3 + ("pre", "post", "change").index(phase),
            })
    return pd.DataFrame(rows)


def regression_band(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bootstrap of OLS mean predictions using paired resamples."""
    grid = np.linspace(x.min(), x.max(), 200)
    indices = np.random.default_rng(seed).integers(0, len(x), size=(10000, len(x)))
    bx, by = x[indices], y[indices]
    mx, my = bx.mean(axis=1), by.mean(axis=1)
    centered = bx - mx[:, None]
    ssx = (centered ** 2).sum(axis=1)
    valid = ssx > 0
    if valid.sum() < 9900:
        raise ValueError("Too many degenerate bootstrap samples")
    slopes = (centered[valid] * (by[valid] - my[valid, None])).sum(axis=1) / ssx[valid]
    predictions = my[valid, None] + slopes[:, None] * (grid[None, :] - mx[valid, None])
    low, high = np.percentile(predictions, [2.5, 97.5], axis=0)
    return grid, low, high


def _correlation_label(row: pd.Series) -> str:
    p = row["p_two_sided_uncorrected"]
    p_text = "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"
    return f"$r$ = {row['pearson_r']:.2f}, {p_text}"


def make_figure(source: pd.DataFrame, statistics: pd.DataFrame, *, change: bool) -> plt.Figure:
    configure_matplotlib()
    phases = ("change",) if change else ("pre", "post")
    selected = source.loc[source["phase"].isin(phases)]
    bands = {}
    for comparison in COMPARISONS:
        for phase in phases:
            group = selected.loc[selected["choice_class"].eq(comparison) & selected["phase"].eq(phase)]
            row = statistics.loc[statistics["choice_class"].eq(comparison) & statistics["phase"].eq(phase)].iloc[0]
            bands[comparison, phase] = regression_band(
                group["rating_difference"].to_numpy(), group["choice_proportion"].to_numpy(),
                int(row["bootstrap_seed"]),
            )
    # Include every observation and the full interval, with shared limits.
    x_min, x_max = selected["rating_difference"].agg(["min", "max"])
    x_limits = (min(-4.4, x_min - 0.2), max(2.1, x_max + 0.2)) if change else (
        min(-6.2, x_min - 0.2), max(6.2, x_max + 0.2),
    )
    band_min = min(band[1].min() for band in bands.values())
    band_max = max(band[2].max() for band in bands.values())
    y_limits = (min(-1.08, band_min - 0.05), max(1.08, band_max + 0.05)) if change else (
        min(-0.55, np.floor((band_min - 0.05) * 10) / 10),
        max(1.5, np.ceil((band_max + 0.05) * 10) / 10),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 10.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.84, bottom=0.13, wspace=0.29, hspace=0.61)
    n = source["subject"].nunique()
    fig.text(0.11, 0.975, "Preconditioned cues", fontsize=18, fontweight="bold", va="top")
    fig.text(0.985, 0.975, f"n = {n} participants", fontsize=NOTE_SIZE, color=MUTED_TEXT_COLOR, ha="right", va="top")
    lower_top = axes[1, 0].get_position().y1
    fig.text(0.11, lower_top + 0.12, "Conditioned cues", fontsize=18, fontweight="bold", va="top")

    for i, (ax, comparison, description) in enumerate(zip(axes.flat, COMPARISONS, DESCRIPTIONS)):
        style_axis(ax)
        add_panel_header(ax, "ABCDEF"[i], f"{description}\n({comparison.replace('>', ' > ')})")
        ax.axhline(0 if change else 0.5, color=REFERENCE_COLOR,
                   linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
        for j, phase in enumerate(phases):
            group = source.loc[source["choice_class"].eq(comparison) & source["phase"].eq(phase)]
            row = statistics.loc[statistics["choice_class"].eq(comparison) & statistics["phase"].eq(phase)].iloc[0]
            x, y = group["rating_difference"].to_numpy(), group["choice_proportion"].to_numpy()
            grid, low, high = bands[comparison, phase]
            color = PHASE_COLORS[phase]
            ax.fill_between(grid, low, high, color=color, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(grid, row["ols_intercept"] + row["ols_slope"] * grid, color=color,
                    linewidth=2, linestyle="--" if phase == "pre" else "-", zorder=3)
            ax.scatter(x, y, color=color, marker=PHASE_MARKERS[phase], s=35,
                       alpha=0.57 if phase != "post" else 0.69,
                       edgecolors="white", linewidths=0.45, zorder=4)
            prefix = "" if change else ("Pre: " if phase == "pre" else "Post: ")
            ax.text(0.04, 0.96 - j * 0.095, prefix + _correlation_label(row),
                    transform=ax.transAxes, fontsize=12, color=SUMMARY_COLOR,
                    va="top", zorder=5)
        ax.set(xlim=x_limits, ylim=y_limits,
               xticks=[-4, -2, 0, 2] if change else [-6, -3, 0, 3, 6],
               yticks=[-1, -0.5, 0, 0.5, 1] if change else [-0.5, 0, 0.5, 1, 1.5])

    if not change:
        fig.legend(handles=[
            Line2D([0], [0], color=PHASE_COLORS[p], marker=PHASE_MARKERS[p],
                   linestyle="--" if p == "pre" else "-", linewidth=2,
                   markersize=6, label="Pre-meal" if p == "pre" else "Post-meal")
            for p in ("pre", "post")
        ], loc="upper center", bbox_to_anchor=(0.56, 0.947), ncol=2,
           frameon=False, handlelength=2.4, columnspacing=2)
    fig.supxlabel("Relative change in odor pleasantness (post - pre)" if change
                  else "Odor pleasantness difference (first cue - second cue)", y=0.065, fontsize=16)
    fig.supylabel("Change in choice proportion (post - pre)" if change
                  else "Choice proportion for the first cue", x=0.018, fontsize=16)
    fig.text(0.55, 0.018, "Points: participants   |   Lines: linear fits   |   Shading: 95% confidence intervals",
             ha="center", fontsize=11, color=MUTED_TEXT_COLOR)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Figure output directory (default: <project>/figures/supplementary). CSVs always go to <project>/beh_processed_subs.")
    parser.add_argument("--skip-subs", nargs="*", default=list(DEFAULT_SKIP_SUBS))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    root = args.project_root.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve() if args.output_dir else root / "figures/supplementary"
    data_destination = root / "beh_processed_subs"
    source = build_source_data(root, tuple(args.skip_subs))
    statistics = compute_statistics(source)
    print(f"Excluded subjects: {', '.join(args.skip_subs) or '(none)'}")
    print(f"Included participants (n = {source.subject.nunique()}): {', '.join(sorted(source.subject.unique()))}")
    print(statistics[["choice_class", "phase", "n", "pearson_r", "p_two_sided_uncorrected"]].to_string(index=False))
    destination.mkdir(parents=True, exist_ok=True)
    source.to_csv(data_destination / "probe_pleasantness_source_data.csv", index=False)
    statistics.to_csv(data_destination / "probe_pleasantness_statistics.csv", index=False)
    for stem, change in (("probe_change_corr", True), ("probe_corr_pre_post", False)):
        figure = make_figure(source, statistics, change=change)
        for extension in ("pdf", "png"):
            path = destination / f"{stem}.{extension}"
            figure.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.08)
            print(f"Saved: {path}")
        plt.close(figure)


if __name__ == "__main__":
    main()
