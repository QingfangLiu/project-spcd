#!/usr/bin/env python3
"""Plot bilateral, left, and right ROI cross-task decoding results."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
if "XDG_CACHE_HOME" not in os.environ:
    font_cache = Path(tempfile.gettempdir()) / "spcd_font_cache"
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(font_cache)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats


METRIC = "cue_balanced_accuracy"
CHANCE = 0.5
DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
ROI_ORDER = (
    "OFC_K2_cluster1_medial_posterior_lateral",
    "OFC_K2_cluster2_central_anterior_lateral",
)
ROI_LABELS = {
    "OFC_K2_cluster1_medial_posterior_lateral": (
        "Medial/posterior-lateral OFC"
    ),
    "OFC_K2_cluster2_central_anterior_lateral": (
        "Central/anterior-lateral OFC"
    ),
}
CUE_ORDER = (
    "preconditioned_odor_only",
    "conditioned_odor_only",
)
CUE_LABELS = {
    "preconditioned_odor_only": "Preconditioned cues",
    "conditioned_odor_only": "Conditioned cues",
}
SCOPE_ORDER = ("bilateral", "left", "right")
SCOPE_LABELS = {
    "bilateral": "Bilateral",
    "left": "Left",
    "right": "Right",
}
SCOPE_COLORS = {
    "bilateral": "#555555",
    "left": "#D84A3A",
    "right": "#3B6EA8",
}
MEAN_COLOR = "#202020"


def _format_p(value: float) -> str:
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def _input_path(results_dir: Path, roi: str, scope: str) -> Path:
    suffix = "" if scope == "bilateral" else f"_{scope}"
    return results_dir / f"roi-{roi}{suffix}_subject_results.csv"


def _load_data(
    results_dir: Path,
    skip_subs: list[str],
) -> pd.DataFrame:
    frames = []
    required = {"sub", "roi", "cue_subset", METRIC}
    for roi in ROI_ORDER:
        for scope in SCOPE_ORDER:
            input_csv = _input_path(results_dir, roi, scope)
            if not input_csv.is_file():
                raise FileNotFoundError(f"Subject results not found: {input_csv}")
            frame = pd.read_csv(input_csv)
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Input CSV missing columns {sorted(missing)}: "
                    f"{input_csv}"
                )
            frame = frame.loc[frame["cue_subset"].isin(CUE_ORDER)].copy()
            frame["roi_base"] = roi
            frame["scope"] = scope
            frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    if skip_subs:
        data = data.loc[~data["sub"].isin(skip_subs)].copy()
    if data.empty:
        raise ValueError("No subject rows remain after exclusions.")
    if data.duplicated(
        ["sub", "roi_base", "scope", "cue_subset"]
    ).any():
        raise ValueError(
            "Duplicate subject/ROI/scope/cue-subset rows in input data."
        )
    if not np.isfinite(data[METRIC].to_numpy(dtype=float)).all():
        raise ValueError(f"{METRIC} contains nonfinite values.")

    expected_cells = {
        (roi, scope, cue)
        for roi in ROI_ORDER
        for scope in SCOPE_ORDER
        for cue in CUE_ORDER
    }
    cells = set(
        zip(data["roi_base"], data["scope"], data["cue_subset"])
    )
    missing_cells = expected_cells - cells
    if missing_cells:
        raise ValueError(
            "Missing ROI/scope/cue cells: "
            + ", ".join(
                " / ".join(cell) for cell in sorted(missing_cells)
            )
        )

    subject_sets = {
        cell: set(
            data.loc[
                (data["roi_base"] == cell[0])
                & (data["scope"] == cell[1])
                & (data["cue_subset"] == cell[2]),
                "sub",
            ]
        )
        for cell in expected_cells
    }
    reference_subjects = subject_sets[
        (ROI_ORDER[0], SCOPE_ORDER[0], CUE_ORDER[0])
    ]
    mismatched = [
        " / ".join(cell)
        for cell, subjects in subject_sets.items()
        if subjects != reference_subjects
    ]
    if mismatched:
        raise ValueError(
            "Cells do not contain the same subjects: "
            + ", ".join(sorted(mismatched))
        )

    return data.sort_values(
        ["roi_base", "cue_subset", "scope", "sub"]
    ).reset_index(drop=True)


def _one_sample_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    n = int(len(values))
    if n < 2:
        raise ValueError("At least two subjects are required.")
    mean = float(values.mean())
    sem = float(stats.sem(values))
    ci_low, ci_high = stats.t.interval(
        0.95,
        n - 1,
        loc=mean,
        scale=sem,
    )
    centered = np.round(values - CHANCE, decimals=12)
    wilcoxon = stats.wilcoxon(
        centered,
        zero_method="wilcox",
        alternative="greater",
    )
    return {
        "n": n,
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "wilcoxon_w": float(wilcoxon.statistic),
        "wilcoxon_p": float(wilcoxon.pvalue),
    }


def _paired_summary(differences: np.ndarray) -> dict[str, float]:
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        raise ValueError("At least two paired subjects are required.")
    rounded = np.round(differences, decimals=12)
    if np.all(rounded == 0):
        wilcoxon_w = 0.0
        wilcoxon_p = 1.0
    else:
        wilcoxon = stats.wilcoxon(
            rounded,
            zero_method="wilcox",
            alternative="two-sided",
        )
        wilcoxon_w = float(wilcoxon.statistic)
        wilcoxon_p = float(wilcoxon.pvalue)
    return {
        "n": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "wilcoxon_w": wilcoxon_w,
        "wilcoxon_p": wilcoxon_p,
    }


def _swarm_offsets(values: np.ndarray) -> np.ndarray:
    positions = np.zeros(len(values), dtype=float)
    rounded = np.round(values, decimals=12)
    for value in np.unique(rounded):
        indices = np.flatnonzero(rounded == value)
        if len(indices) == 1:
            offsets = np.asarray([0.0])
        else:
            width = min(0.12, 0.018 * (len(indices) - 1))
            offsets = np.linspace(-width, width, len(indices))
        positions[indices] = offsets
    return positions


def _draw_chance(ax: plt.Axes) -> None:
    ax.axhline(
        CHANCE,
        color="0.45",
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.text(
        2.42,
        CHANCE - 0.012,
        "Chance",
        color="0.38",
        fontsize=8,
        ha="right",
        va="top",
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.48, 2.48)
    ax.set_ylim(0.18, 0.92)
    ax.set_xticks(range(len(SCOPE_ORDER)))
    ax.set_xticklabels([SCOPE_LABELS[scope] for scope in SCOPE_ORDER])
    ax.set_yticks(np.arange(0.2, 0.81, 0.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(
        axis="y",
        color="0.88",
        linewidth=0.7,
        linestyle="-",
        zorder=-1,
    )


def _draw_group_distribution(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    scope: str,
) -> dict[str, float]:
    summary = _one_sample_summary(values)
    ax.scatter(
        x + _swarm_offsets(values),
        values,
        s=27,
        color=SCOPE_COLORS[scope],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.82,
        zorder=2,
    )
    ax.errorbar(
        x,
        summary["mean"],
        yerr=np.asarray(
            [
                [summary["mean"] - summary["ci_low"]],
                [summary["ci_high"] - summary["mean"]],
            ]
        ),
        fmt="D",
        markersize=6.2,
        markerfacecolor=MEAN_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.7,
        ecolor=MEAN_COLOR,
        elinewidth=1.5,
        capsize=3.5,
        capthick=1.2,
        zorder=4,
    )
    ax.text(
        x,
        summary["ci_high"] + 0.018,
        f"{summary['mean']:.1%}",
        color="0.18",
        fontsize=8.5,
        fontweight="medium",
        ha="center",
        va="bottom",
    )
    ax.text(
        x,
        0.875,
        f"p {_format_p(summary['wilcoxon_p'])}",
        color="0.25",
        fontsize=8.5,
        ha="center",
        va="center",
    )
    return summary


def _save_figure(
    fig: plt.Figure,
    output_prefix: Path,
    formats: list[str],
    dpi: int,
) -> list[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for output_format in formats:
        output_path = output_prefix.with_suffix(f".{output_format}")
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        output_paths.append(output_path)
        print(f"Saved {output_path}")
    plt.close(fig)
    return output_paths


def _plot_summary(
    data: pd.DataFrame,
    output_prefix: Path,
    formats: list[str],
    dpi: int,
) -> dict[tuple[str, str, str], dict[str, float]]:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(10.2, 7.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    summaries = {}
    for row_index, cue in enumerate(CUE_ORDER):
        for column_index, roi in enumerate(ROI_ORDER):
            ax = axes[row_index, column_index]
            _draw_chance(ax)
            _style_axis(ax)
            for x, scope in enumerate(SCOPE_ORDER):
                cell = data.loc[
                    (data["roi_base"] == roi)
                    & (data["scope"] == scope)
                    & (data["cue_subset"] == cue)
                ]
                values = cell[METRIC].to_numpy(dtype=float)
                summaries[(roi, cue, scope)] = _draw_group_distribution(
                    ax,
                    float(x),
                    values,
                    scope,
                )

            if row_index == 0:
                ax.set_title(
                    ROI_LABELS[roi],
                    fontsize=11.5,
                    fontweight="semibold",
                    pad=9,
                )
            if column_index == 1:
                ax.text(
                    1.06,
                    0.5,
                    CUE_LABELS[cue],
                    transform=ax.transAxes,
                    rotation=-90,
                    ha="left",
                    va="center",
                    fontsize=10.5,
                    fontweight="semibold",
                    color="0.22",
                )

    fig.suptitle(
        "Cross-task odor identity decoding by ROI extent",
        fontsize=15,
        fontweight="semibold",
    )
    fig.supylabel("Cross-task decoding accuracy", fontsize=11)
    fig.supxlabel(
        "Dots: subjects  |  Diamond/error bar: mean and 95% CI  |  "
        "p: one-sided Wilcoxon versus 50%",
        fontsize=9,
        color="0.35",
    )
    _save_figure(fig, output_prefix, formats, dpi)
    return summaries


def _wide_scope_data(
    data: pd.DataFrame,
    roi: str,
    cue: str,
) -> pd.DataFrame:
    cell = data.loc[
        (data["roi_base"] == roi)
        & (data["cue_subset"] == cue),
        ["sub", "scope", METRIC],
    ]
    wide = cell.pivot(index="sub", columns="scope", values=METRIC)
    return wide.loc[:, list(SCOPE_ORDER)].sort_index()


def _plot_paired_preconditioned(
    data: pd.DataFrame,
    output_prefix: Path,
    formats: list[str],
    dpi: int,
) -> dict[str, dict[str, dict[str, float]]]:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10.2, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    results = {}
    cue = "preconditioned_odor_only"
    for ax, roi in zip(axes, ROI_ORDER):
        wide = _wide_scope_data(data, roi, cue)
        _draw_chance(ax)
        _style_axis(ax)

        for _, subject_values in wide.iterrows():
            values = subject_values.to_numpy(dtype=float)
            ax.plot(
                range(len(SCOPE_ORDER)),
                values,
                color="0.68",
                linewidth=0.65,
                alpha=0.34,
                zorder=1,
            )
            for x, (scope, value) in enumerate(subject_values.items()):
                ax.scatter(
                    x,
                    value,
                    s=17,
                    color=SCOPE_COLORS[scope],
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.72,
                    zorder=2,
                )

        scope_summaries = {}
        for x, scope in enumerate(SCOPE_ORDER):
            values = wide[scope].to_numpy(dtype=float)
            summary = _one_sample_summary(values)
            scope_summaries[scope] = summary
            ax.errorbar(
                x,
                summary["mean"],
                yerr=np.asarray(
                    [
                        [summary["mean"] - summary["ci_low"]],
                        [summary["ci_high"] - summary["mean"]],
                    ]
                ),
                fmt="D",
                markersize=6.5,
                markerfacecolor=MEAN_COLOR,
                markeredgecolor="white",
                markeredgewidth=0.7,
                ecolor=MEAN_COLOR,
                elinewidth=1.6,
                capsize=4,
                capthick=1.3,
                zorder=5,
            )
            ax.text(
                x,
                summary["ci_high"] + 0.018,
                f"{summary['mean']:.1%}",
                fontsize=8.5,
                fontweight="medium",
                color="0.18",
                ha="center",
                va="bottom",
            )

        left_right = _paired_summary(
            wide["left"].to_numpy() - wide["right"].to_numpy()
        )
        hemisphere_average = (
            wide["left"].to_numpy() + wide["right"].to_numpy()
        ) / 2
        bilateral_average = _paired_summary(
            wide["bilateral"].to_numpy() - hemisphere_average
        )
        results[roi] = {
            "scopes": scope_summaries,
            "left_minus_right": left_right,
            "bilateral_minus_hemisphere_average": bilateral_average,
        }

        ax.text(
            0.5,
            0.968,
            (
                "Left vs right p "
                f"{_format_p(left_right['wilcoxon_p'])}  |  "
                "Bilateral vs mean(L,R) p "
                f"{_format_p(bilateral_average['wilcoxon_p'])}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.7,
        )
        ax.set_title(
            ROI_LABELS[roi],
            fontsize=11.5,
            fontweight="semibold",
            pad=9,
        )

    fig.suptitle(
        "Within-subject decoding by ROI extent: preconditioned cues",
        fontsize=14.5,
        fontweight="semibold",
    )
    fig.supylabel("Cross-task decoding accuracy", fontsize=11)
    fig.supxlabel(
        "Lines connect the same subject  |  Paired p-values are two-sided "
        "rounding-stable Wilcoxon tests",
        fontsize=9,
        color="0.35",
    )
    _save_figure(fig, output_prefix, formats, dpi)
    return results


def _bilateral_roi_comparison(data: pd.DataFrame) -> dict[str, float]:
    cue = "preconditioned_odor_only"
    cell = data.loc[
        (data["cue_subset"] == cue)
        & (data["scope"] == "bilateral"),
        ["sub", "roi_base", METRIC],
    ]
    wide = cell.pivot(index="sub", columns="roi_base", values=METRIC)
    differences = (
        wide[ROI_ORDER[1]].to_numpy() - wide[ROI_ORDER[0]].to_numpy()
    )
    return _paired_summary(differences)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bilateral, left, and right cross-task decoding for "
            "the two OFC K=2 ROIs."
        )
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing bilateral and unilateral subject CSVs.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for comparison plots.",
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
        help="Subjects to exclude. Default: sub-19 sub-33.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats. Default: pdf png.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    data = _load_data(results_dir, args.skip_subs)

    summary_prefix = (
        output_dir
        / "OFC_K2_bilateral_hemisphere_crosstask_decoding_summary"
    )
    paired_prefix = (
        output_dir
        / "OFC_K2_bilateral_hemisphere_preconditioned_paired"
    )
    summaries = _plot_summary(
        data,
        summary_prefix,
        args.formats,
        args.dpi,
    )
    paired_results = _plot_paired_preconditioned(
        data,
        paired_prefix,
        args.formats,
        args.dpi,
    )
    roi_comparison = _bilateral_roi_comparison(data)

    print("\nOne-sample summaries:")
    for roi in ROI_ORDER:
        for cue in CUE_ORDER:
            for scope in SCOPE_ORDER:
                summary = summaries[(roi, cue, scope)]
                print(
                    f"{roi} / {cue} / {scope}: "
                    f"n={summary['n']}, mean={summary['mean']:.6f}, "
                    f"95% CI=[{summary['ci_low']:.6f}, "
                    f"{summary['ci_high']:.6f}], "
                    f"one-sided Wilcoxon "
                    f"W={summary['wilcoxon_w']:.1f}, "
                    f"p={summary['wilcoxon_p']:.6f}"
                )

    print("\nPaired preconditioned summaries:")
    for roi in ROI_ORDER:
        result = paired_results[roi]
        left_right = result["left_minus_right"]
        bilateral_average = result[
            "bilateral_minus_hemisphere_average"
        ]
        print(
            f"{roi}: left-right mean="
            f"{left_right['mean_difference']:.6f}, "
            f"p={left_right['wilcoxon_p']:.6f}; "
            f"bilateral-mean(left,right)="
            f"{bilateral_average['mean_difference']:.6f}, "
            f"p={bilateral_average['wilcoxon_p']:.6f}"
        )
    print(
        "Bilateral central-minus-medial preconditioned mean="
        f"{roi_comparison['mean_difference']:.6f}, "
        f"two-sided Wilcoxon p={roi_comparison['wilcoxon_p']:.6f}"
    )


if __name__ == "__main__":
    main()
