#!/usr/bin/env python3
"""Plot two OFC ROIs by two separately analyzed cue subsets."""

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
import numpy as np
import pandas as pd
from scipy import stats

from aggregate_roi_odor_template_alignment import _bootstrap_mean_ci


METRIC = "matching_correlation_advantage"
DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
ROIS = (
    "OFC_K2_cluster1_medial_posterior_lateral",
    "OFC_K2_cluster2_central_anterior_lateral",
)
ROI_LABELS = {
    ROIS[0]: "Medial/posterior-lateral OFC",
    ROIS[1]: "Central/anterior-lateral OFC",
}
CUES = ("preconditioned_odor_only", "conditioned_odor_only")
CUE_LABELS = {
    CUES[0]: "Preconditioned cues",
    CUES[1]: "Conditioned cues",
}
CUE_COLORS = {CUES[0]: "#D84A3A", CUES[1]: "#3B6EA8"}


def _format_p(value: float) -> str:
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def _load_data(
    input_csvs: list[Path],
    hemisphere: str,
    skip_subs: list[str],
) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in input_csvs]
    data = pd.concat(frames, ignore_index=True)
    required = {"sub", "roi", "hemisphere", "cue_subset", METRIC}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Input CSVs missing columns: {sorted(missing)}")
    data = data.loc[
        (data["hemisphere"] == hemisphere)
        & data["roi"].isin(ROIS)
        & data["cue_subset"].isin(CUES)
        & ~data["sub"].isin(skip_subs)
    ].copy()
    if data.duplicated(["sub", "roi", "cue_subset"]).any():
        raise ValueError("Input CSVs contain duplicate subject/cell rows.")
    expected = {(roi, cue) for roi in ROIS for cue in CUES}
    observed = set(zip(data["roi"], data["cue_subset"]))
    if expected != observed:
        raise ValueError(f"Missing plot cells: {sorted(expected - observed)}")
    subject_sets = {
        cell: set(
            data.loc[
                (data["roi"] == cell[0])
                & (data["cue_subset"] == cell[1]),
                "sub",
            ]
        )
        for cell in expected
    }
    if len({frozenset(subjects) for subjects in subject_sets.values()}) != 1:
        raise ValueError("Plot cells do not contain identical subject sets.")
    return data


def _summary(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    if n < 2:
        raise ValueError("At least two subjects are required.")
    mean = float(values.mean())
    ci_low, ci_high = _bootstrap_mean_ci(values)
    rounded = np.round(values, decimals=12)
    if np.all(rounded == 0):
        p_value = np.nan
    else:
        p_value = float(
            stats.wilcoxon(
                rounded,
                zero_method="wilcox",
                alternative="greater",
            ).pvalue
        )
    return {
        "n": n,
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p": p_value,
    }


def _plot(
    data: pd.DataFrame,
    hemisphere: str,
    output_prefix: Path,
    formats: list[str],
    dpi: int,
) -> None:
    limit = max(0.02, float(np.abs(data[METRIC]).max()) * 1.25)
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12.8, 4.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panel = 0
    for roi in ROIS:
        for cue in CUES:
            ax = axes[panel]
            panel += 1
            cell = data.loc[
                (data["roi"] == roi) & (data["cue_subset"] == cue)
            ].sort_values("sub")
            values = cell[METRIC].to_numpy(dtype=float)
            summary = _summary(values)
            offsets = np.linspace(-0.07, 0.07, len(values))
            ax.axhline(0, color="0.45", linewidth=1, linestyle=(0, (4, 3)))
            ax.scatter(
                offsets,
                values,
                s=31,
                color=CUE_COLORS[cue],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.82,
                zorder=2,
            )
            ax.errorbar(
                0,
                summary["mean"],
                yerr=np.asarray(
                    [
                        [summary["mean"] - summary["ci_low"]],
                        [summary["ci_high"] - summary["mean"]],
                    ]
                ),
                fmt="D",
                markersize=6.5,
                markerfacecolor="#202020",
                markeredgecolor="white",
                ecolor="#202020",
                elinewidth=1.5,
                capsize=4,
                zorder=4,
            )
            p_text = "NA" if np.isnan(summary["p"]) else _format_p(summary["p"])
            ax.text(
                0.5,
                0.96,
                f"One-sided Wilcoxon p {p_text}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
            ax.set_title(
                f"{CUE_LABELS[cue]}\n{ROI_LABELS[roi]}",
                fontsize=9.7,
                fontweight="semibold",
                pad=8,
            )
            ax.set_xlim(-0.12, 0.12)
            ax.set_ylim(-limit, limit)
            ax.set_xticks([0])
            ax.set_xticklabels([f"n = {summary['n']}"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="0.88", linewidth=0.7)
            ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Passive odor-template alignment: {hemisphere}",
        fontsize=14.5,
        fontweight="semibold",
    )
    fig.supylabel("Matching - nonmatching prototype correlation", fontsize=11)
    fig.supxlabel(
        "Dots: subjects  |  Diamond/error bar: mean and 95% CI",
        fontsize=9,
        color="0.35",
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for output_format in formats:
        path = output_prefix.with_suffix(f".{output_format}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot OFC K=2 passive-odor template alignment."
    )
    parser.add_argument("--input_csv", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--hemisphere",
        choices=("bilateral", "left", "right"),
        required=True,
    )
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument(
        "--skip_subs", nargs="*", default=list(DEFAULT_SKIP_SUBS)
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    data = _load_data(args.input_csv, args.hemisphere, args.skip_subs)
    _plot(
        data,
        args.hemisphere,
        args.output_prefix.expanduser().resolve(),
        args.formats,
        args.dpi,
    )


if __name__ == "__main__":
    main()
