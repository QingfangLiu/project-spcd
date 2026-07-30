#!/usr/bin/env python3
"""Plot the two-ROI by two-cue cross-task decoding accuracy figure."""

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
DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
CHANCE = 0.5
CUE_ORDER = (
    "preconditioned_odor_only",
    "conditioned_odor_only",
)
CUE_LABELS = {
    "preconditioned_odor_only": "Preconditioned cues",
    "conditioned_odor_only": "Conditioned cues",
}
CUE_COLORS = {
    "preconditioned_odor_only": "#D84A3A",
    "conditioned_odor_only": "#3B6EA8",
}
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
MEAN_COLOR = "#202020"


def _format_p(value: float) -> str:
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def _load_data(
    input_csvs: list[Path],
    skip_subs: list[str],
) -> pd.DataFrame:
    frames = []
    required = {"sub", "roi", "cue_subset", METRIC}
    for input_csv in input_csvs:
        frame = pd.read_csv(input_csv)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"Input CSV missing required columns {sorted(missing)}: "
                f"{input_csv}"
            )
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.loc[
        data["roi"].isin(ROI_ORDER)
        & data["cue_subset"].isin(CUE_ORDER)
    ].copy()
    if skip_subs:
        data = data.loc[~data["sub"].isin(skip_subs)].copy()
    if data.empty:
        raise ValueError("No subject rows remain after filtering.")
    if data.duplicated(["sub", "roi", "cue_subset"]).any():
        raise ValueError(
            "Input CSVs contain duplicate subject/ROI/cue-subset rows."
        )

    cells = set(zip(data["roi"], data["cue_subset"]))
    expected_cells = {
        (roi, cue_subset)
        for roi in ROI_ORDER
        for cue_subset in CUE_ORDER
    }
    missing_cells = expected_cells - cells
    if missing_cells:
        raise ValueError(
            "Missing ROI/cue-subset cells: "
            + ", ".join(
                f"{roi} / {cue_subset}"
                for roi, cue_subset in sorted(missing_cells)
            )
        )
    if not np.isfinite(data[METRIC].to_numpy(dtype=float)).all():
        raise ValueError(f"{METRIC} contains nonfinite values.")

    subject_sets = {
        (roi, cue_subset): set(
            data.loc[
                (data["roi"] == roi)
                & (data["cue_subset"] == cue_subset),
                "sub",
            ]
        )
        for roi, cue_subset in expected_cells
    }
    reference_subjects = subject_sets[(ROI_ORDER[0], CUE_ORDER[0])]
    mismatched_cells = [
        f"{roi} / {cue_subset}"
        for (roi, cue_subset), subjects in subject_sets.items()
        if subjects != reference_subjects
    ]
    if mismatched_cells:
        raise ValueError(
            "ROI/cue-subset cells do not contain the same subjects: "
            + ", ".join(sorted(mismatched_cells))
        )

    return data.sort_values(
        ["roi", "cue_subset", "sub"]
    ).reset_index(drop=True)


def _summary(values: np.ndarray) -> dict[str, float]:
    n = int(len(values))
    if n < 2:
        raise ValueError("At least two subjects are required.")

    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    ci_low, ci_high = stats.t.interval(
        0.95,
        n - 1,
        loc=mean,
        scale=sem,
    )

    # Accuracy is discrete; rounding prevents machine-precision differences
    # from changing tie ranks in the Wilcoxon test.
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
        "wilcoxon_p_greater": float(wilcoxon.pvalue),
    }


def _swarm_positions(values: np.ndarray) -> np.ndarray:
    """Spread tied accuracy values horizontally without random jitter."""
    positions = np.zeros(len(values), dtype=float)
    rounded = np.round(values, decimals=12)
    for value in np.unique(rounded):
        indices = np.flatnonzero(rounded == value)
        if len(indices) == 1:
            offsets = np.asarray([0.0])
        else:
            width = min(0.14, 0.02 * (len(indices) - 1))
            offsets = np.linspace(-width, width, len(indices))
        positions[indices] = offsets
    return positions


def _plot_panel(
    ax: plt.Axes,
    values: np.ndarray,
    cue_subset: str,
) -> dict[str, float]:
    summary = _summary(values)
    subject_x = _swarm_positions(values)
    mean_x = 0.0

    ax.axhline(
        CHANCE,
        color="0.42",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=0,
    )
    ax.text(
        0.31,
        CHANCE - 0.014,
        "Chance",
        color="0.35",
        fontsize=8.5,
        ha="right",
        va="top",
    )

    ax.scatter(
        subject_x,
        values,
        s=34,
        color=CUE_COLORS[cue_subset],
        edgecolor="white",
        linewidth=0.55,
        alpha=0.84,
        zorder=2,
    )

    ax.errorbar(
        mean_x,
        summary["mean"],
        yerr=np.asarray(
            [
                [summary["mean"] - summary["ci_low"]],
                [summary["ci_high"] - summary["mean"]],
            ]
        ),
        fmt="D",
        markersize=6.8,
        markerfacecolor=MEAN_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.8,
        ecolor=MEAN_COLOR,
        elinewidth=1.6,
        capsize=4,
        capthick=1.3,
        zorder=4,
    )
    ax.text(
        mean_x,
        summary["ci_high"] + 0.025,
        f"{summary['mean']:.1%}",
        color="0.18",
        fontsize=9,
        fontweight="medium",
        ha="center",
        va="bottom",
    )

    p_text = _format_p(summary["wilcoxon_p_greater"])
    ax.text(
        0.5,
        0.965,
        f"One-sided Wilcoxon p {p_text}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
    )
    ax.set_xlim(-0.34, 0.34)
    ax.set_ylim(0.18, 0.92)
    ax.set_xticks([0.0])
    ax.set_xticklabels([f"Subjects (n = {summary['n']})"])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_yticks(np.arange(0.2, 0.81, 0.1))

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
    return summary


def _plot(
    data: pd.DataFrame,
    output_prefix: Path,
    formats: list[str],
    dpi: int,
    title: str,
) -> tuple[list[Path], dict[tuple[str, str], dict[str, float]]]:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(13.2, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    summaries = {}
    panel_index = 0
    for roi in ROI_ORDER:
        for cue_subset in CUE_ORDER:
            ax = axes[panel_index]
            panel_index += 1
            cell = data.loc[
                (data["roi"] == roi)
                & (data["cue_subset"] == cue_subset)
            ]
            values = cell[METRIC].to_numpy(dtype=float)
            summaries[(roi, cue_subset)] = _plot_panel(
                ax=ax,
                values=values,
                cue_subset=cue_subset,
            )

            ax.set_title(
                f"{CUE_LABELS[cue_subset]}\n{ROI_LABELS[roi]}",
                fontsize=10,
                fontweight="semibold",
                pad=9,
                color=CUE_COLORS[cue_subset],
            )

    fig.suptitle(title, fontsize=15, fontweight="semibold")
    fig.supylabel("Cross-task decoding accuracy", fontsize=11)
    fig.supxlabel(
        "Train: passive odor perception  |  Test: premeal cues",
        fontsize=9.5,
        color="0.35",
    )

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
    return output_paths, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the two OFC ROIs by two cue subsets for cross-task "
            "decoding accuracy, with one-sided Wilcoxon tests against "
            "chance."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        nargs="+",
        required=True,
        help="The two ROI subject-results CSVs.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
        help="Subjects to exclude. Default: sub-19 sub-33.",
    )
    parser.add_argument(
        "--title",
        default="Cross-task odor identity decoding in OFC",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats. Default: pdf png.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    input_csvs = [
        input_csv.expanduser().resolve()
        for input_csv in args.input_csv
    ]
    if len(input_csvs) != 2:
        raise ValueError(
            "--input_csv requires exactly two ROI subject-results CSVs."
        )
    output_prefix = args.output_prefix.expanduser().resolve()
    if output_prefix.suffix:
        raise ValueError(
            "--output_prefix must not include a filename extension: "
            f"{output_prefix}"
        )
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")

    data = _load_data(
        input_csvs=input_csvs,
        skip_subs=args.skip_subs,
    )
    _, summaries = _plot(
        data=data,
        output_prefix=output_prefix,
        formats=args.formats,
        dpi=args.dpi,
        title=args.title,
    )
    for roi in ROI_ORDER:
        for cue_subset in CUE_ORDER:
            summary = summaries[(roi, cue_subset)]
            print(
                f"{roi} / {cue_subset}: "
                f"subjects={summary['n']}; "
                f"mean={summary['mean']:.6f}; "
                f"95% CI=[{summary['ci_low']:.6f}, "
                f"{summary['ci_high']:.6f}]; "
                f"one-sided Wilcoxon "
                f"W={summary['wilcoxon_w']:.1f}, "
                f"p={summary['wilcoxon_p_greater']:.6f}"
            )


if __name__ == "__main__":
    main()
