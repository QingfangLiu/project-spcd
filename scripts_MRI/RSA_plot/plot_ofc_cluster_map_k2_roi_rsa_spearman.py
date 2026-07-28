#!/usr/bin/env python3
"""Plot paired OFC K=2 subject-level Spearman ROI RSA results."""

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


METRIC = "odor_identity_no_image_spearman_rho"
DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
CUE_ORDER = ("conditioned_odor_only", "preconditioned_odor_only")
CUE_LABELS = {
    "conditioned_odor_only": "Conditioned",
    "preconditioned_odor_only": "Preconditioned",
}
CUE_COLORS = {
    "conditioned_odor_only": "#3B6EA8",
    "preconditioned_odor_only": "#D84A3A",
}


def _format_p(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    value = float(value)
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def _load_subject_data(
    input_csv: Path,
    skip_subs: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"sub", "cue_subset", METRIC}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns {sorted(missing)}: "
            f"{input_csv}"
        )
    if skip_subs:
        df = df.loc[~df["sub"].isin(skip_subs)].copy()

    wide = df.pivot_table(
        index="sub",
        columns="cue_subset",
        values=METRIC,
        aggfunc="first",
    )
    missing_cues = [cue for cue in CUE_ORDER if cue not in wide.columns]
    if missing_cues:
        raise ValueError(
            f"Input CSV missing cue subsets {missing_cues}: {input_csv}"
        )
    wide = wide.loc[
        wide[list(CUE_ORDER)].notna().all(axis=1),
        list(CUE_ORDER),
    ]
    if wide.empty:
        raise ValueError("No complete subject pairs remain after exclusions.")

    return (
        wide.reset_index()
        .melt(
            id_vars="sub",
            var_name="cue_subset",
            value_name=METRIC,
        )
        .sort_values(["sub", "cue_subset"])
        .reset_index(drop=True)
    )


def _load_group_summary(summary_csv: Path) -> dict[str, dict]:
    if not summary_csv.exists():
        return {}
    summary = pd.read_csv(summary_csv)
    if "metric" not in summary.columns:
        raise ValueError(
            f"Group summary has no metric column: {summary_csv}"
        )
    summary = summary.loc[summary["metric"] == METRIC]
    return {
        str(row["contrast"]): row.to_dict()
        for _, row in summary.iterrows()
    }


def _summary_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cue in CUE_ORDER:
        values = data.loc[
            data["cue_subset"] == cue,
            METRIC,
        ].to_numpy(dtype=float)
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(
            {
                "cue_subset": cue,
                "mean": float(np.mean(values)),
                "sem": (
                    float(sd / np.sqrt(len(values)))
                    if len(values) > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _annotation_text(summary: dict[str, dict]) -> str:
    conditioned = summary.get("conditioned_odor_only_gt_zero", {})
    preconditioned = summary.get(
        "preconditioned_odor_only_gt_zero",
        {},
    )
    paired = summary.get("conditioned_minus_preconditioned", {})

    def df_value(row):
        value = row.get("df")
        return (
            int(value)
            if value is not None and np.isfinite(value)
            else "n/a"
        )

    return (
        "Preconditioned: "
        f"t({df_value(preconditioned)}) = "
        f"{float(preconditioned.get('t', np.nan)):.2f}, "
        f"p {_format_p(preconditioned.get('p_greater_than_zero'))}; "
        "Wilcoxon p "
        f"{_format_p(preconditioned.get('p_wilcoxon_greater_than_zero'))}\n"
        "Conditioned: "
        f"t({df_value(conditioned)}) = "
        f"{float(conditioned.get('t', np.nan)):.2f}, "
        f"p {_format_p(conditioned.get('p_greater_than_zero'))}; "
        "Wilcoxon p "
        f"{_format_p(conditioned.get('p_wilcoxon_greater_than_zero'))}\n"
        "Preconditioned > conditioned: "
        f"t({df_value(paired)}) = "
        f"{-float(paired.get('t', np.nan)):.2f}, "
        f"p {_format_p(paired.get('p_less_than_zero'))}; "
        "Wilcoxon p "
        f"{_format_p(paired.get('p_wilcoxon_less_than_zero'))} "
        "(one-sided)"
    )


def _plot(
    data: pd.DataFrame,
    group_summary: dict[str, dict],
    output_prefix: Path,
    formats: list[str],
    dpi: int,
    title: str,
) -> list[Path]:
    summary_stats = _summary_stats(data)
    wide = data.pivot_table(
        index="sub",
        columns="cue_subset",
        values=METRIC,
        aggfunc="first",
    )
    x_positions = np.arange(len(CUE_ORDER), dtype=float)
    rng = np.random.default_rng(20260728)
    subject_jitter = {
        sub: float(rng.uniform(-0.045, 0.045))
        for sub in sorted(wide.index)
    }

    fig = plt.figure(figsize=(4.9, 6.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.35], hspace=0.06)
    ax = fig.add_subplot(grid[0, 0])
    stats_ax = fig.add_subplot(grid[1, 0])

    means = [
        float(
            summary_stats.loc[
                summary_stats["cue_subset"] == cue,
                "mean",
            ].iloc[0]
        )
        for cue in CUE_ORDER
    ]
    sems = [
        float(
            summary_stats.loc[
                summary_stats["cue_subset"] == cue,
                "sem",
            ].iloc[0]
        )
        for cue in CUE_ORDER
    ]
    colors = [CUE_COLORS[cue] for cue in CUE_ORDER]

    ax.bar(
        x_positions,
        means,
        width=0.58,
        color=colors,
        alpha=0.20,
        edgecolor=colors,
        linewidth=1.4,
        zorder=1,
    )
    ax.errorbar(
        x_positions,
        means,
        yerr=sems,
        fmt="none",
        ecolor="black",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
        zorder=4,
    )
    ax.scatter(
        x_positions,
        means,
        marker="D",
        s=42,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=5,
    )

    for sub, row in wide.iterrows():
        jitter = subject_jitter[sub]
        xs = x_positions + jitter
        ys = [float(row[cue]) for cue in CUE_ORDER]
        ax.plot(
            xs,
            ys,
            color="0.72",
            linewidth=0.8,
            alpha=0.65,
            zorder=2,
        )
        for index, cue in enumerate(CUE_ORDER):
            ax.scatter(
                xs[index],
                ys[index],
                s=26,
                color=CUE_COLORS[cue],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=3,
            )

    ax.axhline(0, color="0.25", linewidth=0.9, zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [CUE_LABELS[cue] for cue in CUE_ORDER],
        fontsize=11,
    )
    ax.set_ylabel(
        "Spearman rho: neural vs odor-identity RDM",
        fontsize=10.5,
    )
    ax.set_title(title, fontsize=12, pad=10)

    y_values = data[METRIC].to_numpy(dtype=float)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    y_span = max(y_max - y_min, 0.08)
    ax.set_ylim(y_min - 0.16 * y_span, y_max + 0.32 * y_span)
    ax.set_xlim(-0.62, 1.62)

    stats_ax.axis("off")
    stats_ax.text(
        0.5,
        0.95,
        _annotation_text(group_summary),
        transform=stats_ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "linewidth": 0.8,
            "alpha": 0.95,
        },
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    stats_ax.set_xlim(0, 1)
    stats_ax.set_ylim(0, 1)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for output_format in formats:
        output_path = output_prefix.with_suffix(f".{output_format}")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        output_paths.append(output_path)
        print(f"Saved {output_path}")
    plt.close(fig)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired subject-level Spearman ROI RSA for one Kahnt OFC "
            "K=2 cluster."
        )
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
        help="Output path without an extension.",
    )
    parser.add_argument("--title", required=True)
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

    input_csv = args.input_csv.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve()
    if output_prefix.suffix:
        raise ValueError(
            "--output_prefix must not include a filename extension: "
            f"{output_prefix}"
        )

    data = _load_subject_data(input_csv, args.skip_subs)
    group_summary = _load_group_summary(summary_csv)
    _plot(
        data=data,
        group_summary=group_summary,
        output_prefix=output_prefix,
        formats=args.formats,
        dpi=args.dpi,
        title=args.title,
    )


if __name__ == "__main__":
    main()
