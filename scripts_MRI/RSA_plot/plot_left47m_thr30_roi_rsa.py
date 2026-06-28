#!/usr/bin/env python3
"""Plot Left 47m threshold-30 ROI RSA subject betas.

The default inputs are the copied project-level ROI RSA outputs for the
no-image-first-image model variant:

``rsa_roi/probe_imagewise_odor_identity_no_image_first_image/premeal_thr30``.
"""

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


DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
DEFAULT_THRESHOLD = "30"
DEFAULT_ROI = "Left_47m"
RESULT_ROOT = (
    Path("rsa_roi")
    / "probe_imagewise_odor_identity_no_image_first_image"
)
BETA_COL = "odor_identity_no_image_beta"
CUE_ORDER = ("conditioned_odor_only", "preconditioned_odor_only")
CUE_LABELS = {
    "conditioned_odor_only": "Conditioned",
    "preconditioned_odor_only": "Preconditioned",
}
CUE_COLORS = {
    "conditioned_odor_only": "#3B6EA8",
    "preconditioned_odor_only": "#D84A3A",
}


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_result_dir(project_root: Path, threshold: str) -> Path:
    return project_root / RESULT_ROOT / f"premeal_thr{threshold}"


def _default_output_prefix(project_root: Path, roi: str, threshold: str) -> Path:
    return (
        project_root
        / "plots"
        / f"{roi}_thr{threshold}_no_image_first_image_roi_rsa_paired"
    )


def _format_p(p_value: float | int | None) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "n/a"
    p_value = float(p_value)
    if p_value < 0.001:
        return "< .001"
    return f"= {p_value:.3f}".replace("0.", ".")


def _load_subject_data(input_csv: Path, skip_subs: list[str]) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"sub", "cue_subset", BETA_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns {sorted(missing)}: {input_csv}")

    if skip_subs:
        df = df.loc[~df["sub"].isin(skip_subs)].copy()

    wide = df.pivot_table(
        index="sub",
        columns="cue_subset",
        values=BETA_COL,
        aggfunc="first",
    )
    missing_cues = [cue for cue in CUE_ORDER if cue not in wide.columns]
    if missing_cues:
        raise ValueError(f"Input CSV missing cue subsets {missing_cues}: {input_csv}")

    wide = wide.loc[wide[list(CUE_ORDER)].notna().all(axis=1), list(CUE_ORDER)]
    if wide.empty:
        raise ValueError("No complete subject pairs remain after exclusions")

    long_df = (
        wide.reset_index()
        .melt(id_vars="sub", var_name="cue_subset", value_name=BETA_COL)
        .sort_values(["sub", "cue_subset"])
        .reset_index(drop=True)
    )
    return long_df


def _summary_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cue in CUE_ORDER:
        values = data.loc[data["cue_subset"] == cue, BETA_COL].to_numpy(dtype=float)
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(
            {
                "cue_subset": cue,
                "mean": float(np.mean(values)),
                "sem": float(sd / np.sqrt(len(values))) if len(values) > 1 else np.nan,
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def _load_group_summary(summary_csv: Path) -> dict[str, dict]:
    if not summary_csv.exists():
        return {}
    summary = pd.read_csv(summary_csv)
    rows = {}
    for _, row in summary.iterrows():
        rows[str(row["contrast"])] = row.to_dict()
    return rows


def _annotation_text(group_summary: dict[str, dict], n_subjects: int) -> str:
    cond = group_summary.get("conditioned_odor_only_gt_zero", {})
    pre = group_summary.get("preconditioned_odor_only_gt_zero", {})
    paired = group_summary.get("conditioned_minus_preconditioned", {})

    cond_t = float(cond.get("t")) if cond.get("t") is not None else np.nan
    cond_df = int(cond.get("df")) if cond.get("df") is not None and np.isfinite(cond.get("df")) else "n/a"
    cond_p = _format_p(cond.get("p_greater_than_zero"))

    pre_t = float(pre.get("t")) if pre.get("t") is not None else np.nan
    pre_df = int(pre.get("df")) if pre.get("df") is not None and np.isfinite(pre.get("df")) else "n/a"
    pre_p = _format_p(pre.get("p_greater_than_zero"))

    pair_t = float(paired.get("t")) if paired.get("t") is not None else np.nan
    pair_df = int(paired.get("df")) if paired.get("df") is not None and np.isfinite(paired.get("df")) else "n/a"
    pair_p = _format_p(paired.get("p_less_than_zero"))

    return (
        f"Preconditioned: t({pre_df}) = {pre_t:.2f}, p {pre_p}, one-sided\n"
        f"Conditioned: t({cond_df}) = {cond_t:.2f}, p {cond_p}, one-sided\n"
        f"Preconditioned > conditioned: t({pair_df}) = {-pair_t:.2f}, p {pair_p}, one-sided"
    )


def _plot_paired_bar(
    data: pd.DataFrame,
    group_summary: dict[str, dict],
    output_prefix: Path,
    formats: list[str],
    dpi: int,
    title: str,
) -> list[Path]:
    stats = _summary_stats(data)
    wide = data.pivot_table(index="sub", columns="cue_subset", values=BETA_COL, aggfunc="first")
    x_positions = np.arange(len(CUE_ORDER), dtype=float)
    rng = np.random.default_rng(20260628)
    subject_jitter = {
        sub: float(rng.uniform(-0.045, 0.045))
        for sub in sorted(wide.index)
    }

    fig = plt.figure(figsize=(4.6, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[4.5, 1.2], hspace=0.06)
    ax = fig.add_subplot(grid[0, 0])
    stats_ax = fig.add_subplot(grid[1, 0])

    means = [float(stats.loc[stats["cue_subset"] == cue, "mean"].iloc[0]) for cue in CUE_ORDER]
    sems = [float(stats.loc[stats["cue_subset"] == cue, "sem"].iloc[0]) for cue in CUE_ORDER]
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
        ax.plot(xs, ys, color="0.72", linewidth=0.8, alpha=0.65, zorder=2)
        for idx, cue in enumerate(CUE_ORDER):
            ax.scatter(
                xs[idx],
                ys[idx],
                s=26,
                color=CUE_COLORS[cue],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.88,
                zorder=3,
            )

    ax.axhline(0, color="0.25", linewidth=0.9, linestyle="-", zorder=0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([CUE_LABELS[cue] for cue in CUE_ORDER], fontsize=11)
    ax.set_ylabel("Odor identity RSA coefficient", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    y_values = data[BETA_COL].to_numpy(dtype=float)
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    y_span = max(y_max - y_min, 0.04)
    ax.set_ylim(y_min - 0.16 * y_span, y_max + 0.32 * y_span)
    ax.set_xlim(-0.62, 1.62)

    annotation = _annotation_text(group_summary, wide.shape[0])
    stats_ax.axis("off")
    stats_ax.text(
        0.5,
        0.92,
        annotation,
        transform=stats_ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
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
    out_paths = []
    for fmt in formats:
        out_path = output_prefix.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        out_paths.append(out_path)
        print(f"Saved {out_path}")

    plt.close(fig)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paired subject-level Left 47m threshold-30 ROI RSA betas."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=_default_project_root(),
        help="Project root. Default: infer from this script location.",
    )
    parser.add_argument(
        "--threshold",
        default=DEFAULT_THRESHOLD,
        help="Threshold label. Default: 30.",
    )
    parser.add_argument(
        "--roi",
        default=DEFAULT_ROI,
        help="ROI label. Default: Left_47m.",
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=None,
        help="Subject betas CSV. Default: copied project roi result for ROI/threshold.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Group summary CSV. Default: copied project roi summary for ROI/threshold.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=None,
        help="Output path without extension. Default: plots/<roi>_thr<threshold>_...",
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
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs. Default: 300.",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    threshold = str(args.threshold)
    result_dir = _default_result_dir(project_root, threshold)
    input_csv = (
        args.input_csv.expanduser().resolve()
        if args.input_csv is not None
        else result_dir / f"roi-{args.roi}_subject_betas.csv"
    )
    summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv is not None
        else result_dir / f"roi-{args.roi}_group_summary.csv"
    )
    output_prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix is not None
        else _default_output_prefix(project_root, args.roi, threshold)
    )

    print(f"Input CSV:     {input_csv}")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Output prefix: {output_prefix}")
    print(f"Skip subjects: {', '.join(args.skip_subs) if args.skip_subs else 'none'}")

    data = _load_subject_data(input_csv, args.skip_subs)
    group_summary = _load_group_summary(summary_csv)
    title = "Odor identity coding in left lateral OFC"
    _plot_paired_bar(
        data=data,
        group_summary=group_summary,
        output_prefix=output_prefix,
        formats=args.formats,
        dpi=args.dpi,
        title=title,
    )


if __name__ == "__main__":
    main()
