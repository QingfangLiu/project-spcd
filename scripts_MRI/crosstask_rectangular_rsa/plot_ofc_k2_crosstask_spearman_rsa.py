#!/usr/bin/env python3
"""Plot subject-level rectangular Spearman RSA in Kahnt K=2 OFC ROIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats


matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC = "cross_task_odor_identity_spearman_rho"
CUE_ORDER = (
    "conditioned_odor_only",
    "preconditioned_odor_only",
)
CUE_LABELS = ("Conditioned", "Preconditioned")
ROI_TITLES = {
    "OFC_K2_cluster1_medial_posterior_lateral": (
        "Medial/posterior-lateral OFC"
    ),
    "OFC_K2_cluster2_central_anterior_lateral": (
        "Central/anterior-lateral OFC"
    ),
}
COLORS = ("#4C78A8", "#E45745")
DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")


def _wilcoxon_greater(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size or np.allclose(values, 0.0):
        return np.nan
    try:
        result = stats.wilcoxon(
            values,
            alternative="greater",
            zero_method="wilcox",
        )
    except ValueError:
        return np.nan
    return float(getattr(result, "pvalue", result[1]))


def _one_sided_t_greater(values: np.ndarray) -> tuple[float, int, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, len(values) - 1, np.nan
    result = stats.ttest_1samp(values, 0.0)
    statistic = float(getattr(result, "statistic", result[0]))
    return (
        statistic,
        len(values) - 1,
        float(stats.t.sf(statistic, len(values) - 1)),
    )


def _load_scope(path: Path, skip_subs: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sub", "roi", "hemisphere", "cue_subset", METRIC}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[~frame["sub"].isin(skip_subs)].copy()
    if frame.empty:
        raise ValueError(f"No subjects remain after exclusions: {path}")
    if frame["roi"].nunique() != 1 or frame["hemisphere"].nunique() != 1:
        raise ValueError(f"Expected one ROI and hemisphere in: {path}")
    if frame.duplicated(["sub", "cue_subset"]).any():
        raise ValueError(f"Duplicate subject/cue rows in: {path}")
    return frame


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return "< .001"
    return f"= {value:.3f}".replace("0.", ".")


def _plot_panel(ax, frame: pd.DataFrame, title: str) -> None:
    wide = frame.pivot(index="sub", columns="cue_subset", values=METRIC)
    if not set(CUE_ORDER).issubset(wide.columns):
        raise ValueError(f"Both cue subsets are required for plotting {title}.")
    wide = wide.loc[:, list(CUE_ORDER)].dropna()
    if wide.empty:
        raise ValueError(f"No complete subject pairs are available for {title}.")

    x = np.arange(2, dtype=float)
    for _, row in wide.iterrows():
        values = row.to_numpy(dtype=float)
        ax.plot(x, values, color="#BDBDBD", linewidth=0.8, alpha=0.65, zorder=1)
    rng = np.random.default_rng(20261018)
    for index, cue_subset in enumerate(CUE_ORDER):
        values = wide[cue_subset].to_numpy(dtype=float)
        jitter = rng.uniform(-0.045, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=28,
            color=COLORS[index],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.95,
            zorder=2,
        )
        mean = float(np.mean(values))
        sem = float(stats.sem(values)) if len(values) > 1 else np.nan
        ax.errorbar(
            index,
            mean,
            yerr=sem,
            fmt="D",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            capsize=4,
            linewidth=1.4,
            zorder=3,
        )

    conditioned = wide[CUE_ORDER[0]].to_numpy(dtype=float)
    preconditioned = wide[CUE_ORDER[1]].to_numpy(dtype=float)
    pre_t, pre_df, pre_p = _one_sided_t_greater(preconditioned)
    cond_t, cond_df, cond_p = _one_sided_t_greater(conditioned)
    difference = preconditioned - conditioned
    diff_t, diff_df, diff_p = _one_sided_t_greater(difference)
    annotation = (
        f"Preconditioned: t({pre_df}) = {pre_t:.2f}, p {_format_p(pre_p)}; "
        f"Wilcoxon p {_format_p(_wilcoxon_greater(preconditioned))}\n"
        f"Conditioned: t({cond_df}) = {cond_t:.2f}, p {_format_p(cond_p)}; "
        f"Wilcoxon p {_format_p(_wilcoxon_greater(conditioned))}\n"
        f"Preconditioned > conditioned: t({diff_df}) = {diff_t:.2f}, "
        f"p {_format_p(diff_p)}; Wilcoxon p "
        f"{_format_p(_wilcoxon_greater(difference))}"
    )
    ax.text(
        0.5,
        -0.16,
        annotation,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#C7C7C7",
        },
    )
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, CUE_LABELS)
    ax.set_xlim(-0.55, 1.55)
    ax.set_title(title, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paired cross-task Spearman RSA coefficients."
    )
    parser.add_argument("--input_csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_prefix", type=Path, required=True)
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    args = parser.parse_args()

    frames = [
        _load_scope(path.expanduser().resolve(), args.skip_subs)
        for path in args.input_csvs
    ]
    hemispheres = {
        str(frame["hemisphere"].iloc[0]) for frame in frames
    }
    if len(hemispheres) != 1:
        raise ValueError("All plot inputs must use the same hemisphere scope.")
    frames.sort(key=lambda frame: str(frame["roi"].iloc[0]))

    figure, axes = plt.subplots(
        1,
        len(frames),
        figsize=(6.1 * len(frames), 5.6),
        squeeze=False,
        sharey=True,
    )
    for axis, frame in zip(axes[0], frames):
        roi = str(frame["roi"].iloc[0])
        _plot_panel(axis, frame, ROI_TITLES.get(roi, roi))
    axes[0, 0].set_ylabel(
        "Spearman rho: cross-task neural distance\nvs odor-identity distance"
    )
    hemisphere = next(iter(hemispheres))
    figure.suptitle(
        "Passive odor → premeal cue rectangular RSA in OFC "
        f"({hemisphere})",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(bottom=0.30, top=0.84, wspace=0.25)

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in dict.fromkeys(args.formats):
        extension = extension.lstrip(".")
        output_path = output_prefix.with_suffix(f".{extension}")
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
