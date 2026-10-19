#!/usr/bin/env python3
"""Build the draft PNAS odor-identity convergence figure.

Panels
------
A. OFC K=2 ROIs and analysis logic.
B. Probe-only odor-identity Spearman RSA.
C. Passive-odor to premeal-cue cross-task decoding.
D. Intentionally blank for a future neural-behavior relationship.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(
        Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"
    )
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(
        Path(tempfile.gettempdir()) / "spcd_font_cache"
    )

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats


CONDITIONED = "conditioned_odor_only"
PRECONDITIONED = "preconditioned_odor_only"
CUE_ORDER = (CONDITIONED, PRECONDITIONED)
CUE_LABELS = ("Conditioned", "Preconditioned")
DEFAULT_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")

ROI_ORDER = (
    "OFC_K2_cluster1_medial_posterior_lateral",
    "OFC_K2_cluster2_central_anterior_lateral",
)
ROI_SHORT = {
    ROI_ORDER[0]: "Medial/posterior-\nlateral OFC",
    ROI_ORDER[1]: "Central/anterior-\nlateral OFC",
}
ROI_FILE_KEYS = {
    ROI_ORDER[0]: "cluster1_medial_posterior_lateral",
    ROI_ORDER[1]: "cluster2_central_anterior_lateral",
}
ROI_COLORS = {
    ROI_ORDER[0]: "#8C6BB1",
    ROI_ORDER[1]: "#238B8E",
}
CUE_COLORS = {
    CONDITIONED: "#3B6EA8",
    PRECONDITIONED: "#D9544D",
}

RSA_METRIC = "odor_identity_no_image_spearman_rho"
DECODING_METRIC = "cue_balanced_accuracy"


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _significance_label(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("Cannot label significance for a nonfinite p-value.")
    return "*" if value < 0.05 else "n.s."


def _load_subject_results(
    paths: list[Path],
    metric: str,
    skip_subs: tuple[str, ...],
) -> pd.DataFrame:
    frames = []
    required = {"sub", "roi", "cue_subset", metric}
    for path in paths:
        frame = pd.read_csv(path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[
        data["roi"].isin(ROI_ORDER)
        & data["cue_subset"].isin(CUE_ORDER)
        & ~data["sub"].isin(skip_subs)
    ].copy()
    if data.duplicated(["sub", "roi", "cue_subset"]).any():
        raise ValueError("Duplicate subject/ROI/cue-subset rows found.")
    expected = {
        (roi, cue_subset) for roi in ROI_ORDER for cue_subset in CUE_ORDER
    }
    observed = set(zip(data["roi"], data["cue_subset"]))
    if observed != expected:
        raise ValueError(f"Missing result cells: {sorted(expected - observed)}")
    if not np.isfinite(data[metric].to_numpy(dtype=float)).all():
        raise ValueError(f"{metric} contains nonfinite values.")
    return data


def _one_sample_tests(
    values: np.ndarray,
    null: float,
    *,
    round_for_ranks: bool,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    centered = values - null
    if round_for_ranks:
        # Accuracy is discrete. Rounding prevents machine-precision noise
        # from breaking true ties and changing the Wilcoxon rank statistic.
        centered_for_ranks = np.round(centered, decimals=12)
    else:
        centered_for_ranks = centered
    t_result = stats.ttest_1samp(centered, 0.0)
    t_stat = float(getattr(t_result, "statistic", t_result[0]))
    p_t = float(stats.t.sf(t_stat, len(centered) - 1))
    if np.allclose(centered_for_ranks, 0.0):
        p_wilcoxon = np.nan
    else:
        p_wilcoxon = float(
            stats.wilcoxon(
                centered_for_ranks,
                zero_method="wilcox",
                alternative="greater",
                # Match the saved RSA summaries' rank-test convention,
                # while retaining tie-aware automatic handling for accuracy.
                method="auto" if round_for_ranks else "exact",
            ).pvalue
        )
    return {
        "mean": float(np.mean(values)),
        "t": t_stat,
        "p_t": p_t,
        "p_wilcoxon": p_wilcoxon,
    }


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, np.nan, np.nan
    low, high = stats.t.interval(
        0.95,
        len(values) - 1,
        loc=mean,
        scale=stats.sem(values),
    )
    return mean, float(low), float(high)


def _swarm_offsets(values: np.ndarray, width: float = 0.16) -> np.ndarray:
    """Deterministically spread tied or nearby observations."""
    values = np.asarray(values, dtype=float)
    rounded = np.round(values, decimals=5)
    offsets = np.zeros(len(values), dtype=float)
    for value in np.unique(rounded):
        indices = np.flatnonzero(rounded == value)
        if len(indices) > 1:
            offsets[indices] = np.linspace(-width, width, len(indices))
    singles = np.flatnonzero(offsets == 0.0)
    if len(singles):
        sequence = np.linspace(-0.035, 0.035, len(singles))
        offsets[singles] = sequence
    return offsets


def _panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.105,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="top",
        clip_on=False,
    )


def _canonical_projection(path: Path) -> tuple[np.ndarray, list[float]]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(image.dataobj, dtype=float)
    projection = np.max(data, axis=1).T
    shape = np.asarray(image.shape[:3], dtype=float)
    corners = nib.affines.apply_affine(
        image.affine,
        np.array([[0, 0, 0], shape - 1]),
    )
    extent = [
        float(corners[0, 0]),
        float(corners[1, 0]),
        float(corners[0, 2]),
        float(corners[1, 2]),
    ]
    return projection, extent


def _draw_method_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#F6F6F6",
) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.8,
        edgecolor="#555555",
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )


def _draw_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#444444"},
    )


def _plot_analysis_logic(ax, project_root: Path) -> None:
    ax.set_axis_off()
    _panel_letter(ax, "A")

    gm_path = project_root / "masks" / "gm_0.1.nii"
    mask_dir = project_root / "masks" / "OFC_clusterMap_K2_gm0p1"
    gm_projection, extent = _canonical_projection(gm_path)
    roi_projections = []
    for roi in ROI_ORDER:
        mask_path = (
            mask_dir
            / f"mask-{roi}_gm0p1.nii.gz"
        )
        projection, roi_extent = _canonical_projection(mask_path)
        if not np.allclose(extent, roi_extent):
            raise ValueError("GM and OFC masks do not share one image space.")
        roi_projections.append((roi, projection))

    brain_ax = ax.inset_axes([0.02, 0.56, 0.43, 0.39])
    brain_ax.imshow(
        np.ma.masked_where(gm_projection <= 0, gm_projection > 0),
        cmap=matplotlib.colors.ListedColormap(["#D6D6D6"]),
        alpha=0.72,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
    )
    brain_ax.contour(
        (gm_projection > 0).astype(float),
        levels=[0.5],
        colors=["#AFAFAF"],
        linewidths=0.45,
        origin="lower",
        extent=extent,
    )
    for roi, projection in roi_projections:
        rgba = matplotlib.colors.to_rgba(ROI_COLORS[roi], 0.86)
        cmap = matplotlib.colors.ListedColormap([rgba])
        brain_ax.imshow(
            np.ma.masked_where(projection <= 0, projection),
            cmap=cmap,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            aspect="equal",
        )
    brain_ax.set_xlim(-76, 76)
    brain_ax.set_ylim(-65, 25)
    brain_ax.axis("off")
    brain_ax.text(
        0.5,
        0.02,
        "Coronal maximum projection",
        transform=brain_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    ax.text(
        0.48,
        0.92,
        "Independent OFC K=2 ROIs",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    for row, roi in enumerate(ROI_ORDER):
        y = 0.82 - row * 0.13
        ax.add_patch(
            patches.Rectangle(
                (0.49, y - 0.025),
                0.035,
                0.035,
                transform=ax.transAxes,
                color=ROI_COLORS[roi],
            )
        )
        ax.text(
            0.54,
            y,
            ROI_SHORT[roi].replace("\n", " "),
            transform=ax.transAxes,
            fontsize=10,
            va="center",
        )

    ax.text(
        0.02,
        0.49,
        "Within-probe RSA",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    _draw_method_box(ax, (0.02, 0.32), 0.22, 0.11, "Premeal cue\npatterns")
    _draw_method_box(ax, (0.34, 0.32), 0.20, 0.11, "Neural RDM")
    _draw_method_box(
        ax,
        (0.65, 0.32),
        0.29,
        0.11,
        "Odor-identity RDM\nsame = 0; different = 1",
    )
    _draw_arrow(ax, (0.245, 0.375), (0.335, 0.375))
    ax.text(
        0.595,
        0.375,
        "ρ",
        transform=ax.transAxes,
        fontsize=16,
        ha="center",
        va="center",
    )

    ax.text(
        0.02,
        0.24,
        "Cross-task decoding",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    _draw_method_box(
        ax,
        (0.02, 0.04),
        0.26,
        0.13,
        "Passive odor patterns\nodor 1 vs odor 2",
        facecolor="#F2F6FA",
    )
    _draw_method_box(ax, (0.39, 0.04), 0.18, 0.13, "Linear SVM")
    _draw_method_box(
        ax,
        (0.68, 0.04),
        0.26,
        0.13,
        "Premeal cue\nodor identity",
        facecolor="#FAF3F2",
    )
    _draw_arrow(ax, (0.285, 0.105), (0.385, 0.105))
    _draw_arrow(ax, (0.575, 0.105), (0.675, 0.105))
    ax.text(
        0.335,
        0.177,
        "train",
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        color="#555555",
    )
    ax.text(
        0.625,
        0.177,
        "test",
        transform=ax.transAxes,
        fontsize=9,
        ha="center",
        color="#555555",
    )


def _plot_paired_results(
    ax,
    data: pd.DataFrame,
    *,
    metric: str,
    baseline: float,
    title: str,
    ylabel: str,
    panel_letter: str,
    ylim: tuple[float, float],
    round_for_ranks: bool,
    baseline_label: str,
) -> None:
    _panel_letter(ax, panel_letter)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    positions = {
        (ROI_ORDER[0], CONDITIONED): 0.0,
        (ROI_ORDER[0], PRECONDITIONED): 0.9,
        (ROI_ORDER[1], CONDITIONED): 2.7,
        (ROI_ORDER[1], PRECONDITIONED): 3.6,
    }

    for roi in ROI_ORDER:
        subset = data.loc[data["roi"] == roi]
        wide = subset.pivot(index="sub", columns="cue_subset", values=metric)
        wide = wide.loc[:, list(CUE_ORDER)].dropna().sort_index()
        x_conditioned = positions[(roi, CONDITIONED)]
        x_preconditioned = positions[(roi, PRECONDITIONED)]
        for _, row in wide.iterrows():
            ax.plot(
                [x_conditioned, x_preconditioned],
                [row[CONDITIONED], row[PRECONDITIONED]],
                color="#D0D0D0",
                linewidth=0.45,
                alpha=0.40,
                zorder=1,
            )

        for cue_subset in CUE_ORDER:
            values = wide[cue_subset].to_numpy(dtype=float)
            x = positions[(roi, cue_subset)]
            ax.scatter(
                x + _swarm_offsets(values),
                values,
                s=14,
                color=CUE_COLORS[cue_subset],
                alpha=0.40,
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
            mean, low, high = _mean_ci(values)
            ax.errorbar(
                x,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="D",
                markersize=6.5,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=1.6,
                color="black",
                linewidth=2.2,
                capsize=5,
                capthick=2.2,
                zorder=3,
            )

            summary = _one_sample_tests(
                values,
                baseline,
                round_for_ranks=round_for_ranks,
            )
            annotation = _significance_label(summary["p_t"])
            ax.text(
                x,
                ylim[1] - 0.015 * (ylim[1] - ylim[0]),
                annotation,
                ha="center",
                va="top",
                fontsize=16 if annotation == "*" else 10,
                color="black",
            )

    ax.axhline(baseline, color="#333333", linewidth=0.9, linestyle="--")
    ax.text(
        3.98,
        baseline,
        baseline_label,
        fontsize=8.5,
        ha="right",
        va="bottom",
        color="#444444",
    )
    ax.set_xlim(-0.48, 4.08)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xticks(
        [positions[(roi, cue)] for roi in ROI_ORDER for cue in CUE_ORDER],
        CUE_LABELS * 2,
        rotation=24,
        ha="right",
        fontsize=9.5,
    )
    x_transform = ax.get_xaxis_transform()
    for roi in ROI_ORDER:
        center = np.mean([positions[(roi, cue)] for cue in CUE_ORDER])
        ax.text(
            center,
            -0.20,
            ROI_SHORT[roi],
            transform=x_transform,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=ROI_COLORS[roi],
            clip_on=False,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.8)


def _plot_blank_panel(ax) -> None:
    ax.set_axis_off()
    _panel_letter(ax, "D")


def _subject_paths(project_root: Path, analysis: str) -> list[Path]:
    if analysis == "rsa":
        root = (
            project_root
            / "rsa_roi"
            / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_ofc_k2_spearman"
        )
        return [
            root
            / ROI_FILE_KEYS[roi]
            / f"roi-{roi}_subject_spearman.csv"
            for roi in ROI_ORDER
        ]
    root = (
        project_root
        / "roi_decoding"
        / "probe_crosstask_odor2preconditioned"
    )
    return [root / f"roi-{roi}_subject_results.csv" for roi in ROI_ORDER]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the draft 2x2 PNAS odor-identity figure."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=_default_project_root(),
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        help=(
            "Output path without extension. Default: <project_root>/figures/"
            "odor_identity_fmri."
        ),
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    output_prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix is not None
        else project_root / "figures" / "odor_identity_fmri"
    )
    rsa_paths = _subject_paths(project_root, "rsa")
    decoding_paths = _subject_paths(project_root, "decoding")
    required_paths = [
        *rsa_paths,
        *decoding_paths,
        project_root / "masks" / "gm_0.1.nii",
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required figure inputs are missing:\n{formatted}")

    rsa_data = _load_subject_results(
        rsa_paths,
        RSA_METRIC,
        tuple(args.skip_subs),
    )
    decoding_data = _load_subject_results(
        decoding_paths,
        DECODING_METRIC,
        tuple(args.skip_subs),
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(13.4, 9.4), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.985,
        top=0.965,
        bottom=0.075,
        wspace=0.32,
        hspace=0.43,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    _plot_analysis_logic(ax_a, project_root)
    _plot_paired_results(
        ax_b,
        rsa_data,
        metric=RSA_METRIC,
        baseline=0.0,
        title="Odor identity within cue representations",
        ylabel="Spearman ρ: neural vs odor-identity RDM",
        panel_letter="B",
        ylim=(-0.215, 0.315),
        round_for_ranks=False,
        baseline_label="zero",
    )
    _plot_paired_results(
        ax_c,
        decoding_data,
        metric=DECODING_METRIC,
        baseline=0.5,
        title="Transfer from odor perception to cues",
        ylabel="Cross-task balanced accuracy",
        panel_letter="C",
        ylim=(0.18, 0.835),
        round_for_ranks=True,
        baseline_label="chance",
    )
    ax_c.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
    )
    _plot_blank_panel(ax_d)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in dict.fromkeys(args.formats):
        extension = extension.lstrip(".")
        output_path = output_prefix.with_suffix(f".{extension}")
        save_options = {"bbox_inches": "tight", "facecolor": "white"}
        if extension.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_options["dpi"] = args.dpi
        figure.savefig(output_path, **save_options)
        print(f"Saved: {output_path}")
    plt.close(figure)

    print("\nPrimary displayed statistics (one-sided):")
    for roi in ROI_ORDER:
        for cue_subset in CUE_ORDER:
            rsa_values = rsa_data.loc[
                (rsa_data["roi"] == roi)
                & (rsa_data["cue_subset"] == cue_subset),
                RSA_METRIC,
            ].to_numpy(dtype=float)
            decoding_values = decoding_data.loc[
                (decoding_data["roi"] == roi)
                & (decoding_data["cue_subset"] == cue_subset),
                DECODING_METRIC,
            ].to_numpy(dtype=float)
            rsa_summary = _one_sample_tests(
                rsa_values,
                0.0,
                round_for_ranks=False,
            )
            decoding_summary = _one_sample_tests(
                decoding_values,
                0.5,
                round_for_ranks=True,
            )
            print(
                f"  {roi} / {cue_subset}: "
                f"RSA {_significance_label(rsa_summary['p_t'])}, "
                f"n={len(rsa_values)}, "
                f"mean={rsa_summary['mean']:.6f}, "
                f"t p={rsa_summary['p_t']:.6f}, "
                f"W p={rsa_summary['p_wilcoxon']:.6f}; "
                f"decoding {_significance_label(decoding_summary['p_t'])}, "
                f"n={len(decoding_values)}, "
                f"mean={decoding_summary['mean']:.6f}, "
                f"t p={decoding_summary['p_t']:.6f}, "
                f"W p={decoding_summary['p_wilcoxon']:.6f}"
            )


if __name__ == "__main__":
    main()
