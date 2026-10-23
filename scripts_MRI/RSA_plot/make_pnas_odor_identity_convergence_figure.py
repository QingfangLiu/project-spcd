#!/usr/bin/env python3
"""Build the draft PNAS odor-identity convergence figure.

Panels
------
A. OFC K=2 ROIs and analysis logic.
B. Probe-only odor-identity Spearman RSA.
C. Passive-odor to premeal-cue cross-task decoding.
D. Conditioned/preconditioned RSA correlations within each OFC ROI.
"""

from __future__ import annotations

import argparse
import os
import sys
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
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figure_style import (
    LABEL_SIZE, MUTED_TEXT_COLOR, NOTE_SIZE, REFERENCE_COLOR,
    SPINE_COLOR, SUMMARY_COLOR, TICK_SIZE, add_panel_header,
    configure_matplotlib, style_axis,
)


CONDITIONED = "conditioned_odor_only"
PRECONDITIONED = "preconditioned_odor_only"
CUE_ORDER = (CONDITIONED, PRECONDITIONED)
CUE_LABELS = ("Conditioned", "Preconditioned")
DEFAULT_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")
ANATOMY_RELATIVE_PATH = Path("masks/MNI152_T1_1mm_brain.nii.gz")

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
INDIVIDUAL_COLOR = "#7F8992"

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


def _participant_offsets(
    subjects: list[str], width: float = 0.16,
) -> dict[str, float]:
    """Assign reproducible, value-independent jitter shared across B and C."""
    subjects = sorted(set(subjects))
    offsets = (
        np.linspace(-width, width, len(subjects))
        if len(subjects) > 1 else np.zeros(len(subjects))
    )
    offsets = np.random.default_rng(20261022).permutation(offsets)
    return dict(zip(subjects, offsets))


def _draw_method_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    fontsize: float = NOTE_SIZE,
) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.009,rounding_size=0.012",
        linewidth=0.8,
        edgecolor=SPINE_COLOR,
        facecolor=facecolor,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
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
        arrowprops={"arrowstyle": "-|>", "lw": 1.3, "color": MUTED_TEXT_COLOR},
    )


def _plot_analysis_logic(ax, project_root: Path) -> None:
    ax.set_axis_off()
    add_panel_header(ax, "A", "", title_lines=2)
    anatomy = nib.as_closest_canonical(
        nib.load(str(project_root / ANATOMY_RELATIVE_PATH))
    )
    anatomy_data = np.asanyarray(anatomy.dataobj, dtype=float)
    corners = nib.affines.apply_affine(
        anatomy.affine,
        np.array([[-0.5, -0.5, -0.5], np.asarray(anatomy.shape[:3]) - 0.5]),
    )
    extent = [corners[0, 0], corners[1, 0], corners[0, 2], corners[1, 2]]
    mask_dir = project_root / "masks" / "OFC_clusterMap_K2_gm0p1"
    roi_masks = {}
    for roi in ROI_ORDER:
        mask = nib.load(str(mask_dir / f"mask-{roi}_gm0p1.nii.gz"))
        # Preserve discrete ROI labels when matching the anatomical image grid.
        roi_masks[roi] = resample_from_to(mask, anatomy, order=0).get_fdata() > 0

    intensity_max = np.percentile(anatomy_data[anatomy_data > 0], 99)
    for column, (y_mm, label) in enumerate(((24, "Posterior"), (40, "Anterior"))):
        y_index = int(round(nib.affines.apply_affine(
            np.linalg.inv(anatomy.affine), [0, y_mm, 0],
        )[1]))
        brain_ax = ax.inset_axes([column * 0.53, 0.755, 0.47, 0.30])
        brain_ax.set_facecolor("none")
        anatomical_slice = anatomy_data[:, y_index, :].T
        brain_ax.imshow(
            np.ma.masked_where(anatomical_slice <= 0, anatomical_slice),
            cmap="gray",
            vmin=0,
            vmax=intensity_max,
            origin="lower",
            extent=extent,
            interpolation="bilinear",
            aspect="equal",
        )
        for roi in ROI_ORDER:
            roi_slice = roi_masks[roi][:, y_index, :].T
            brain_ax.imshow(
                np.ma.masked_where(~roi_slice, roi_slice),
                cmap=matplotlib.colors.ListedColormap([ROI_COLORS[roi]]),
                alpha=0.85,
                origin="lower",
                extent=extent,
                interpolation="nearest",
                aspect="equal",
            )
        # Radiological convention: positive MNI x (right) appears on the left.
        brain_ax.set_xlim(74, -74)
        brain_ax.set_ylim(-34, 70)
        brain_ax.axis("off")
        brain_ax.set_title(
            f"{label} · y = +{y_mm} mm", fontsize=10,
            color=MUTED_TEXT_COLOR, pad=5,
        )
        for x, side, alignment in ((0.02, "R", "left"), (0.98, "L", "right")):
            brain_ax.text(
                x, 0.5, side, transform=brain_ax.transAxes, fontsize=10,
                fontweight="bold", color=MUTED_TEXT_COLOR,
                ha=alignment, va="center",
            )

    ax.text(
        0.50,
        1.13,
        "Anatomical OFC ROIs",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False,
    )
    for row, roi in enumerate(ROI_ORDER):
        y = 0.70 - row * 0.06
        ax.add_patch(
            patches.Rectangle(
                (0.14, y - 0.013),
                0.028,
                0.028,
                transform=ax.transAxes,
                color=ROI_COLORS[roi],
            )
        )
        ax.text(
            0.19,
            y,
            ROI_SHORT[roi].replace("\n", ""),
            transform=ax.transAxes,
            fontsize=12,
            va="center",
        )
    ax.text(
        0.00,
        0.52,
        "Within-probe RSA",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.42, 0.52, "(B,D)", transform=ax.transAxes,
        fontsize=12,
    )
    _draw_method_box(ax, (0.00, 0.35), 0.26, 0.12, "Premeal cue\npatterns")
    _draw_method_box(ax, (0.38, 0.35), 0.24, 0.12, "Neural RDM")
    _draw_method_box(
        ax,
        (0.74, 0.35),
        0.26,
        0.12,
        "Odor-identity\nRDM",
    )
    _draw_arrow(ax, (0.27, 0.41), (0.37, 0.41))
    ax.text(
        0.68,
        0.41,
        "ρ",
        transform=ax.transAxes,
        fontsize=20,
        ha="center",
        va="center",
    )
    ax.text(
        1.00,
        0.325,
        "same = 0; different = 1",
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
        va="top",
        color=MUTED_TEXT_COLOR,
    )

    ax.text(
        0.00,
        0.21,
        "Perception-to-cue decoding",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.66, 0.21, "(C)", transform=ax.transAxes,
        fontsize=12,
    )
    _draw_method_box(
        ax,
        (0.00, 0.015),
        0.26,
        0.12,
        "Odor-evoked\npatterns",
    )
    _draw_method_box(ax, (0.38, 0.015), 0.24, 0.12, "Linear SVM")
    _draw_method_box(
        ax,
        (0.74, 0.015),
        0.26,
        0.12,
        "Cue-predicted\nodor identity",
    )
    _draw_arrow(ax, (0.27, 0.075), (0.37, 0.075))
    _draw_arrow(ax, (0.63, 0.075), (0.73, 0.075))
    ax.text(
        0.32,
        0.145,
        "train",
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        color=MUTED_TEXT_COLOR,
    )
    ax.text(
        0.68,
        0.145,
        "test",
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        color=MUTED_TEXT_COLOR,
    )
    ax.text(
        0.13,
        -0.025,
        "odor 1 vs odor 2",
        transform=ax.transAxes,
        fontsize=12,
        ha="center",
        va="top",
        color=MUTED_TEXT_COLOR,
    )


def _plot_paired_results(
    ax,
    data: pd.DataFrame,
    *,
    subject_offsets: dict[str, float],
    metric: str,
    baseline: float,
    title: str,
    ylabel: str,
    panel_letter: str,
    round_for_ranks: bool,
    baseline_label: str,
) -> None:
    ax.set_axis_off()
    add_panel_header(ax, panel_letter, title)
    positions = {CONDITIONED: 0.0, PRECONDITIONED: 0.9}

    for index, roi in enumerate(ROI_ORDER):
        child = ax.inset_axes([0.56 * index, 0.0, 0.44, 1.0])
        subset = data.loc[data["roi"] == roi]
        wide = subset.pivot(index="sub", columns="cue_subset", values=metric)
        wide = wide.loc[:, list(CUE_ORDER)].dropna().sort_index()
        summaries = {cue: _mean_ci(wide[cue].to_numpy(dtype=float)) for cue in CUE_ORDER}
        lower = min(baseline, wide.to_numpy().min(), *(ci[1] for ci in summaries.values()))
        upper = max(baseline, wide.to_numpy().max(), *(ci[2] for ci in summaries.values()))
        span = max(upper - lower, 0.01)
        child.set_ylim(lower - 0.12 * span, upper + 0.20 * span)
        x_conditioned = positions[CONDITIONED]
        x_preconditioned = positions[PRECONDITIONED]
        jitter = np.array([subject_offsets[sub] for sub in wide.index])
        for sub, row in wide.iterrows():
            offset = subject_offsets[sub]
            child.plot(
                [x_conditioned + offset, x_preconditioned + offset],
                [row[CONDITIONED], row[PRECONDITIONED]],
                color=SPINE_COLOR,
                linewidth=0.6,
                alpha=0.40,
                zorder=1,
            )

        for cue_subset in CUE_ORDER:
            values = wide[cue_subset].to_numpy(dtype=float)
            x = positions[cue_subset]
            child.scatter(
                x + jitter,
                values,
                s=28,
                color=INDIVIDUAL_COLOR,
                alpha=0.40,
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
            mean, low, high = summaries[cue_subset]
            child.errorbar(
                x,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="D",
                markersize=6.5,
                markerfacecolor="white",
                markeredgecolor=SUMMARY_COLOR,
                markeredgewidth=1.6,
                color=SUMMARY_COLOR,
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
            child.text(
                x,
                0.985,
                annotation,
                transform=child.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=20 if annotation == "*" else NOTE_SIZE,
                color=SUMMARY_COLOR,
            )

        child.axhline(baseline, color=REFERENCE_COLOR, linewidth=1.2, linestyle=(0, (3, 3)))
        child.annotate(
            baseline_label,
            xy=(0.98, baseline),
            xycoords=child.get_yaxis_transform(),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=10,
            ha="right",
            va="bottom",
            color=MUTED_TEXT_COLOR,
        )
        child.set_xlim(-0.48, 1.60)
        if index == 0:
            child.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        child.set_xticks(
            list(positions.values()), CUE_LABELS,
            rotation=24, ha="right", fontsize=TICK_SIZE,
        )
        child.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        if metric == DECODING_METRIC:
            child.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
            )
        child.text(
            0.5,
            -0.32,
            ROI_SHORT[roi],
            transform=child.transAxes,
            ha="center",
            va="top",
            fontsize=NOTE_SIZE,
            fontweight="bold",
            color=ROI_COLORS[roi],
            clip_on=False,
        )
        style_axis(child)


def _plot_rsa_correlations(ax, data: pd.DataFrame) -> list[dict]:
    """Show negative-direction Spearman tests and descriptive linear fits.

    P-values are one-sided and uncorrected; * denotes p < 0.05, otherwise n.s.
    Shading is the pointwise 95% confidence interval for the OLS mean response,
    not for Spearman rho.
    """
    ax.set_axis_off()
    add_panel_header(
        ax, "D", "Competition between direct and\ninferred odor-identity representations",
    )
    paired = []
    for roi in ROI_ORDER:
        wide = data.loc[data["roi"] == roi].pivot(
            index="sub", columns="cue_subset", values=RSA_METRIC,
        )
        wide = wide.loc[:, list(CUE_ORDER)].sort_index()
        if wide.isna().any().any():
            raise ValueError(f"Incomplete conditioned/preconditioned pairs: {roi}")
        if len(wide) < 3:
            raise ValueError(f"At least three paired subjects are required: {roi}")
        paired.append(wide)
    if not paired[0].index.equals(paired[1].index):
        raise ValueError("Panel D requires matching subjects across the two ROIs.")

    all_values = np.concatenate([wide.to_numpy().ravel() for wide in paired])
    padding = 0.12 * np.ptp(all_values)
    limits = (float(all_values.min() - padding), float(all_values.max() + padding))
    summaries = []
    for index, (roi, wide) in enumerate(zip(ROI_ORDER, paired)):
        # Fill the parent plot height so D aligns with C in the lower row.
        child = ax.inset_axes([0.56 * index, 0.0, 0.44, 1.0])
        x = wide[CONDITIONED].to_numpy(dtype=float)
        y = wide[PRECONDITIONED].to_numpy(dtype=float)
        result = stats.spearmanr(x, y, alternative="less")
        fit = stats.linregress(x, y)
        fit_x = np.linspace(x.min(), x.max(), 250)
        fit_y = fit.intercept + fit.slope * fit_x
        residuals = y - (fit.intercept + fit.slope * x)
        residual_sd = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        mean_se = residual_sd * np.sqrt(
            1 / len(x)
            + (fit_x - x.mean())**2 / np.sum((x - x.mean())**2)
        )
        half_width = stats.t.ppf(0.975, len(x) - 2) * mean_se
        color = ROI_COLORS[roi]
        child.fill_between(
            fit_x, fit_y - half_width, fit_y + half_width,
            color=color, alpha=0.18, linewidth=0, zorder=1,
        )
        child.plot(fit_x, fit_y, color=color, linewidth=1.6, zorder=2)
        child.scatter(
            x, y, s=28, color=color, alpha=0.85, edgecolor="white",
            linewidth=0.4, zorder=3,
        )
        child.axhline(0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--")
        child.axvline(0, color=REFERENCE_COLOR, linewidth=0.8, linestyle="--")
        child.set_xlim(*limits)
        child.set_ylim(*limits)
        child.set_xlabel(
            "Direct odor-identity\ncoding (RSA ρ)", fontsize=LABEL_SIZE,
        )
        if index == 0:
            child.set_ylabel(
                "Inferred odor-identity\ncoding (RSA ρ)", fontsize=LABEL_SIZE,
            )
        child.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        child.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        style_axis(child)
        child.text(
            0.5, -0.32, ROI_SHORT[roi], transform=child.transAxes,
            ha="center", va="top", fontsize=NOTE_SIZE,
            fontweight="bold", color=color, clip_on=False,
        )
        significance = _significance_label(float(result.pvalue))
        child.text(
            0.72, 0.97, f"ρ = {result.statistic:.3f}",
            transform=child.transAxes, ha="center", va="top", fontsize=NOTE_SIZE,
        )
        child.text(
            0.72, 0.89, significance,
            transform=child.transAxes, ha="center", va="top",
            fontsize=20 if significance == "*" else NOTE_SIZE,
            color=SUMMARY_COLOR,
        )
        summaries.append({
            "roi": roi, "n": len(x), "rho": float(result.statistic),
            "p_spearman": float(result.pvalue),
        })
    return summaries


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
            "Output path without extension. Default: <project_root>/fig/"
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
        else project_root / "fig" / "odor_identity_fmri"
    )
    rsa_paths = _subject_paths(project_root, "rsa")
    decoding_paths = _subject_paths(project_root, "decoding")
    required_paths = [
        *rsa_paths,
        *decoding_paths,
        project_root / ANATOMY_RELATIVE_PATH,
        *[
            project_root / "masks" / "OFC_clusterMap_K2_gm0p1"
            / f"mask-{roi}_gm0p1.nii.gz"
            for roi in ROI_ORDER
        ],
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
    subject_offsets = _participant_offsets(
        [*rsa_data["sub"], *decoding_data["sub"]],
    )

    configure_matplotlib()
    figure = plt.figure(figsize=(13.4, 11.6), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.095,
        right=0.985,
        top=0.90,
        bottom=0.155,
        wspace=0.43,
        hspace=0.72,
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    # Use the open inter-row space for a readable diagram, keeping its header
    # aligned with panel B and all result panels in their existing positions.
    position = ax_a.get_position()
    ax_a.set_position([
        position.x0, position.y0 - 0.08, position.width, position.height + 0.08,
    ])
    _plot_analysis_logic(ax_a, project_root)
    _plot_paired_results(
        ax_b,
        rsa_data,
        subject_offsets=subject_offsets,
        metric=RSA_METRIC,
        baseline=0.0,
        title="Representation of\npredicted odor identity",
        ylabel="Spearman ρ: neural vs\nodor-identity RDM",
        panel_letter="B",
        round_for_ranks=False,
        baseline_label="zero",
    )
    _plot_paired_results(
        ax_c,
        decoding_data,
        subject_offsets=subject_offsets,
        metric=DECODING_METRIC,
        baseline=0.5,
        title="Sensory reinstatement of\npredicted odor identity",
        ylabel="Perception-to-cue\ndecoding accuracy",
        panel_letter="C",
        round_for_ranks=True,
        baseline_label="chance",
    )
    correlation_summaries = _plot_rsa_correlations(ax_d, rsa_data)

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

    print("\nPanel D Spearman correlations (one-sided negative, uncorrected):")
    for summary in correlation_summaries:
        print(
            f"  {summary['roi']}: n={summary['n']}, "
            f"rho={summary['rho']:.6f}, p={summary['p_spearman']:.6f}"
        )

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
