#!/usr/bin/env python3
"""Build the draft PNAS odor-identity convergence figure.

Panels
------
A. Independently defined OFC K=2 ROIs.
B. Cross-run RSA illustrated with an actual example neural RDM.
C. Probe-only odor-identity Spearman RSA results.
D. Passive-odor to premeal-cue decoding schematic.
E. Cross-task decoding results.

Figure 4 presents the empirical correlation analysis alongside the model prediction.
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path as MplPath
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figure_style import (
    LABEL_SIZE, MUTED_TEXT_COLOR, NOTE_SIZE, REFERENCE_COLOR,
    SPINE_COLOR, SUMMARY_COLOR, TITLE_SIZE, add_panel_header,
    configure_matplotlib, style_axis,
)


CONDITIONED = "conditioned_odor_only"
PRECONDITIONED = "preconditioned_odor_only"
CUE_ORDER = (CONDITIONED, PRECONDITIONED)
CUE_LABELS = ("Conditioned", "Preconditioned")
DEFAULT_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")
ANATOMY_RELATIVE_PATH = Path("masks/MNI152_T1_1mm_brain.nii.gz")
# Reuse the illustrative cue/outcome mapping shown in Figure 1.
DECODING_ASSETS = {
    "odor1": "fig/design_assets/sweet_odor.png",
    "odor2": "fig/design_assets/savory_odor.png",
    "A1": "cues/characters_gray/charmander.jpeg",
    "B1": "cues/places_gray/waterfall_small.jpg",
    "A2": "cues/characters_gray/stitch.jpg",
    "B2": "cues/places_gray/museum.jpg",
}

LATERAL_ROI = "OFC_K2_cluster2_central_anterior_lateral"
MEDIAL_ROI = "OFC_K2_cluster1_medial_posterior_lateral"
ROI_ORDER = (LATERAL_ROI, MEDIAL_ROI)
ROI_SHORT = {LATERAL_ROI: "Lateral OFC", MEDIAL_ROI: "Medial OFC"}
ROI_FILE_KEYS = {
    LATERAL_ROI: "cluster2_central_anterior_lateral",
    MEDIAL_ROI: "cluster1_medial_posterior_lateral",
}
ROI_COLORS = {LATERAL_ROI: "#238B8E", MEDIAL_ROI: "#8C6BB1"}
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
    """Assign reproducible, value-independent jitter shared across C and E."""
    subjects = sorted(set(subjects))
    offsets = (
        np.linspace(-width, width, len(subjects))
        if len(subjects) > 1 else np.zeros(len(subjects))
    )
    offsets = np.random.default_rng(20261022).permutation(offsets)
    return dict(zip(subjects, offsets))


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


def _plot_roi_anatomy(ax, project_root: Path) -> None:
    """Show two coronal views of the original masks with a compact key."""
    ax.set_axis_off()
    ax.set_title("Independently defined OFC regions", loc="left", pad=16)
    ax.text(-0.045, 1.06, "A", transform=ax.transAxes, fontsize=20,
            fontweight="bold", ha="left", va="bottom", clip_on=False)
    anatomy = nib.as_closest_canonical(nib.load(str(project_root / ANATOMY_RELATIVE_PATH)))
    anatomy_data = np.asanyarray(anatomy.dataobj, dtype=float)
    corners = nib.affines.apply_affine(
        anatomy.affine,
        np.array([[-0.5, -0.5, -0.5], np.asarray(anatomy.shape[:3]) - 0.5]),
    )
    extent = [corners[0, 0], corners[1, 0], corners[0, 2], corners[1, 2]]
    mask_dir = project_root / "masks" / "OFC_clusterMap_K2_gm0p1"
    roi_masks = {
        roi: resample_from_to(
            nib.load(str(mask_dir / f"mask-{roi}_gm0p1.nii.gz")), anatomy, order=0,
        ).get_fdata() > 0 for roi in ROI_ORDER
    }
    intensity_max = np.percentile(anatomy_data[anatomy_data > 0], 99)
    views, bounds = [], []
    for y_mm in (24, 40):
        y_index = int(round(nib.affines.apply_affine(
            np.linalg.inv(anatomy.affine), [0, y_mm, 0],
        )[1]))
        occupied = np.argwhere(anatomy_data[:, y_index, :] > 0)
        if not len(occupied):
            raise ValueError(f"Empty anatomical slice at y={y_mm} mm")
        low, high = occupied.min(axis=0), occupied.max(axis=0)
        bounds.extend(nib.affines.apply_affine(
            anatomy.affine, [[low[0], y_index, low[1]], [high[0], y_index, high[1]]],
        ))
        views.append((y_mm, y_index))
    bounds = np.asarray(bounds)
    # A common padded field of view includes the full brain in both slices.
    x_limits = (float(bounds[:, 0].max() + 4), float(bounds[:, 0].min() - 4))
    z_limits = (float(bounds[:, 2].min() - 4), float(bounds[:, 2].max() + 4))
    for column, (y_mm, y_index) in enumerate(views):
        brain_ax = ax.inset_axes([0.07 + column * 0.34, 0.0, 0.30, 0.91])
        brain_ax.set_facecolor("none")
        anatomical_slice = anatomy_data[:, y_index, :].T
        brain_ax.imshow(
            np.ma.masked_where(anatomical_slice <= 0, anatomical_slice),
            cmap="gray", vmin=0, vmax=intensity_max, origin="lower", extent=extent,
            interpolation="bilinear", aspect="equal",
        )
        for roi in ROI_ORDER:
            roi_slice = roi_masks[roi][:, y_index, :].T
            brain_ax.imshow(
                np.ma.masked_where(~roi_slice, roi_slice),
                cmap=matplotlib.colors.ListedColormap([ROI_COLORS[roi]]),
                alpha=0.85, origin="lower", extent=extent,
                interpolation="nearest", aspect="equal",
            )
        # Radiological convention: anatomical right is shown on the left.
        brain_ax.set_xlim(*x_limits)
        brain_ax.set_ylim(*z_limits)
        brain_ax.axis("off")
        brain_ax.set_title(f"y = +{y_mm} mm", fontsize=12, color=MUTED_TEXT_COLOR, pad=5)
        for x, side, alignment in ((0.02, "R", "left"), (0.98, "L", "right")):
            brain_ax.text(x, 0.5, side, transform=brain_ax.transAxes, fontsize=11,
                          fontweight="bold", color=MUTED_TEXT_COLOR, ha=alignment, va="center")
    for row, roi in enumerate(ROI_ORDER):
        y = 0.62 - row * 0.25
        ax.add_patch(patches.Rectangle((0.78, y - 0.025), 0.025, 0.065,
                                      transform=ax.transAxes, color=ROI_COLORS[roi]))
        ax.text(0.825, y, ROI_SHORT[roi], transform=ax.transAxes,
                fontsize=14, fontweight="bold", color=ROI_COLORS[roi], va="center")


def _example_rdms(
    project_root: Path, subject: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the first retained participant, without selecting on RSA strength."""
    root = (project_root / "rsa_roi" / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_ofc_k2_spearman" / ROI_FILE_KEYS[LATERAL_ROI]
            / "diagnostics" / subject)
    path = root / "preconditioned_odor_only_roi_patterns.npz"
    with np.load(path) as saved:
        patterns = saved["analysis_patterns"]
        runs = saved["run_ids"]
        conditions = saved["conditions"]
    odors = np.array([condition[1] for condition in conditions])
    if set(odors) != {"1", "2"} or len(np.unique(runs)) < 2:
        raise ValueError("Example RDM requires both odor identities and multiple runs")
    order = np.lexsort((conditions, odors, runs))
    neural = (1 - np.corrcoef(patterns))[np.ix_(order, order)]
    model = (odors[:, None] != odors[None, :]).astype(float)[np.ix_(order, order)]
    runs = runs[order]
    odors = odors[order]
    mask = runs[:, None] == runs[None, :]
    return np.ma.array(neural, mask=mask), np.ma.array(model, mask=mask), runs, odors


def _plot_rsa_logic(ax, project_root: Path, subject: str) -> None:
    ax.set_axis_off()
    add_panel_header(ax, "B", "Representational\nsimilarity analysis")
    neural, model, runs, odors = _example_rdms(project_root, subject)
    positions = ((0.0, neural, "Neural RDM"), (0.62, model, "Odor-identity RDM"))
    excluded_color = "#E5E7E9"
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(excluded_color)
    identity_colors = [cmap(0.0), cmap(1.0)]
    identity_cmap = ListedColormap(identity_colors)
    identity_cmap.set_bad(excluded_color)
    identity_norm = BoundaryNorm([-0.5, 0.5, 1.5], identity_cmap.N)
    run_groups = []
    odor_centers, odor_labels = [], []
    for run in np.unique(runs):
        indices = np.flatnonzero(runs == run)
        run_groups.append((indices[0], indices[-1], run))
        for odor in np.unique(odors[indices]):
            odor_indices = indices[odors[indices] == odor]
            odor_centers.append(float(odor_indices.mean()))
            odor_labels.append(f"O{odor}")
    for x, matrix, title in positions:
        child = ax.inset_axes([x, 0.26, 0.37, 0.53])
        if x == 0:
            low, high = float(matrix.min()), float(matrix.max())
            im = child.imshow(matrix, cmap=cmap, vmin=low, vmax=high,
                              interpolation="nearest")
        else:
            child.imshow(matrix, cmap=identity_cmap, norm=identity_norm,
                         interpolation="nearest")
        child.set_title(title, fontsize=12, y=1.28, pad=7)
        child.set_xticks(odor_centers, odor_labels)
        child.set_yticks(odor_centers, odor_labels)
        child.tick_params(axis="both", which="both", length=0, labelsize=8, pad=2)
        child.xaxis.tick_top()
        # Put the outer row labels away from the central correlation label.
        if x != 0:
            child.yaxis.tick_right()
        for start, end, run in run_groups:
            center = (start + end) / 2
            child.text(center, 1.16, f"Run {run}", transform=child.get_xaxis_transform(),
                       ha="center", va="bottom", fontsize=8.5)
            child.text(-0.25 if x == 0 else 1.25, center, f"Run {run}",
                       transform=child.get_yaxis_transform(), ha="center", va="center",
                       rotation=90 if x == 0 else -90, fontsize=8.5)
            if end < len(runs) - 1:
                child.axvline(end + 0.5, color="white", linewidth=0.65)
                child.axhline(end + 0.5, color="white", linewidth=0.65)
        for spine in child.spines.values():
            spine.set_visible(False)
        legend_ax = ax.inset_axes([x - 0.025, 0.015, 0.42, 0.24])
        legend_ax.set_axis_off()
        legend_ax.text(
            0.5, 0.82, "Neural dissimilarity" if x == 0 else "Odor dissimilarity",
            transform=legend_ax.transAxes, ha="center", va="center", fontsize=10,
        )
        if x == 0:
            bar_ax = legend_ax.inset_axes([0.13, 0.40, 0.74, 0.15])
            bar = ax.figure.colorbar(im, cax=bar_ax, orientation="horizontal")
            bar.set_ticks([low, high], labels=["Low", "High"])
            bar.ax.tick_params(labelsize=10, length=2, pad=3)
        else:
            for y, color, label in zip(
                (0.47, 0.18), identity_colors, ("Same odor (0)", "Different odors (1)"),
            ):
                legend_ax.add_patch(patches.Rectangle(
                    (0.06, y - 0.075), 0.075, 0.15,
                    transform=legend_ax.transAxes, facecolor=color, edgecolor="none",
                ))
                legend_ax.text(0.18, y, label, transform=legend_ax.transAxes,
                               va="center", fontsize=10)
    ax.text(0.495, 0.50, "Spearman\nρ", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="black")
    ax.text(0.49, -0.07, "Cross-run pairs only", transform=ax.transAxes,
            ha="center", fontsize=12, fontweight="bold")


def _decoding_asset(ax, project_root: Path, key: str, x: float, y: float,
                    width: float, height: float, *, alpha: float = 1.0) -> None:
    child = ax.inset_axes([x - width / 2, y - height / 2, width, height])
    child.imshow(plt.imread(project_root / DECODING_ASSETS[key]), cmap="gray", alpha=alpha)
    child.set_axis_off()


def _draw_smelling_icon(ax) -> None:
    """Draw a compact nose profile and scent lines as editable vector paths."""
    icon = ax.inset_axes([0.0, 0.85, 0.12, 0.16])
    icon.set_xlim(0, 1.1)
    icon.set_ylim(0, 1)
    icon.set_aspect("equal")
    icon.set_axis_off()
    nose = MplPath(
        [(0.28, 0.94),
         (0.30, 0.73), (0.39, 0.57), (0.57, 0.43),
         (0.69, 0.33), (0.61, 0.24), (0.47, 0.26),
         (0.40, 0.28), (0.37, 0.33), (0.31, 0.27),
         (0.26, 0.21), (0.22, 0.25), (0.22, 0.32),
         (0.22, 0.20), (0.25, 0.14), (0.20, 0.08)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 15,
    )
    icon.add_patch(patches.PathPatch(
        nose, fill=False, edgecolor=SUMMARY_COLOR, linewidth=1.6,
        capstyle="round", joinstyle="round",
    ))
    for y in (0.48, 0.72):
        scent = MplPath(
            [(0.72, y), (0.82, y + 0.09), (0.93, y - 0.07), (1.04, y + 0.02)],
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        icon.add_patch(patches.PathPatch(
            scent, fill=False, edgecolor=MUTED_TEXT_COLOR,
            linewidth=1.3, capstyle="round",
        ))


def _draw_viewing_icon(ax) -> None:
    """Match the smelling icon with an outline eye for cue viewing."""
    icon = ax.inset_axes([0.0, 0.405, 0.12, 0.13])
    icon.set_xlim(0, 1.1)
    icon.set_ylim(0, 1)
    icon.set_aspect("equal")
    icon.set_axis_off()
    outline = MplPath(
        [(0.10, 0.50), (0.32, 0.88), (0.78, 0.88), (1.00, 0.50),
         (0.78, 0.12), (0.32, 0.12), (0.10, 0.50)],
        [MplPath.MOVETO] + [MplPath.CURVE4] * 6,
    )
    icon.add_patch(patches.PathPatch(
        outline, fill=False, edgecolor=SUMMARY_COLOR, linewidth=1.6,
        capstyle="round", joinstyle="round",
    ))
    icon.add_patch(patches.Circle(
        (0.55, 0.50), 0.19, fill=False, edgecolor=SUMMARY_COLOR, linewidth=1.3,
    ))
    icon.add_patch(patches.Circle(
        (0.55, 0.50), 0.085, facecolor=SUMMARY_COLOR, edgecolor="none",
    ))


def _plot_decoding_logic(ax, project_root: Path) -> None:
    """Different task inputs share one schematic binary odor classifier."""
    ax.set_axis_off()
    add_panel_header(ax, "D", "Cross-task decoding")
    # Widen the schematic while keeping its heading aligned with panel B.
    ax = ax.inset_axes([0, 0, 1.15, 1])
    ax.set_axis_off()
    blue, orange = "#2878B5", "#D97721"
    for bottom, height in ((0.565, 0.465), (-0.15, 0.68)):
        ax.add_patch(patches.FancyBboxPatch(
            (-0.025, bottom), 0.56, height,
            boxstyle="round,pad=0,rounding_size=0.018",
            transform=ax.transAxes, facecolor="none", edgecolor=SPINE_COLOR,
            linewidth=0.9, clip_on=False,
        ))
    _draw_smelling_icon(ax)
    ax.text(0.14, 0.93, "Smelling odors", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="center")
    ax.text(0.805, 0.95, "Odor-identity\nSVM classifier", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="center")
    for center, key, label, color in (
        (0.105, "odor1", "Odor 1", blue), (0.37, "odor2", "Odor 2", orange),
    ):
        _decoding_asset(ax, project_root, key, center, 0.73, 0.19, 0.19)
        ax.text(center, 0.59, label, transform=ax.transAxes, fontsize=11,
                color=color, fontweight="bold", ha="center")
    _draw_arrow(ax, (0.535, 0.73), (0.61, 0.73))
    ax.text(0.58, 0.79, "Train", transform=ax.transAxes, fontsize=11,
            ha="center", color="black")

    _draw_viewing_icon(ax)
    ax.text(0.14, 0.465, "Viewing cues", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="center")
    for center, cues, key, color in (
        (0.105, ("A1", "B1"), "odor1", blue),
        (0.37, ("A2", "B2"), "odor2", orange),
    ):
        for offset, cue in zip((-0.055, 0.055), cues):
            _decoding_asset(ax, project_root, cue, center + offset, 0.31, 0.09, 0.15)
            ax.text(center + offset, 0.205, cue, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", ha="center")
        ax.annotate("", xy=(center, 0.085), xytext=(center, 0.165),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops={"arrowstyle": "-|>", "lw": 1.0,
                                "linestyle": (0, (3, 3)), "color": color})
        _decoding_asset(ax, project_root, key, center, 0.015, 0.14, 0.12, alpha=0.72)
        ax.add_patch(patches.FancyBboxPatch(
            (center - 0.085, -0.055), 0.17, 0.14,
            boxstyle="round,pad=0,rounding_size=0.012",
            transform=ax.transAxes, facecolor="none", edgecolor=color,
            linewidth=1.0, linestyle=(0, (3, 3)), clip_on=False,
        ))
    ax.text(0.24, -0.105, "Expected odors only", transform=ax.transAxes,
            ha="center", fontsize=11, color=MUTED_TEXT_COLOR)
    _draw_arrow(ax, (0.61, 0.31), (0.535, 0.31))
    ax.text(0.58, 0.37, "Test", transform=ax.transAxes, fontsize=11,
            ha="center", color="black")

    space = ax.inset_axes([0.63, 0.29, 0.35, 0.50])
    space.set_xlim(-2.3, 2.3)
    space.set_ylim(-1.8, 1.8)
    x = np.linspace(-2.3, 2.3, 200)
    space.fill_between(x, np.clip(0.63 * x, -1.8, 1.8), 1.8, color=blue, alpha=0.065, lw=0)
    space.fill_between(x, -1.8, np.clip(0.63 * x, -1.8, 1.8), color=orange, alpha=0.065, lw=0)
    odor1 = np.array([[-1.7, 0.5], [-1.4, 0.9], [-1.1, 1.25], [-0.8, 0.8], [-1.5, 1.25]])
    odor2 = np.array([[0.8, -0.9], [1.3, -0.6], [1.6, -1.2], [1.0, -1.4], [1.7, -0.5]])
    space.scatter(odor1[:, 0], odor1[:, 1], s=17, color=blue,
                  edgecolors="white", linewidths=0.4, zorder=3)
    space.scatter(odor2[:, 0], odor2[:, 1], s=17, marker="o", color=orange,
                  edgecolors="white", linewidths=0.4, zorder=3)
    space.plot(x, 0.63 * x, color=SUMMARY_COLOR, lw=1.3)
    space.scatter(-0.1, 0.48, s=46, marker="D", color=SUMMARY_COLOR,
                  edgecolors="white", linewidths=0.7, zorder=5)
    space.annotate("Cue", xy=(-0.1, 0.48), xytext=(-1.1, -0.2), fontsize=10,
                   ha="center", arrowprops={"arrowstyle": "->", "lw": 0.8,
                                           "color": SUMMARY_COLOR, "shrinkB": 5})
    space.set_xticks([])
    space.set_yticks([])
    for spine in space.spines.values():
        spine.set_visible(False)


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
    heading_lines: int = 1,
) -> None:
    ax.set_axis_off()
    add_panel_header(ax, panel_letter, title, title_lines=heading_lines)
    if heading_lines > 1:
        ax.set_title(title, loc="left", pad=16 + TITLE_SIZE * 1.2 * (heading_lines - 1))
    positions = {CONDITIONED: 0.0, PRECONDITIONED: 1.2}
    summaries_all = [
        _mean_ci(group[metric].to_numpy(dtype=float))
        for _, group in data.groupby(["roi", "cue_subset"])
    ]
    lower = min(baseline, data[metric].min(), *(ci[1] for ci in summaries_all))
    upper = max(baseline, data[metric].max(), *(ci[2] for ci in summaries_all))
    span = max(upper - lower, 0.01)
    limits = (lower - 0.12 * span, upper + 0.20 * span)

    for index, roi in enumerate(ROI_ORDER):
        child = ax.inset_axes([0.58 * index, 0.0, 0.42, 0.90])
        subset = data.loc[data["roi"] == roi]
        wide = subset.pivot(index="sub", columns="cue_subset", values=metric)
        wide = wide.loc[:, list(CUE_ORDER)].dropna().sort_index()
        summaries = {cue: _mean_ci(wide[cue].to_numpy(dtype=float)) for cue in CUE_ORDER}
        child.set_ylim(*limits)
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
        child.set_xlim(-0.48, 1.68)
        if index == 0:
            child.set_ylabel(ylabel, fontsize=LABEL_SIZE)
        child.set_xticks(
            list(positions.values()), CUE_LABELS,
            rotation=0, ha="center", fontsize=12,
        )
        child.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
        if metric == DECODING_METRIC:
            child.yaxis.set_major_formatter(
                matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0)
            )
        child.text(
            0.5,
            1.045,
            ROI_SHORT[roi],
            transform=child.transAxes,
            ha="center",
            va="bottom",
            fontsize=NOTE_SIZE,
            fontweight="bold",
            color=ROI_COLORS[roi],
            clip_on=False,
        )
        style_axis(child)
        child.tick_params(axis="x", labelsize=12)


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
        description="Create the five-panel odor-identity convergence figure."
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
            "fig3_fmri."
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
        else project_root / "fig" / "fig3_fmri"
    )
    rsa_paths = _subject_paths(project_root, "rsa")
    decoding_paths = _subject_paths(project_root, "decoding")
    required_paths = [
        *rsa_paths,
        *decoding_paths,
        project_root / ANATOMY_RELATIVE_PATH,
        *[project_root / asset for asset in DECODING_ASSETS.values()],
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
    figure = plt.figure(figsize=(13.4, 13.5), facecolor="white")
    # Give each row's labels clear separation from the next row's headings,
    # using a taller canvas to preserve the anatomy and result plot heights.
    grid = figure.add_gridspec(
        5, 2, left=0.08, right=0.98, top=0.94, bottom=0.13,
        height_ratios=(0.85, 0.35, 1, 0.42, 1), width_ratios=(1, 1.5),
        wspace=0.50, hspace=0,
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[2, 0])
    ax_c = figure.add_subplot(grid[2, 1])
    ax_d = figure.add_subplot(grid[4, 0])
    ax_e = figure.add_subplot(grid[4, 1])
    _plot_roi_anatomy(ax_a, project_root)
    example_subject = sorted(rsa_data["sub"].unique())[0]
    _plot_rsa_logic(ax_b, project_root, example_subject)
    _plot_paired_results(
        ax_c, rsa_data, subject_offsets=subject_offsets, metric=RSA_METRIC,
        baseline=0.0, title="Odor-identity RSA", ylabel="Spearman ρ: neural vs\nodor-identity RDM",
        panel_letter="C", round_for_ranks=False, heading_lines=2,
    )
    _plot_decoding_logic(ax_d, project_root)
    _plot_paired_results(
        ax_e, decoding_data, subject_offsets=subject_offsets, metric=DECODING_METRIC,
        baseline=0.5, title="Odor-identity decoding accuracy", ylabel="Perception-to-cue\ndecoding accuracy",
        panel_letter="E", round_for_ranks=True,
    )

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
