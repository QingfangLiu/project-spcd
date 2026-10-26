#!/usr/bin/env python3
"""Render orthogonal views of the whole-brain odor-value RSA map.

Anatomy: SPM-normalized ICBM152NLin2009, copied from MRIcroGL's
standard/spm152.nii.gz into masks/. The independent vmPFC ROI selects
only the illustrated peak; the displayed map is not clipped to that ROI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_font_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy import ndimage, stats
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figure_style import configure_matplotlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / "temp_maps/spm_value"
    spm = loadmat(source / "SPM.mat", simplify_cells=True)["SPM"]
    df = float(spm["xX"]["erdf"])
    threshold = stats.t.isf(0.001, df)

    roi_image = nib.load(root / "masks/nifti_roi-masks_fig03/fig03D/C_POSvsNEG_kda10_clust_tstat1_bin.nii.gz")
    labels, _ = ndimage.label(np.asarray(roi_image.dataobj) > 0, structure=np.ones((3, 3, 3)))
    # A fixed anatomical seed selects the published vmPFC component, before
    # consulting the current statistic map; striatum and PCC remain excluded.
    seed = np.rint(nib.affines.apply_affine(np.linalg.inv(roi_image.affine), [0, 44, -6])).astype(int)
    component = labels[tuple(seed)]
    if component == 0:
        raise ValueError("The anatomical vmPFC seed is outside the supplied mask.")
    roi = labels == component
    if int(roi.sum()) != 1245:
        raise ValueError("Unexpected vmPFC component: expected 1,245 native voxels.")
    roi_image = nib.Nifti1Image(roi.astype(np.uint8), roi_image.affine)

    statistic_image = nib.as_closest_canonical(nib.load(source / "spmT_0001.nii"))
    model_image = nib.as_closest_canonical(nib.load(source / "mask.nii"))
    native_roi = np.asarray(resample_from_to(roi_image, statistic_image, order=0).dataobj) > 0
    native_roi &= np.asarray(model_image.dataobj) > 0
    statistic = np.asarray(statistic_image.dataobj)
    candidates = np.argwhere(native_roi & np.isfinite(statistic))
    peak_index = candidates[np.argmax(statistic[tuple(candidates.T)])]
    peak = nib.affines.apply_affine(statistic_image.affine, peak_index)
    peak_t = float(statistic[tuple(peak_index)])
    displayed = np.where((np.asarray(model_image.dataobj) > 0) & np.isfinite(statistic)
                         & (statistic > threshold), statistic, 0)
    display_max = float(displayed.max())
    anatomy = nib.as_closest_canonical(nib.load(root / "masks/MNI152NLin2009_SPM_T1_0p74mm.nii.gz"))
    anatomy_data = np.asarray(anatomy.dataobj)
    anatomy_peak = np.rint(nib.affines.apply_affine(np.linalg.inv(anatomy.affine), peak)).astype(int)
    edge_arrays = []
    for image in (anatomy, statistic_image):
        spacing = np.diag(image.affine)[:3]
        if not np.allclose(image.affine[:3, :3], np.diag(spacing)):
            raise ValueError("Slice rendering requires axis-aligned MNI references.")
        coords = [image.affine[i, 3] + spacing[i] * np.arange(image.shape[i]) for i in range(3)]
        edge_arrays.append([np.r_[c - spacing[i] / 2, c[-1] + spacing[i] / 2]
                            for i, c in enumerate(coords)])

    configure_matplotlib()
    fig = plt.figure(figsize=(10.2, 9.4), facecolor="black")
    grid = fig.add_gridspec(2, 2, left=0.025, right=0.975, bottom=0.025, top=0.91,
                            wspace=0.015, hspace=0.04)
    fig.suptitle("Odor-value representation", fontsize=18, fontweight="bold", color="white", y=0.985)
    norm = Normalize(vmin=threshold, vmax=display_max)
    cmap = LinearSegmentedColormap.from_list("value_activation", ["#FF7000", "#FFA000", "#FFFF35"])
    anatomy_max = np.percentile(anatomy_data[anatomy_data > 0], 99)
    views = [
        # fixed axis, horizontal axis, vertical axis, grid position, title
        (1, 0, 2, (0, 0), f"Coronal: y = {peak[1]:.0f} mm"),
        (0, 1, 2, (0, 1), f"Sagittal: x = {peak[0]:.0f} mm"),
        (2, 0, 1, (1, 0), f"Axial: z = {peak[2]:.0f} mm"),
    ]
    for fixed, horizontal, vertical, position, title in views:
        ax = fig.add_subplot(grid[position])
        brain = np.take(anatomy_data, anatomy_peak[fixed], axis=fixed).T
        values = np.take(displayed, peak_index[fixed], axis=fixed).T
        ae, se = edge_arrays
        extent = (ae[horizontal][0], ae[horizontal][-1], ae[vertical][0], ae[vertical][-1])
        ax.set_facecolor("black")
        ax.imshow(brain, origin="lower", extent=extent, cmap="gray", vmin=0,
                  vmax=anatomy_max, interpolation="bicubic")
        # Statistical cells retain their exact native 2 mm bounds as vectors.
        for iv, ih in np.argwhere(values > 0):
            xe, ye = se[horizontal], se[vertical]
            ax.fill([xe[ih], xe[ih + 1], xe[ih + 1], xe[ih]],
                    [ye[iv], ye[iv], ye[iv + 1], ye[iv + 1]],
                    color=cmap(norm(values[iv, ih])), linewidth=0, antialiased=False)
        ax.axvline(peak[horizontal], color="#B955BA", lw=0.6, alpha=0.7)
        ax.axhline(peak[vertical], color="#B955BA", lw=0.6, alpha=0.7)
        if fixed == 1:
            ax.set_xlim(-78, 78)
            ax.set_ylim(-60, 88)
        elif fixed == 0:
            ax.set_xlim(-110, 80)
            ax.set_ylim(-60, 88)
        else:
            ax.set_xlim(-78, 78)
            ax.set_ylim(-110, 80)
        ax.set_title(title, color="white", fontsize=13, fontweight="normal", loc="center", pad=7)
        left, right = ("P", "A") if fixed == 0 else ("L", "R")
        ax.text(0.025, 0.04, left, color="white", transform=ax.transAxes, fontsize=11)
        ax.text(0.975, 0.04, right, color="white", transform=ax.transAxes, ha="right", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    info = fig.add_subplot(grid[1, 1])
    info.set_facecolor("black")
    info.set_axis_off()
    info.text(0.5, 0.62, "Group-level RSA T map", ha="center", color="white", fontsize=14)
    info.text(0.5, 0.51, "Display: p < 0.001, uncorrected", ha="center", color="white", fontsize=12)
    info.text(0.5, 0.40, "Crosshairs: MNI [12, 48, 0]", ha="center", color="#D9ACDC", fontsize=12)
    color_axis = info.inset_axes([0.16, 0.24, 0.68, 0.045])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
                           orientation="horizontal", ticks=[threshold, display_max])
    colorbar.ax.set_xticklabels([f"{threshold:.2f}", f"{display_max:.2f}"])
    colorbar.ax.tick_params(labelsize=10, length=2, colors="white")
    colorbar.set_label("T value", fontsize=11, labelpad=4, color="white")
    colorbar.outline.set_edgecolor("white")

    output = root / "supp_fig/vmpfc_value_rsa"
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", pad_inches=0.12, facecolor="black")
    plt.close(fig)
    print(f"Peak: {peak.tolist()}; T({df:g}) = {peak_t:.8f}")
    print(f"Display threshold: T > {threshold:.8f}; p < 0.001 uncorrected")
    print(f"Whole-brain displayed voxels: {(displayed > 0).sum()}; SVC overlap: {native_roi.sum()}")
    print(f"Saved: {output}.pdf and {output}.png")


if __name__ == "__main__":
    main()
