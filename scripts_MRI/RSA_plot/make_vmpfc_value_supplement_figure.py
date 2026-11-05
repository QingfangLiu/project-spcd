#!/usr/bin/env python3
"""Render orthogonal views of the whole-brain odor-value RSA map.

Anatomy: SPM-normalized ICBM152NLin2009, copied from MRIcroGL's
standard/spm152.nii.gz into masks/. Independent vmPFC and medial OFC
masks select the illustrated peaks and provide outlines. Both rows show
the same whole-brain map, without clipping it to either ROI. The table
reports peak-level SVC within each of three ROIs, without a Holm column.
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
import pandas as pd
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
    statistic = np.asarray(statistic_image.dataobj)
    model_mask = np.asarray(model_image.dataobj) > 0
    summaries = pd.read_csv(root / "rsa_roi/probe_imagewise_value/"
                            "searchlight_svc_existing_group_v1/peak_svc_summary.csv")
    summaries = summaries.set_index("ROI").loc[["vmPFC", "Medial_OFC", "Lateral_OFC"]]
    medial_image = nib.load(root / "masks/OFC_clusterMap_K2_gm0p1/"
                           "mask-OFC_K2_cluster1_medial_posterior_lateral_gm0p1.nii.gz")
    roi_specs = [
        ("vmPFC", "vmPFC", roi_image, "#4A99D3"),
        ("Medial_OFC", "Medial OFC", medial_image, "#A58BC5"),
    ]
    peaks = []
    for key, label, image, color in roi_specs:
        mask = np.asarray(resample_from_to(image, statistic_image, order=0).dataobj) > 0
        mask &= model_mask
        candidates = np.argwhere(mask & np.isfinite(statistic))
        index = candidates[np.argmax(statistic[tuple(candidates.T)])]
        xyz = nib.affines.apply_affine(statistic_image.affine, index)
        peak_t = float(statistic[tuple(index)])
        row = summaries.loc[key]
        np.testing.assert_allclose(xyz, row[["MNI_x", "MNI_y", "MNI_z"]].to_numpy(float), atol=1e-8)
        np.testing.assert_allclose(peak_t, row.peak_T, atol=1e-8)
        assert int(mask.sum()) == int(row.analysis_overlap_voxels)
        peaks.append(dict(label=label, mask=mask, index=index, xyz=xyz, t=peak_t, color=color))
    displayed = np.where(model_mask & np.isfinite(statistic)
                         & (statistic > threshold), statistic, 0)
    display_max = float(displayed.max())
    anatomy = nib.as_closest_canonical(nib.load(root / "masks/MNI152NLin2009_SPM_T1_0p74mm.nii.gz"))
    anatomy_data = np.asarray(anatomy.dataobj)
    edge_arrays = []
    for image in (anatomy, statistic_image):
        spacing = np.diag(image.affine)[:3]
        if not np.allclose(image.affine[:3, :3], np.diag(spacing)):
            raise ValueError("Slice rendering requires axis-aligned MNI references.")
        coords = [image.affine[i, 3] + spacing[i] * np.arange(image.shape[i]) for i in range(3)]
        edge_arrays.append([np.r_[c - spacing[i] / 2, c[-1] + spacing[i] / 2]
                            for i, c in enumerate(coords)])

    configure_matplotlib()
    fig = plt.figure(figsize=(11.2, 8.8), facecolor="white")
    grid = fig.add_gridspec(3, 3, left=0.045, right=0.98, bottom=0.025, top=0.865,
                            height_ratios=[1, 1, 0.58], wspace=0.045, hspace=0.38)
    fig.suptitle("Searchlight RSA of subjective outcome value", fontsize=17,
                 fontweight="bold", y=0.99)
    fig.text(0.5, 0.948, r"Same group-level $t$ map in A and B; display: $p < 0.001$, uncorrected",
             ha="center", fontsize=11)
    norm = Normalize(vmin=threshold, vmax=display_max)
    cmap = LinearSegmentedColormap.from_list("value_activation", ["#FF7000", "#FFA000", "#FFFF35"])
    anatomy_max = np.percentile(anatomy_data[anatomy_data > 0], 99)
    for row_idx, peak in enumerate(peaks):
        xyz, index, color = peak["xyz"], peak["index"], peak["color"]
        anatomy_index = np.rint(nib.affines.apply_affine(np.linalg.inv(anatomy.affine), xyz)).astype(int)
        row_top = grid[row_idx, :].get_position(fig).y1
        fig.text(0.045, row_top + 0.040, f"{'AB'[row_idx]}  {peak['label']}",
                 fontsize=14, fontweight="bold", color=color, va="bottom")
        fig.text(0.98, row_top + 0.040, "Outline: independent ROI; crosshairs: regional peak",
                 fontsize=10, ha="right", va="bottom")
        views = [(1, 0, 2, f"Coronal: y = {xyz[1]:.0f} mm"),
                 (0, 1, 2, f"Sagittal: x = {xyz[0]:.0f} mm"),
                 (2, 0, 1, f"Axial: z = {xyz[2]:.0f} mm")]
        for col_idx, (fixed, horizontal, vertical, title) in enumerate(views):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            brain = np.take(anatomy_data, anatomy_index[fixed], axis=fixed).T
            values = np.take(displayed, index[fixed], axis=fixed).T
            outline = np.take(peak["mask"], index[fixed], axis=fixed).T
            ae, se = edge_arrays
            extent = (ae[horizontal][0], ae[horizontal][-1], ae[vertical][0], ae[vertical][-1])
            ax.set_facecolor("black")
            ax.imshow(brain, origin="lower", extent=extent, cmap="gray", vmin=0,
                      vmax=anatomy_max, interpolation="bicubic")
            # Statistical cells retain their exact native 2 mm bounds as vectors.
            xe, ye = se[horizontal], se[vertical]
            for iv, ih in np.argwhere(values > 0):
                ax.fill([xe[ih], xe[ih + 1], xe[ih + 1], xe[ih]],
                        [ye[iv], ye[iv], ye[iv + 1], ye[iv + 1]],
                        color=cmap(norm(values[iv, ih])), linewidth=0, antialiased=False)
            xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
            ax.contour(xc, yc, outline.astype(float), levels=[0.5], colors=[color], linewidths=0.8)
            ax.axvline(xyz[horizontal], color=color, lw=0.45, alpha=0.75)
            ax.axhline(xyz[vertical], color=color, lw=0.45, alpha=0.75)
            if fixed == 1:
                ax.set_xlim(-66, 66)
                ax.set_ylim(-36, 66)
            elif fixed == 0:
                ax.set_xlim(-54, 80)
                ax.set_ylim(-36, 66)
            else:
                ax.set_xlim(-66, 66)
                ax.set_ylim(-48, 80)
            ax.set_title(title, fontsize=11, fontweight="normal", loc="center", pad=5)
            left, right = ("P", "A") if fixed == 0 else ("L", "R")
            ax.text(0.025, 0.04, left, color="white", transform=ax.transAxes, fontsize=10)
            ax.text(0.975, 0.04, right, color="white", transform=ax.transAxes, ha="right", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    table_ax = fig.add_subplot(grid[2, :2])
    table_ax.set_axis_off()
    table_ax.text(0, 1.02, "C  Peak-level small-volume correction", fontsize=13,
                  fontweight="bold", transform=table_ax.transAxes, va="bottom")
    cells = []
    for label, (_, row) in zip(["vmPFC", "Medial OFC", "Lateral OFC"], summaries.iterrows()):
        xyz = ", ".join(f"{row[k]:.0f}" for k in ["MNI_x", "MNI_y", "MNI_z"])
        cells.append([label, f"[{xyz}]", f"{row.peak_T:.2f}", f"{row.peak_p_FWE_SVC:.3f}"])
    table = table_ax.table(cellText=cells,
                          colLabels=["ROI", "Peak MNI (mm)", rf"$t({df:.0f})$", "SVC-FWE p"],
                          colWidths=[0.24, 0.34, 0.16, 0.26],
                          cellLoc="center", bbox=[0, 0.08, 1, 0.80])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E1E5E8")
        cell.set_linewidth(0.5)
        cell.set_facecolor("#EDF0F3" if r == 0 else "white")
        if r == 0:
            cell.set_text_props(weight="bold")
    info = fig.add_subplot(grid[2, 2])
    info.set_axis_off()
    info.text(0.5, 0.88, r"Shared $t$ scale", ha="center", fontsize=11)
    color_axis = info.inset_axes([0.10, 0.50, 0.80, 0.12])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_axis,
                           orientation="horizontal", ticks=[threshold, display_max])
    colorbar.ax.set_xticklabels([f"{threshold:.2f}", f"{display_max:.2f}"])
    colorbar.ax.tick_params(labelsize=10, length=2)
    colorbar.outline.set_edgecolor("#7A858D")
    info.text(0.5, 0.04, "SVC p values corrected\nwithin each ROI", ha="center", va="bottom", fontsize=10)

    output = root / "supp_fig/vmpfc_value_rsa"
    for suffix in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(fig)
    for peak in peaks:
        print(f"{peak['label']} peak: {peak['xyz'].tolist()}; T({df:g}) = {peak['t']:.8f}")
    print(f"Display threshold: T > {threshold:.8f}; p < 0.001 uncorrected")
    print(f"Whole-brain displayed voxels: {(displayed > 0).sum()}")
    print(f"Saved: {output}.pdf and {output}.png")


if __name__ == "__main__":
    main()
