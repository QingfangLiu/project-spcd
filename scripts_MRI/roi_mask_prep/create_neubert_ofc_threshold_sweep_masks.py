#!/usr/bin/env python3
"""Create threshold-sweep Neubert OFC masks under the SPCD data tree.

This is a standalone mask-generation script for threshold sweeps beyond the
existing thr50/thr60 masks. It thresholds the resliced Neubert OFC probability
masks, intersects them with ``masks/gm_0.1.nii > 0``, and writes saved binary
masks that can be passed to ROI RSA with ``--roi_threshold 0``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_BASE_DIR = Path("/Users/liuq13/SPCD_data")
DEFAULT_THRESHOLDS = (10, 20, 30, 40, 70, 80, 90)
DEFAULT_SKIP_THRESHOLDS = (50, 60)
SOURCE_MASK_SUBDIR = Path("masks") / "neubert2015_ofc_resliced_gm0p1"
MASK_ROOT_SUBDIR = Path("rsa_roi") / "neubert2015_ofc_thresholded_masks"


@dataclass(frozen=True)
class SourceMask:
    roi: str
    path: Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _threshold_label(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return f"{threshold:g}".replace(".", "p")


def _source_mask_label(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    prefix = "gm0p1_"
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected source mask filename: {path}")
    return name[len(prefix) :]


def _same_space(img, ref_img, atol: float = 1e-5) -> bool:
    return img.shape[:3] == ref_img.shape[:3] and np.allclose(
        img.affine, ref_img.affine, atol=atol
    )


def _choose_existing_path(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    joined = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {description}. Checked:\n  {joined}")


def _discover_source_masks(source_mask_dir: Path, rois: set[str] | None) -> list[SourceMask]:
    paths = sorted(source_mask_dir.glob("gm0p1_*.nii*"))
    masks = []
    for path in paths:
        roi = _source_mask_label(path)
        if rois is not None and roi not in rois:
            continue
        masks.append(SourceMask(roi=roi, path=path.resolve()))

    if not masks:
        roi_text = "any requested ROI" if rois else "any ROI"
        raise FileNotFoundError(f"Did not find {roi_text} in {source_mask_dir}")
    return masks


def _make_thresholded_mask(roi_img, gm_img, threshold: float) -> np.ndarray:
    if not _same_space(roi_img, gm_img):
        raise ValueError("ROI mask and GM mask are not in the same voxel space")

    roi_data = np.asanyarray(roi_img.dataobj)
    gm_data = np.asanyarray(gm_img.dataobj)

    if threshold <= 0:
        roi_mask = np.isfinite(roi_data) & (roi_data > 0)
    else:
        roi_mask = np.isfinite(roi_data) & (roi_data >= threshold)

    gm_mask = np.isfinite(gm_data) & (gm_data > 0)
    return roi_mask & gm_mask


def _save_mask(mask: np.ndarray, roi_img, out_path: Path, roi: str, threshold: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = roi_img.header.copy()
    out_img = nib.Nifti1Image(mask.astype(np.uint8), roi_img.affine, header)
    out_img.set_data_dtype(np.uint8)
    out_img.header["descrip"] = f"Neubert OFC {roi}, ROI>={threshold:g}, gm_0.1>0"
    nib.save(out_img, str(out_path))


def _filtered_thresholds(thresholds: list[float], skip_thresholds: set[float]) -> list[float]:
    filtered = []
    skipped = []
    for threshold in thresholds:
        if threshold in skip_thresholds:
            skipped.append(threshold)
            continue
        filtered.append(threshold)

    if skipped:
        labels = ", ".join(_threshold_label(threshold) for threshold in skipped)
        print(f"Skipping thresholds by request/default: {labels}")
    if not filtered:
        raise ValueError("No thresholds left after applying --skip_thresholds")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create saved binary Neubert OFC masks for a threshold sweep. "
            "By default this writes under /Users/liuq13/SPCD_data."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="SPCD data root. Default: /Users/liuq13/SPCD_data.",
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=_default_project_root(),
        help="Project root used as a fallback source-mask location.",
    )
    parser.add_argument(
        "--source_mask_dir",
        type=Path,
        default=None,
        help=(
            "Folder with gm0p1_*.nii Neubert OFC masks. Default: "
            "<base_dir>/masks/neubert2015_ofc_resliced_gm0p1, falling back to "
            "<project_root>/masks/neubert2015_ofc_resliced_gm0p1."
        ),
    )
    parser.add_argument(
        "--gm_mask",
        type=Path,
        default=None,
        help=(
            "Gray-matter mask. Default: <base_dir>/masks/gm_0.1.nii, falling "
            "back to <project_root>/masks/gm_0.1.nii."
        ),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help=(
            "Output root. Default: "
            "<base_dir>/rsa_roi/neubert2015_ofc_thresholded_masks."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Probability thresholds to save. Default: 10 20 30 40 70 80 90.",
    )
    parser.add_argument(
        "--skip_thresholds",
        nargs="*",
        type=float,
        default=list(DEFAULT_SKIP_THRESHOLDS),
        help="Thresholds to skip even if requested. Default: 50 60.",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        default=None,
        help="Optional ROI labels, e.g. Left_47m. Default: all gm0p1_* masks.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load masks and report output paths/voxel counts without saving.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else base_dir / MASK_ROOT_SUBDIR
    )
    source_mask_dir = (
        args.source_mask_dir.expanduser().resolve()
        if args.source_mask_dir is not None
        else _choose_existing_path(
            [
                base_dir / SOURCE_MASK_SUBDIR,
                project_root / SOURCE_MASK_SUBDIR,
            ],
            "Neubert OFC source mask folder",
        )
    )
    gm_mask_path = (
        args.gm_mask.expanduser().resolve()
        if args.gm_mask is not None
        else _choose_existing_path(
            [
                base_dir / "masks" / "gm_0.1.nii",
                project_root / "masks" / "gm_0.1.nii",
            ],
            "gray-matter mask",
        )
    )
    thresholds = _filtered_thresholds(args.thresholds, set(args.skip_thresholds or []))
    rois = set(args.rois) if args.rois else None

    print(f"Base dir:        {base_dir}")
    print(f"Project root:    {project_root}")
    print(f"Source masks:    {source_mask_dir}")
    print(f"GM mask:         {gm_mask_path}")
    print(f"Output root:     {out_root}")
    print(f"Thresholds:      {', '.join(_threshold_label(t) for t in thresholds)}")
    print(f"ROIs:            {', '.join(sorted(rois)) if rois else 'all'}")
    if args.dry_run:
        print("Dry run:         true")

    gm_img = nib.load(str(gm_mask_path))
    source_masks = _discover_source_masks(source_mask_dir, rois)
    manifest_rows = []

    for threshold in thresholds:
        label = _threshold_label(threshold)
        out_dir = out_root / f"thr{label}"
        for source_mask in source_masks:
            roi_img = nib.load(str(source_mask.path))
            mask = _make_thresholded_mask(roi_img, gm_img, threshold)
            out_path = out_dir / f"mask-{source_mask.roi}_thr{label}_gm0p1.nii.gz"
            if not args.dry_run:
                _save_mask(mask, roi_img, out_path, source_mask.roi, threshold)

            n_voxels = int(mask.sum())
            manifest_rows.append(
                {
                    "roi": source_mask.roi,
                    "threshold": f"{threshold:g}",
                    "source_mask": str(source_mask.path),
                    "output_mask": str(out_path),
                    "n_voxels": n_voxels,
                    "gm_mask": str(gm_mask_path),
                }
            )
            action = "Would save" if args.dry_run else "Saved"
            print(f"{action} {out_path} | voxels={n_voxels}")

    manifest_path = out_root / "manifest_threshold_sweep.csv"
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "roi",
                    "threshold",
                    "source_mask",
                    "output_mask",
                    "n_voxels",
                    "gm_mask",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Saved manifest: {manifest_path}")
    else:
        print(f"Would save manifest: {manifest_path}")


if __name__ == "__main__":
    main()
