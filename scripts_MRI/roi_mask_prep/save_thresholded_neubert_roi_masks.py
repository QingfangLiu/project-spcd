#!/usr/bin/env python3
"""Save binary Neubert OFC ROI masks used by thresholded ROI RSA.

The ROI RSA scripts threshold probabilistic ROI masks in memory using
``roi_data >= roi_threshold`` for positive thresholds, then intersect the result
with ``masks/gm_0.1.nii > 0``. This script recreates those binary masks from the
saved ROI result CSV metadata and writes one shared copy per ROI/threshold.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_RESULT_DIRS = (
    "rsa_roi/probe_imagewise_odor_identity/premeal_thr50",
    "rsa_roi/probe_imagewise_odor_identity/premeal_thr60",
    "rsa_roi/probe_imagewise_odor_identity_no_image/premeal_thr50",
    "rsa_roi/probe_imagewise_odor_identity_no_image/premeal_thr60",
)


@dataclass(frozen=True)
class RoiMaskSpec:
    roi: str
    threshold: float
    source_mask: Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _threshold_label(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def _resolve_mask_path(mask_path_from_csv: str, project_root: Path) -> Path:
    path = Path(mask_path_from_csv).expanduser()
    if path.exists():
        return path.resolve()

    parts = path.parts
    if "masks" in parts:
        rel_path = Path(*parts[parts.index("masks") :])
        candidate = project_root / rel_path
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not find ROI mask from CSV path: {mask_path_from_csv}")


def _same_space(img, ref_img, atol: float = 1e-5) -> bool:
    return img.shape[:3] == ref_img.shape[:3] and np.allclose(
        img.affine, ref_img.affine, atol=atol
    )


def _read_specs(result_dirs: list[Path], project_root: Path) -> dict[tuple[str, float], RoiMaskSpec]:
    specs: dict[tuple[str, float], RoiMaskSpec] = {}
    for result_dir in result_dirs:
        if not result_dir.is_dir():
            print(f"Skipping missing result folder: {result_dir}")
            continue

        for csv_path in sorted(result_dir.glob("roi-*_subject_betas.csv")):
            with csv_path.open(newline="") as f:
                first_row = next(csv.DictReader(f))

            roi = first_row["roi"]
            threshold = float(first_row["roi_threshold"])
            source_mask = _resolve_mask_path(first_row["roi_mask"], project_root)
            key = (roi, threshold)
            spec = RoiMaskSpec(roi=roi, threshold=threshold, source_mask=source_mask)

            previous = specs.get(key)
            if previous is not None and previous.source_mask != source_mask:
                raise ValueError(
                    "Conflicting source masks for the same ROI/threshold:\n"
                    f"  {previous}\n"
                    f"  {spec}"
                )
            specs[key] = spec

    if not specs:
        raise FileNotFoundError("No ROI subject_betas CSVs found in the requested result folders")

    return specs


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create saved binary copies of thresholded Neubert OFC ROI masks."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=_default_project_root(),
        help="Project root. Default: infer from this script location.",
    )
    parser.add_argument(
        "--gm_mask",
        type=Path,
        default=None,
        help="GM mask path. Default: <project_root>/masks/gm_0.1.nii",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=None,
        help=(
            "Output root. Default: "
            "<project_root>/rsa_roi/neubert2015_ofc_thresholded_masks"
        ),
    )
    parser.add_argument(
        "--result_dirs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "ROI result folders to read. Defaults to the current thr50/thr60 "
            "image and no-image result folders."
        ),
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    gm_mask_path = (
        args.gm_mask.expanduser().resolve()
        if args.gm_mask is not None
        else project_root / "masks" / "gm_0.1.nii"
    )
    out_root = (
        args.out_root.expanduser().resolve()
        if args.out_root is not None
        else project_root / "rsa_roi" / "neubert2015_ofc_thresholded_masks"
    )
    result_dirs = (
        [path.expanduser().resolve() for path in args.result_dirs]
        if args.result_dirs is not None
        else [project_root / rel for rel in DEFAULT_RESULT_DIRS]
    )

    gm_img = nib.load(str(gm_mask_path))
    specs = _read_specs(result_dirs, project_root)

    manifest_rows = []
    for spec in sorted(specs.values(), key=lambda item: (item.threshold, item.roi)):
        roi_img = nib.load(str(spec.source_mask))
        mask = _make_thresholded_mask(roi_img, gm_img, spec.threshold)

        label = _threshold_label(spec.threshold)
        out_dir = out_root / f"thr{label}"
        out_path = out_dir / f"mask-{spec.roi}_thr{label}_gm0p1.nii.gz"
        _save_mask(mask, roi_img, out_path, spec.roi, spec.threshold)

        manifest_rows.append(
            {
                "roi": spec.roi,
                "threshold": f"{spec.threshold:g}",
                "source_mask": str(spec.source_mask),
                "output_mask": str(out_path),
                "n_voxels": int(mask.sum()),
                "gm_mask": str(gm_mask_path),
            }
        )
        print(f"Saved {out_path} | voxels={int(mask.sum())}")

    manifest_path = out_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["roi", "threshold", "source_mask", "output_mask", "n_voxels", "gm_mask"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
