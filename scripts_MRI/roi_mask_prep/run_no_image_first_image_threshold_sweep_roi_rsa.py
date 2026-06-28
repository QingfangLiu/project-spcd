#!/usr/bin/env python3
"""Run no-image-first-image ROI RSA over threshold-sweep Neubert OFC masks.

This focused runner consumes saved binary masks from
``<base_dir>/rsa_roi/neubert2015_ofc_thresholded_masks/thrXX`` and runs only
the ``rsa_imagewise_odor_identity_no_image_first_image`` ROI RSA variant.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASE_DIR = Path("/Users/liuq13/SPCD_data")
DEFAULT_THRESHOLDS = (10, 20, 30, 40, 70, 80, 90)
DEFAULT_SKIP_THRESHOLDS = (50, 60)
MASK_ROOT_SUBDIR = Path("rsa_roi") / "neubert2015_ofc_thresholded_masks"
SCRIPT_FOLDER = "rsa_imagewise_odor_identity_no_image_first_image"
OUTPUT_FOLDER = "probe_imagewise_odor_identity_no_image_first_image"


@dataclass(frozen=True)
class RoiMask:
    roi: str
    threshold_label: str
    path: Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _threshold_label(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return f"{threshold:g}".replace(".", "p")


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


def _roi_from_mask_name(path: Path, threshold_label: str) -> str:
    name = path.name
    prefix = "mask-"
    suffixes = (
        f"_thr{threshold_label}_gm0p1.nii.gz",
        f"_thr{threshold_label}_gm0p1.nii",
    )
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected mask filename, missing {prefix!r}: {path}")
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[len(prefix) : -len(suffix)]
    raise ValueError(f"Unexpected mask filename for thr{threshold_label}: {path}")


def _find_masks(mask_root: Path, thresholds: list[float], rois: set[str] | None) -> list[RoiMask]:
    masks: list[RoiMask] = []
    for threshold in thresholds:
        label = _threshold_label(threshold)
        threshold_dir = mask_root / f"thr{label}"
        if not threshold_dir.is_dir():
            raise FileNotFoundError(f"Missing thresholded mask folder: {threshold_dir}")

        threshold_masks = sorted(threshold_dir.glob(f"mask-*_thr{label}_gm0p1.nii*"))
        if not threshold_masks:
            raise FileNotFoundError(f"No thresholded masks found in: {threshold_dir}")

        for path in threshold_masks:
            roi = _roi_from_mask_name(path, label)
            if rois is not None and roi not in rois:
                continue
            masks.append(RoiMask(roi=roi, threshold_label=label, path=path.resolve()))

    if not masks:
        raise FileNotFoundError("No masks matched the requested thresholds/ROIs")
    return masks


def _run_command(cmd: list[str], dry_run: bool) -> None:
    print("+ " + shlex.join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _run_roi(
    *,
    project_root: Path,
    base_dir: Path,
    mask: RoiMask,
    subjects: list[str] | None,
    cue_subsets: list[str] | None,
    save_diagnostics: bool,
    strict: bool,
    dry_run: bool,
) -> None:
    script_dir = project_root / "scripts_MRI" / SCRIPT_FOLDER
    run_script = script_dir / "run_roi_rsa.py"
    aggregate_script = script_dir / "aggregate_roi_rsa.py"
    if not run_script.exists():
        raise FileNotFoundError(f"Missing ROI RSA script: {run_script}")
    if not aggregate_script.exists():
        raise FileNotFoundError(f"Missing ROI aggregation script: {aggregate_script}")

    output_dir = base_dir / "rsa_roi" / OUTPUT_FOLDER / f"premeal_thr{mask.threshold_label}"
    subject_csv = output_dir / f"roi-{mask.roi}_subject_betas.csv"
    group_csv = output_dir / f"roi-{mask.roi}_group_summary.csv"

    run_cmd = [
        sys.executable,
        str(run_script),
        "--base_dir",
        str(base_dir),
        "--roi_mask",
        str(mask.path),
        "--roi_name",
        mask.roi,
        "--output_dir",
        str(output_dir),
        "--roi_threshold",
        "0",
        "--center_within_run",
        "--drop_nonfinite_voxels",
    ]
    if subjects:
        run_cmd.extend(["--subjects", *subjects])
    if cue_subsets:
        run_cmd.extend(["--cue_subsets", *cue_subsets])
    if save_diagnostics:
        run_cmd.append("--save_diagnostics")
    if strict:
        run_cmd.append("--strict")

    aggregate_cmd = [
        sys.executable,
        str(aggregate_script),
        "--input_csv",
        str(subject_csv),
        "--output_csv",
        str(group_csv),
        "--roi_name",
        mask.roi,
    ]

    print(f"\n[no_image_first_image | thr{mask.threshold_label} | {mask.roi}]")
    print(f"Mask: {mask.path}")
    _run_command(run_cmd, dry_run=dry_run)
    _run_command(aggregate_cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-image-first-image odor identity ROI RSA over saved "
            "Neubert OFC threshold-sweep masks."
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
        help="Project root. Default: infer from this script location.",
    )
    parser.add_argument(
        "--mask_root",
        type=Path,
        default=None,
        help=(
            "Folder containing thrXX saved binary masks. Default: "
            "<base_dir>/rsa_roi/neubert2015_ofc_thresholded_masks."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Threshold labels to run. Default: 10 20 30 40 70 80 90.",
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
        help="Optional ROI labels to run, e.g. Left_47m. Default: all masks.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs to pass through to run_roi_rsa.py.",
    )
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=["conditioned_odor_only", "preconditioned_odor_only"],
        default=None,
        help="Optional cue subsets to pass through to run_roi_rsa.py. Default: both.",
    )
    parser.add_argument(
        "--save_diagnostics",
        action="store_true",
        help="Pass --save_diagnostics to each ROI RSA run.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Pass --strict to each ROI RSA run.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    mask_root = (
        args.mask_root.expanduser().resolve()
        if args.mask_root is not None
        else base_dir / MASK_ROOT_SUBDIR
    )
    thresholds = _filtered_thresholds(args.thresholds, set(args.skip_thresholds or []))

    print(f"Project root: {project_root}")
    print(f"Base dir:     {base_dir}")
    print(f"Mask root:    {mask_root}")
    print("Variant:      no_image_first_image")
    print(f"Thresholds:   {', '.join(_threshold_label(t) for t in thresholds)}")
    if args.rois:
        print(f"ROIs:         {', '.join(args.rois)}")
    if args.dry_run:
        print("Dry run:      true")

    masks = _find_masks(mask_root, thresholds, set(args.rois) if args.rois else None)
    for mask in masks:
        _run_roi(
            project_root=project_root,
            base_dir=base_dir,
            mask=mask,
            subjects=args.subjects,
            cue_subsets=args.cue_subsets,
            save_diagnostics=args.save_diagnostics,
            strict=args.strict,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
