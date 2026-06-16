#!/usr/bin/env python3
"""Run Neubert OFC ROI RSA from saved thresholded binary masks.

This batch runner is for rerunning the existing ``premeal_thr50`` and
``premeal_thr60`` ROI analyses without applying a probabilistic ROI threshold
inside ``run_roi_rsa.py``. Instead, it uses the saved binary masks under
``rsa_roi/neubert2015_ofc_thresholded_masks`` and lets ``run_roi_rsa.py`` use
its default mask rule, ``mask > 0``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_THRESHOLDS = (50, 60)
DEFAULT_VARIANTS = ("image", "no_image")


@dataclass(frozen=True)
class RoiMask:
    roi: str
    threshold_label: str
    path: Path


@dataclass(frozen=True)
class Variant:
    key: str
    script_folder: str
    output_folder: str


VARIANTS = {
    "image": Variant(
        key="image",
        script_folder="rsa_imagewise_odor_identity",
        output_folder="probe_imagewise_odor_identity",
    ),
    "no_image": Variant(
        key="no_image",
        script_folder="rsa_imagewise_odor_identity_no_image",
        output_folder="probe_imagewise_odor_identity_no_image",
    ),
}


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _threshold_label(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


def _choose_mask_root(base_dir: Path, project_root: Path, user_value: Path | None) -> Path:
    if user_value is not None:
        return user_value.expanduser().resolve()

    candidates = (
        base_dir / "rsa_roi" / "neubert2015_ofc_thresholded_masks",
        project_root / "rsa_roi" / "neubert2015_ofc_thresholded_masks",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    return candidates[0].resolve()


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
    variant: Variant,
    mask: RoiMask,
    subjects: list[str] | None,
    cue_subsets: list[str] | None,
    save_diagnostics: bool,
    strict: bool,
    dry_run: bool,
) -> None:
    script_dir = project_root / "scripts_MRI" / variant.script_folder
    run_script = script_dir / "run_roi_rsa.py"
    aggregate_script = script_dir / "aggregate_roi_rsa.py"
    if not run_script.exists():
        raise FileNotFoundError(f"Missing ROI RSA script: {run_script}")
    if not aggregate_script.exists():
        raise FileNotFoundError(f"Missing ROI aggregation script: {aggregate_script}")

    output_dir = base_dir / "rsa_roi" / variant.output_folder / f"premeal_thr{mask.threshold_label}"
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

    print(
        f"\n[{variant.key} | thr{mask.threshold_label} | {mask.roi}] "
        f"mask={mask.path}"
    )
    _run_command(run_cmd, dry_run=dry_run)
    _run_command(aggregate_cmd, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun Neubert OFC ROI RSA using saved thresholded binary masks "
            "instead of applying --roi_threshold to probabilistic masks."
        )
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        type=Path,
        help="Data base directory containing GLM, nii, masks, and rsa_roi folders.",
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
            "Folder containing thr50/ and thr60/ saved binary masks. Default: "
            "<base_dir>/rsa_roi/neubert2015_ofc_thresholded_masks if it exists, "
            "otherwise <project_root>/rsa_roi/neubert2015_ofc_thresholded_masks."
        ),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Threshold labels to run. Default: 50 60.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=list(DEFAULT_VARIANTS),
        help="ROI RSA variants to run. Default: image no_image.",
    )
    parser.add_argument(
        "--rois",
        nargs="+",
        default=None,
        help="Optional ROI labels to run, e.g. Left_11m Right_11m. Default: all masks.",
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

    project_root = args.project_root.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    mask_root = _choose_mask_root(base_dir, project_root, args.mask_root)

    print(f"Project root: {project_root}")
    print(f"Base dir:     {base_dir}")
    print(f"Mask root:    {mask_root}")
    print(f"Variants:     {', '.join(args.variants)}")
    print(f"Thresholds:   {', '.join(_threshold_label(t) for t in args.thresholds)}")
    if args.rois:
        print(f"ROIs:         {', '.join(args.rois)}")
    if args.dry_run:
        print("Dry run:      true")

    masks = _find_masks(mask_root, args.thresholds, set(args.rois) if args.rois else None)
    variants = [VARIANTS[key] for key in args.variants]

    for variant in variants:
        for mask in masks:
            _run_roi(
                project_root=project_root,
                base_dir=base_dir,
                variant=variant,
                mask=mask,
                subjects=args.subjects,
                cue_subsets=args.cue_subsets,
                save_diagnostics=args.save_diagnostics,
                strict=args.strict,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
