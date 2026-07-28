#!/usr/bin/env python3
"""Rank-based ROI RSA for first-image premeal odor identity.

This additions-only analysis reuses the established sample selection, ROI
pattern preparation, correlation-distance neural RDM, and cross-run pair mask
from ``run_roi_rsa.py``.  It replaces the OLS RDM fit with:

* Spearman's rho between neural distances and the binary odor-identity RDM.
* Rank-biserial correlation, with its equivalent area under the ROC curve.

Positive values mean that different-odor pairs tend to be more neurally
dissimilar than same-odor pairs.  No pair-level p-value is reported because
entries of an RDM are not independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rsa_imagewise_odor_identity_no_image_first_image import (
        run_roi_rsa as established_roi,
    )
else:
    from . import run_roi_rsa as established_roi


SPEARMAN_COL = "odor_identity_no_image_spearman_rho"
RANK_BISERIAL_COL = "odor_identity_no_image_rank_biserial"
AUC_COL = "odor_identity_no_image_auc"
DISTANCE_CONTRAST_COL = "odor_identity_no_image_distance_contrast"


def compute_rank_rsa_statistics(
    neural_rdm: np.ndarray,
    odor_identity_rdm: np.ndarray,
    pair_mask: np.ndarray,
) -> dict[str, float | int]:
    """Compute rank-based odor-identity RSA statistics on selected RDM pairs."""
    neural_vec = established_roi.vectorize_lower_triangle(
        neural_rdm,
        pair_mask=pair_mask,
    )
    odor_vec = established_roi.vectorize_lower_triangle(
        odor_identity_rdm,
        pair_mask=pair_mask,
    )

    if neural_vec.size == 0:
        raise ValueError("No cross-run RDM pairs are available.")
    if not np.isfinite(neural_vec).all():
        raise ValueError("Selected neural RDM pairs contain nonfinite values.")
    if not np.isfinite(odor_vec).all():
        raise ValueError("Selected odor-identity RDM pairs contain nonfinite values.")

    model_values = set(np.unique(odor_vec).tolist())
    if model_values != {0.0, 1.0}:
        raise ValueError(
            "Odor-identity RDM must contain both binary values 0 and 1; "
            f"found {sorted(model_values)}."
        )
    if np.ptp(neural_vec) == 0:
        raise ValueError("Selected neural RDM distances are constant.")

    same_mask = odor_vec == 0
    different_mask = odor_vec == 1
    n_same = int(same_mask.sum())
    n_different = int(different_mask.sum())

    # Spearman's rho is Pearson correlation after average-rank
    # transformation. Compute it directly so no invalid pair-level
    # inferential p-value is produced.
    neural_ranks = stats.rankdata(neural_vec, method="average")
    odor_ranks = stats.rankdata(odor_vec, method="average")
    spearman_rho = float(np.corrcoef(neural_ranks, odor_ranks)[0, 1])
    if not np.isfinite(spearman_rho):
        raise ValueError("Spearman correlation is nonfinite.")

    # Mann-Whitney U for different-odor distances relative to same-odor
    # distances. Average ranks give half credit to tied neural distances.
    different_rank_sum = float(neural_ranks[different_mask].sum())
    mann_whitney_u = (
        different_rank_sum
        - n_different * (n_different + 1) / 2.0
    )
    auc = float(mann_whitney_u / (n_different * n_same))
    rank_biserial = float(2.0 * auc - 1.0)

    mean_same = float(np.mean(neural_vec[same_mask]))
    mean_different = float(np.mean(neural_vec[different_mask]))

    return {
        SPEARMAN_COL: spearman_rho,
        RANK_BISERIAL_COL: rank_biserial,
        AUC_COL: auc,
        DISTANCE_CONTRAST_COL: mean_different - mean_same,
        "mean_neural_distance_same_odor": mean_same,
        "mean_neural_distance_different_odor": mean_different,
        "same_odor_pairs": n_same,
        "diff_odor_pairs": n_different,
    }


def _run_roi_rank_rsa_for_subset(
    base_dir: Path,
    sub: str,
    cue_subset: str,
    roi_img,
    roi_mask: np.ndarray,
    roi_name: str,
    roi_mask_path: Path,
    center_within_run: bool,
    drop_nonfinite_voxels: bool,
    roi_threshold: float,
    diagnostics_root: Path | None = None,
) -> dict:
    samples = established_roi._build_samples(
        base_dir=base_dir,
        sub=sub,
        cue_subset=cue_subset,
    )
    if not samples:
        raise RuntimeError(f"No samples found for {sub} {cue_subset}")

    run_ids = np.asarray([sample["run"] for sample in samples], dtype=int)
    n_runs = int(np.unique(run_ids).size)
    if n_runs < 2:
        raise RuntimeError(
            f"Need at least 2 runs for cross-run RSA: {sub} {cue_subset}"
        )

    raw_patterns = established_roi._load_roi_patterns(
        samples,
        roi_img,
        roi_mask,
    )
    (
        raw_used_patterns,
        patterns,
        used_voxel_mask,
        voxel_qc,
    ) = established_roi._prepare_analysis_patterns(
        raw_patterns=raw_patterns,
        run_ids=run_ids,
        center_within_run=center_within_run,
        drop_nonfinite_voxels=drop_nonfinite_voxels,
    )

    if voxel_qc["n_voxels_nonfinite"] > 0:
        if drop_nonfinite_voxels:
            print(
                f"[QC] {sub} {cue_subset}: dropped "
                f"{voxel_qc['n_voxels_dropped_nonfinite']} nonfinite ROI voxels; "
                f"using {voxel_qc['n_voxels_used']}/"
                f"{voxel_qc['n_voxels_raw']} voxels"
            )
        else:
            print(
                f"[QC] {sub} {cue_subset}: found "
                f"{voxel_qc['n_voxels_nonfinite']} nonfinite ROI voxels; "
                "rerun with --drop_nonfinite_voxels to omit them"
            )

    sample_labels = np.asarray(
        [sample["sample_label"] for sample in samples],
        dtype=object,
    )
    odor_identity_rdm = established_roi.create_odor_identity_rdm(sample_labels)
    neural_rdm = established_roi.compute_correlation_rdm(patterns)
    pair_mask = established_roi._cross_run_pair_mask(run_ids)

    if diagnostics_root is not None:
        established_roi._write_roi_diagnostics(
            diagnostics_root=diagnostics_root,
            sub=sub,
            cue_subset=cue_subset,
            samples=samples,
            raw_patterns=raw_patterns,
            raw_used_patterns=raw_used_patterns,
            analysis_patterns=patterns,
            run_ids=run_ids,
            roi_mask=roi_mask,
            used_voxel_mask=used_voxel_mask,
            voxel_qc=voxel_qc,
            neural_rdm=neural_rdm,
            pair_mask=pair_mask,
            model_rdms={"odor_identity_no_image": odor_identity_rdm},
        )

    rank_stats = compute_rank_rsa_statistics(
        neural_rdm=neural_rdm,
        odor_identity_rdm=odor_identity_rdm,
        pair_mask=pair_mask,
    )
    voxel_loss_fraction = (
        voxel_qc["n_voxels_dropped_nonfinite"]
        / voxel_qc["n_voxels_raw"]
    )

    row = {
        "sub": sub,
        "session": "premeal",
        "cue_subset": cue_subset,
        "roi": roi_name,
        "roi_mask": str(roi_mask_path),
        "n_voxels": voxel_qc["n_voxels_used"],
        "n_voxels_raw": voxel_qc["n_voxels_raw"],
        "n_voxels_used": voxel_qc["n_voxels_used"],
        "n_voxels_nonfinite": voxel_qc["n_voxels_nonfinite"],
        "n_voxels_dropped_nonfinite": (
            voxel_qc["n_voxels_dropped_nonfinite"]
        ),
        "voxel_loss_fraction": voxel_loss_fraction,
        "high_voxel_loss_20pct": bool(voxel_loss_fraction >= 0.20),
        "n_samples": len(samples),
        "n_runs": n_runs,
        "runs": ",".join(
            str(run) for run in sorted(np.unique(run_ids))
        ),
        "incomplete_three_run_data": bool(n_runs < 3),
        "n_pairs": int(pair_mask.sum()),
        "center_within_run": bool(center_within_run),
        "drop_nonfinite_voxels": bool(drop_nonfinite_voxels),
        "roi_threshold": roi_threshold,
        **rank_stats,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run first-image-only Spearman ROI RSA for premeal odor identity "
            "without image identity in odor-predictive cues."
        )
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory containing GLM, nii, and masks folders.",
    )
    parser.add_argument(
        "--roi_mask",
        required=True,
        help="ROI mask path, absolute or relative to base_dir.",
    )
    parser.add_argument(
        "--roi_name",
        default="lateral_ofc",
        help="Label used in output CSVs.",
    )
    parser.add_argument(
        "--roi_threshold",
        type=float,
        default=0.0,
        help=(
            "ROI voxel cutoff. Zero uses mask > 0; positive values use "
            "mask >= threshold."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject IDs; defaults to discovered premeal GLM subjects.",
    )
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=sorted(established_roi.CUE_SUBSETS),
        default=sorted(established_roi.CUE_SUBSETS),
        help="Cue subsets to run.",
    )
    parser.add_argument("--center_within_run", action="store_true")
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Default: <base_dir>/rsa_roi/"
            "probe_imagewise_odor_identity_no_image_first_image/"
            "premeal_spearman."
        ),
    )
    parser.add_argument(
        "--save_diagnostics",
        action="store_true",
        help=(
            "Save per-subject patterns, sample metadata, NaN reports, and "
            "the selected neural/model RDM pairs."
        ),
    )
    parser.add_argument(
        "--drop_nonfinite_voxels",
        action="store_true",
        help=(
            "Drop ROI voxels with any nonfinite sample before centering and "
            "RDM computation."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately instead of skipping failed subject/subset runs.",
    )
    args = parser.parse_args()

    if args.roi_threshold < 0:
        raise ValueError("--roi_threshold must be nonnegative")

    base_dir = Path(args.base_dir).expanduser().resolve()
    roi_mask_path = established_roi._resolve_under_base(
        args.roi_mask,
        base_dir,
    ).resolve()
    roi_name = established_roi._sanitize_label(args.roi_name)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            base_dir
            / "rsa_roi"
            / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_spearman"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"roi-{roi_name}_subject_spearman.csv"
    diagnostics_root = (
        output_dir / "diagnostics" if args.save_diagnostics else None
    )

    gm_mask_path = base_dir / "masks" / "gm_0.1.nii"
    roi_img, roi_mask, n_roi_raw, n_roi_final, n_gm = (
        established_roi._load_mask(
            roi_mask_path,
            gm_mask_path=gm_mask_path,
            roi_threshold=args.roi_threshold,
        )
    )
    subjects = (
        sorted(args.subjects)
        if args.subjects
        else established_roi._discover_subjects(base_dir)
    )

    print("=" * 80)
    print("Premeal odor-identity no-image Spearman ROI RSA")
    print("=" * 80)
    print(f"Base dir:              {base_dir}")
    print(f"ROI mask:              {roi_mask_path}")
    print(f"ROI label:             {roi_name}")
    print(f"ROI threshold:         {args.roi_threshold:g}")
    print(f"ROI voxels raw/final:  {n_roi_raw} / {n_roi_final}")
    if n_gm is not None:
        print(f"GM mask voxels:        {n_gm}")
    print(f"Subjects:              {len(subjects)}")
    print(f"Cue subsets:           {', '.join(args.cue_subsets)}")
    print("Image scope:           first image only")
    print("Neural distance:       1 - Pearson pattern correlation")
    print("RDM statistic:         Spearman rho + rank-biserial/AUC")
    print("Pair-level p-values:   not calculated (RDM pairs are dependent)")
    print(f"Center within run:     {args.center_within_run}")
    print(f"Drop nonfinite voxels: {args.drop_nonfinite_voxels}")
    print(f"Output CSV:            {output_csv}")
    if diagnostics_root is not None:
        print(f"Diagnostics dir:       {diagnostics_root}")
    print("=" * 80)

    rows = []
    failures = []
    for sub in subjects:
        for cue_subset in args.cue_subsets:
            try:
                row = _run_roi_rank_rsa_for_subset(
                    base_dir=base_dir,
                    sub=sub,
                    cue_subset=cue_subset,
                    roi_img=roi_img,
                    roi_mask=roi_mask,
                    roi_name=roi_name,
                    roi_mask_path=roi_mask_path,
                    center_within_run=args.center_within_run,
                    drop_nonfinite_voxels=args.drop_nonfinite_voxels,
                    roi_threshold=args.roi_threshold,
                    diagnostics_root=diagnostics_root,
                )
                rows.append(row)
                print(
                    f"[OK] {sub} {cue_subset}: "
                    f"rho={row[SPEARMAN_COL]:.6f}, "
                    f"rank-biserial={row[RANK_BISERIAL_COL]:.6f}, "
                    f"samples={row['n_samples']}, "
                    f"pairs={row['n_pairs']}, "
                    f"voxels={row['n_voxels_used']}/"
                    f"{row['n_voxels_raw']}"
                )
            except Exception as exc:
                message = f"{sub} {cue_subset}: {exc}"
                failures.append(message)
                print(f"[FAIL] {message}")
                if args.strict:
                    raise

    if not rows:
        raise RuntimeError("No ROI Spearman RSA rows were produced.")

    out = pd.DataFrame(rows)
    out = out.sort_values(["sub", "cue_subset"]).reset_index(drop=True)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved subject Spearman ROI RSA results: {output_csv}")

    incomplete = sorted(
        out.loc[out["incomplete_three_run_data"], "sub"].unique()
    )
    high_loss = sorted(
        out.loc[out["high_voxel_loss_20pct"], "sub"].unique()
    )
    print("\nQC flags (reported only; no automatic exclusion):")
    print(
        "  Fewer than three runs: "
        + (", ".join(incomplete) if incomplete else "none")
    )
    print(
        "  At least 20% voxel loss: "
        + (", ".join(high_loss) if high_loss else "none")
    )

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
