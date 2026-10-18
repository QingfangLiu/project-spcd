#!/usr/bin/env python3
"""Cross-task Spearman RSA between passive odors and premeal cue patterns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
DECODING_DIR = SCRIPTS_DIR / "Decoding_probe_cross_task"
RSA_DIR = SCRIPTS_DIR / "rsa_imagewise_odor_identity_no_image_first_image"
for dependency_dir in (DECODING_DIR, RSA_DIR):
    if str(dependency_dir) not in sys.path:
        sys.path.insert(0, str(dependency_dir))

import run_roi_crosstask_odor2preconditioned as decoding
import run_roi_rsa as probe_rsa
import run_roi_rsa_spearman as rank_rsa


CUE_SUBSETS = (
    "preconditioned_odor_only",
    "conditioned_odor_only",
)
HEMISPHERES = decoding.HEMISPHERES
DEFAULT_ODOR_RUNS = decoding.DEFAULT_ODOR_RUNS
SUB14 = decoding.SUB14

METRIC_RENAMES = {
    rank_rsa.SPEARMAN_COL: "cross_task_odor_identity_spearman_rho",
    rank_rsa.RANK_BISERIAL_COL: (
        "cross_task_odor_identity_rank_biserial"
    ),
    rank_rsa.AUC_COL: "cross_task_odor_identity_auc",
    rank_rsa.DISTANCE_CONTRAST_COL: (
        "cross_task_odor_identity_distance_contrast"
    ),
}


def _cross_task_pair_mask(
    n_passive: int,
    n_cue: int,
) -> np.ndarray:
    """Select passive-by-cue cells from a combined square RDM."""
    if n_passive <= 0 or n_cue <= 0:
        raise ValueError("Both tasks must contain at least one sample.")
    task = np.concatenate(
        [
            np.zeros(n_passive, dtype=int),
            np.ones(n_cue, dtype=int),
        ]
    )
    full_mask = task[:, None] != task[None, :]
    return full_mask[np.tril_indices(len(task), k=-1)]


def _identity_rdm(
    passive_labels: np.ndarray,
    cue_labels: np.ndarray,
) -> np.ndarray:
    labels = np.concatenate(
        [
            np.asarray(passive_labels, dtype=int),
            np.asarray(cue_labels, dtype=int),
        ]
    )
    if set(labels) != {0, 1}:
        raise ValueError(
            "Cross-task odor labels must contain both binary identities."
        )
    rdm = (labels[:, None] != labels[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _compute_cross_task_rank_statistics(
    passive_patterns: np.ndarray,
    passive_labels: np.ndarray,
    cue_patterns: np.ndarray,
    cue_labels: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    """Compute rank RSA after retaining only passive-to-cue RDM pairs."""
    passive_patterns = np.asarray(passive_patterns, dtype=float)
    cue_patterns = np.asarray(cue_patterns, dtype=float)
    if passive_patterns.ndim != 2 or cue_patterns.ndim != 2:
        raise ValueError("Passive and cue patterns must be two-dimensional.")
    if passive_patterns.shape[1] != cue_patterns.shape[1]:
        raise ValueError(
            "Passive and cue patterns must contain identical ROI voxels."
        )
    if len(passive_patterns) != len(passive_labels):
        raise ValueError("Passive pattern and label counts do not match.")
    if len(cue_patterns) != len(cue_labels):
        raise ValueError("Cue pattern and label counts do not match.")

    all_patterns = np.vstack([passive_patterns, cue_patterns])
    neural_rdm = probe_rsa.compute_correlation_rdm(all_patterns)
    identity_rdm = _identity_rdm(passive_labels, cue_labels)
    pair_mask = _cross_task_pair_mask(
        len(passive_patterns),
        len(cue_patterns),
    )
    rank_statistics = rank_rsa.compute_rank_rsa_statistics(
        neural_rdm=neural_rdm,
        odor_identity_rdm=identity_rdm,
        pair_mask=pair_mask,
    )
    renamed = {
        METRIC_RENAMES.get(name, name): value
        for name, value in rank_statistics.items()
    }
    expected_pairs = len(passive_patterns) * len(cue_patterns)
    if int(pair_mask.sum()) != expected_pairs:
        raise RuntimeError(
            "Cross-task pair mask does not equal passive × cue pairs."
        )
    return renamed, neural_rdm, identity_rdm, pair_mask


def _cross_task_pair_table(
    neural_rdm: np.ndarray,
    identity_rdm: np.ndarray,
    passive_metadata: pd.DataFrame,
    cue_samples: pd.DataFrame,
    cue_labels: np.ndarray,
) -> pd.DataFrame:
    n_passive = len(passive_metadata)
    n_cue = len(cue_samples)
    rows = []
    for cue_index in range(n_cue):
        combined_cue_index = n_passive + cue_index
        cue_row = cue_samples.iloc[cue_index]
        for passive_index in range(n_passive):
            passive_row = passive_metadata.iloc[passive_index]
            rows.append(
                {
                    "passive_sample_index": passive_index,
                    "passive_run": int(passive_row["run"]),
                    "passive_trial": int(
                        passive_row["beta_trial_number"]
                    ),
                    "passive_trial_type": passive_row["trial_type"],
                    "passive_label": int(passive_row["label"]),
                    "cue_sample_index": cue_index,
                    "cue_run": int(cue_row["run"]),
                    "cue_trial": int(cue_row["trial"]),
                    "cue_condition": cue_row["condition"],
                    "cue_label": int(cue_labels[cue_index]),
                    "neural_correlation_distance": float(
                        neural_rdm[passive_index, combined_cue_index]
                    ),
                    "odor_identity_dissimilarity": int(
                        identity_rdm[passive_index, combined_cue_index]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _analyze_subject(
    *,
    base_dir: Path,
    diagnostics_dir: Path,
    sub: str,
    roi_name: str,
    hemisphere: str,
    cue_subsets: tuple[str, ...],
    odor_runs: tuple[int, ...],
    sub14_original_events_dir: Path,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    cue_inputs = {}
    reference_voxel_indices = None
    for cue_subset in cue_subsets:
        raw, voxel_indices, samples, pattern_path = (
            decoding._load_cue_diagnostics(
                diagnostics_dir,
                sub,
                cue_subset,
            )
        )
        labels, label_source = decoding._cue_labels(
            sub,
            samples,
            sub14_original_events_dir,
        )
        if reference_voxel_indices is None:
            reference_voxel_indices = voxel_indices
        elif not np.array_equal(reference_voxel_indices, voxel_indices):
            raise ValueError(
                "Cue subsets do not use identical ROI voxel coordinates."
            )
        cue_inputs[cue_subset] = {
            "raw": raw,
            "samples": samples,
            "labels": labels,
            "label_source": label_source,
            "pattern_path": pattern_path,
        }

    if reference_voxel_indices is None:
        raise RuntimeError("No cue subsets were loaded.")

    (
        passive_raw,
        passive_labels,
        passive_runs,
        passive_metadata,
        passive_affine,
    ) = decoding._load_passive_odor_patterns(
        base_dir,
        sub,
        odor_runs,
        reference_voxel_indices,
    )

    hemisphere_mask, midline_mask = decoding._hemisphere_voxel_mask(
        reference_voxel_indices,
        passive_affine,
        hemisphere,
    )
    passive_selected = passive_raw[:, hemisphere_mask]
    cue_selected = {
        cue_subset: values["raw"][:, hemisphere_mask]
        for cue_subset, values in cue_inputs.items()
    }

    finite_voxels = np.isfinite(passive_selected).all(axis=0)
    for patterns in cue_selected.values():
        finite_voxels &= np.isfinite(patterns).all(axis=0)
    if not finite_voxels.any():
        raise ValueError("No jointly finite passive/cue ROI voxels remain.")

    passive_used = passive_selected[:, finite_voxels]
    passive_patterns = probe_rsa._center_within_run(
        passive_used,
        passive_runs,
    )
    n_voxels_bilateral = int(len(hemisphere_mask))
    n_voxels_raw = int(hemisphere_mask.sum())
    n_voxels_used = int(finite_voxels.sum())

    rows = []
    pair_tables = {}
    for cue_subset in cue_subsets:
        values = cue_inputs[cue_subset]
        samples = values["samples"].copy().reset_index(drop=True)
        cue_labels = np.asarray(values["labels"], dtype=int)
        cue_runs = samples["run"].astype(int).to_numpy()
        cue_patterns = probe_rsa._center_within_run(
            cue_selected[cue_subset][:, finite_voxels],
            cue_runs,
        )
        rank_stats, neural_rdm, identity_rdm, pair_mask = (
            _compute_cross_task_rank_statistics(
                passive_patterns,
                passive_labels,
                cue_patterns,
                cue_labels,
            )
        )
        pair_table = _cross_task_pair_table(
            neural_rdm,
            identity_rdm,
            passive_metadata,
            samples,
            cue_labels,
        )
        pair_table["sub"] = sub
        pair_table["roi"] = roi_name
        pair_table["hemisphere"] = hemisphere
        pair_table["cue_subset"] = cue_subset
        pair_tables[cue_subset] = pair_table

        rows.append(
            {
                "sub": sub,
                "roi": roi_name,
                "hemisphere": hemisphere,
                "hemisphere_rule": (
                    "all_voxels"
                    if hemisphere == "bilateral"
                    else f"MNI_x_{'lt' if hemisphere == 'left' else 'gt'}_0"
                ),
                "cue_subset": cue_subset,
                "cue_label_source": values["label_source"],
                "cue_diagnostic_npz": str(values["pattern_path"]),
                "normalization": "within_task_run_voxelwise_centering",
                "neural_distance": "1_minus_pearson_correlation",
                "pair_scope": "passive_odor_to_premeal_cue_only",
                "n_voxels_bilateral_raw": n_voxels_bilateral,
                "n_voxels_midline": int(midline_mask.sum()),
                "n_voxels_excluded_by_hemisphere": (
                    n_voxels_bilateral - n_voxels_raw
                ),
                "n_voxels_raw": n_voxels_raw,
                "n_voxels_used": n_voxels_used,
                "n_voxels_dropped_nonfinite": (
                    n_voxels_raw - n_voxels_used
                ),
                "n_passive_samples": int(len(passive_labels)),
                "n_passive_odor1": int((passive_labels == 0).sum()),
                "n_passive_odor2": int((passive_labels == 1).sum()),
                "n_passive_runs": int(len(np.unique(passive_runs))),
                "passive_runs": ",".join(
                    map(str, sorted(np.unique(passive_runs)))
                ),
                "n_cue_samples": int(len(cue_labels)),
                "n_cue_odor1": int((cue_labels == 0).sum()),
                "n_cue_odor2": int((cue_labels == 1).sum()),
                "n_cue_runs": int(len(np.unique(cue_runs))),
                "cue_runs": ",".join(
                    map(str, sorted(np.unique(cue_runs)))
                ),
                "n_cross_task_pairs": int(pair_mask.sum()),
                **rank_stats,
            }
        )

    return rows, pair_tables


def _scope_output_dir(
    output_root: Path,
    roi_name: str,
    hemisphere: str,
) -> Path:
    return output_root / f"roi-{roi_name}_{hemisphere}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute trial-level rectangular Spearman RSA between passive "
            "odor patterns and first-image premeal cue patterns."
        )
    )
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--diagnostics_dir", type=Path, required=True)
    parser.add_argument("--roi_name", required=True)
    parser.add_argument(
        "--hemispheres",
        nargs="+",
        choices=HEMISPHERES,
        default=["bilateral"],
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=CUE_SUBSETS,
        default=list(CUE_SUBSETS),
    )
    parser.add_argument(
        "--odor_runs",
        nargs="+",
        type=int,
        default=list(DEFAULT_ODOR_RUNS),
    )
    parser.add_argument("--sub14_original_events_dir", type=Path)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--save_pair_diagnostics",
        action="store_true",
        help="Save one row per passive-to-cue RDM pair.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    diagnostics_dir = args.diagnostics_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    roi_name = decoding._sanitize_label(args.roi_name)
    sub14_events = (
        args.sub14_original_events_dir.expanduser().resolve()
        if args.sub14_original_events_dir is not None
        else base_dir / "nii" / SUB14 / "func_PAID" / "premeal"
    )
    subjects = (
        list(dict.fromkeys(args.subjects))
        if args.subjects is not None
        else decoding._discover_subjects(diagnostics_dir)
    )
    cue_subsets = tuple(dict.fromkeys(args.cue_subsets))
    odor_runs = tuple(dict.fromkeys(args.odor_runs))
    if len(odor_runs) < 2:
        raise ValueError("--odor_runs must contain at least two runs.")

    print("=" * 80)
    print("Passive odor -> premeal cue rectangular Spearman ROI RSA")
    print("=" * 80)
    print(f"SPCD data root:       {base_dir}")
    print(f"RSA diagnostics:      {diagnostics_dir}")
    print(f"ROI:                  {roi_name}")
    print(f"Hemispheres:          {', '.join(args.hemispheres)}")
    print(f"Cue subsets:          {', '.join(cue_subsets)} (separate)")
    print(f"Subjects:             {len(subjects)}")
    print(f"Output root:          {output_root}")
    print("Normalization:        within-task, within-run voxel centering")
    print("Pair scope:           passive odor <-> premeal cue only")
    print("Pair-level p-values:  not calculated")
    print("=" * 80)

    for hemisphere in dict.fromkeys(args.hemispheres):
        scope_dir = _scope_output_dir(output_root, roi_name, hemisphere)
        rows = []
        failures = []
        for sub in subjects:
            try:
                subject_rows, pair_tables = _analyze_subject(
                    base_dir=base_dir,
                    diagnostics_dir=diagnostics_dir,
                    sub=sub,
                    roi_name=roi_name,
                    hemisphere=hemisphere,
                    cue_subsets=cue_subsets,
                    odor_runs=odor_runs,
                    sub14_original_events_dir=sub14_events,
                )
                rows.extend(subject_rows)
                if args.save_pair_diagnostics:
                    subject_dir = scope_dir / "pair_diagnostics" / sub
                    subject_dir.mkdir(parents=True, exist_ok=True)
                    for cue_subset, pair_table in pair_tables.items():
                        pair_table.to_csv(
                            subject_dir / f"{cue_subset}.csv",
                            index=False,
                        )
                summary = ", ".join(
                    f"{row['cue_subset']}="
                    f"{row['cross_task_odor_identity_spearman_rho']:.5f}"
                    for row in subject_rows
                )
                print(f"[OK] {sub} {hemisphere}: {summary}")
            except Exception as exc:
                if args.strict:
                    raise
                failures.append(
                    {
                        "sub": sub,
                        "roi": roi_name,
                        "hemisphere": hemisphere,
                        "cue_subsets": ",".join(cue_subsets),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"[FAIL] {sub} {hemisphere}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

        if not rows:
            raise RuntimeError(
                f"No analyses completed for {roi_name}, {hemisphere}."
            )
        scope_dir.mkdir(parents=True, exist_ok=True)
        subject_csv = scope_dir / "subject_crosstask_spearman_rsa.csv"
        pd.DataFrame(rows).to_csv(subject_csv, index=False)
        pd.DataFrame(
            failures,
            columns=[
                "sub",
                "roi",
                "hemisphere",
                "cue_subsets",
                "error_type",
                "error",
            ],
        ).to_csv(scope_dir / "failures.csv", index=False)
        print(f"Saved {subject_csv}")


if __name__ == "__main__":
    main()
