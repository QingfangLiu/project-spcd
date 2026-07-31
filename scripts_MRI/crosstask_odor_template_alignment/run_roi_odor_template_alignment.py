#!/usr/bin/env python3
"""Align premeal cue patterns with passive odor-identity templates."""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
DECODING_DIR = SCRIPT_DIR.parent / "Decoding_probe_cross_task"
if str(DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(DECODING_DIR))

import run_roi_crosstask_odor2preconditioned as decoding


CUE_SUBSETS = (
    "preconditioned_odor_only",
    "conditioned_odor_only",
)
HEMISPHERES = decoding.HEMISPHERES
DEFAULT_ODOR_RUNS = decoding.DEFAULT_ODOR_RUNS
SUB14 = decoding.SUB14


def _center_and_unit_normalize(patterns: np.ndarray) -> np.ndarray:
    patterns = np.asarray(patterns, dtype=float)
    was_vector = patterns.ndim == 1
    if was_vector:
        patterns = patterns[None, :]
    if patterns.ndim != 2 or patterns.shape[1] < 2:
        raise ValueError(
            "Pattern correlations require at least two ROI voxels."
        )
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("A pattern has zero or nonfinite variance across voxels.")
    normalized = centered / norms
    return normalized[0] if was_vector else normalized


def _pattern_correlations(
    patterns: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    normalized_patterns = _center_and_unit_normalize(patterns)
    normalized_prototypes = _center_and_unit_normalize(prototypes)
    return normalized_patterns @ normalized_prototypes.T


def _build_run_balanced_prototypes(
    patterns: np.ndarray,
    labels: np.ndarray,
    run_ids: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    patterns = np.asarray(patterns, dtype=float)
    labels = np.asarray(labels, dtype=int)
    run_ids = np.asarray(run_ids, dtype=int)
    if patterns.ndim != 2 or len(patterns) != len(labels):
        raise ValueError("Passive patterns and labels have inconsistent shapes.")

    run_prototypes: dict[int, np.ndarray] = {}
    for run in np.unique(run_ids):
        run_mask = run_ids == run
        if set(labels[run_mask]) != {0, 1}:
            raise ValueError(
                f"Passive run {run} does not contain both odor identities."
            )
        run_prototypes[int(run)] = np.vstack(
            [
                patterns[run_mask & (labels == odor)].mean(axis=0)
                for odor in (0, 1)
            ]
        )

    prototypes = np.mean(
        np.stack(list(run_prototypes.values()), axis=0),
        axis=0,
    )
    return prototypes, run_prototypes


def _run_and_label_balanced_mean(
    values: np.ndarray,
    labels: np.ndarray,
    run_ids: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    run_ids = np.asarray(run_ids, dtype=int)
    cell_means = []
    for run in np.unique(run_ids):
        for odor in (0, 1):
            cell = values[(run_ids == run) & (labels == odor)]
            if cell.size == 0:
                raise ValueError(
                    f"Run {run} has no samples for odor label {odor + 1}."
                )
            cell_means.append(float(cell.mean()))
    return float(np.mean(cell_means))


def _run_balanced_mean(values: np.ndarray, run_ids: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    run_ids = np.asarray(run_ids, dtype=int)
    return float(
        np.mean([values[run_ids == run].mean() for run in np.unique(run_ids)])
    )


def _score_against_prototypes(
    patterns: np.ndarray,
    labels: np.ndarray,
    run_ids: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    labels = np.asarray(labels, dtype=int)
    run_ids = np.asarray(run_ids, dtype=int)
    correlations = _pattern_correlations(patterns, prototypes)
    corr_odor1 = correlations[:, 0]
    corr_odor2 = correlations[:, 1]
    projection = corr_odor2 - corr_odor1
    matching = np.where(labels == 0, corr_odor1, corr_odor2)
    nonmatching = np.where(labels == 0, corr_odor2, corr_odor1)
    advantage = matching - nonmatching
    oriented_projection = (2 * labels - 1) * projection
    if not np.allclose(advantage, oriented_projection, atol=1e-12):
        raise RuntimeError("Prototype advantage and odor-axis projection disagree.")

    clipped = np.clip(correlations, -1 + 1e-7, 1 - 1e-7)
    fisher = np.arctanh(clipped)
    fisher_matching = np.where(labels == 0, fisher[:, 0], fisher[:, 1])
    fisher_nonmatching = np.where(labels == 0, fisher[:, 1], fisher[:, 0])
    fisher_advantage = fisher_matching - fisher_nonmatching

    predictions = (corr_odor2 > corr_odor1).astype(int)
    trial_metrics = pd.DataFrame(
        {
            "expected_odor_label": labels,
            "expected_odor": labels + 1,
            "corr_odor1_prototype": corr_odor1,
            "corr_odor2_prototype": corr_odor2,
            "matching_correlation": matching,
            "nonmatching_correlation": nonmatching,
            "matching_correlation_advantage": advantage,
            "fisher_z_matching_advantage": fisher_advantage,
            "odor2_minus_odor1_axis_projection": projection,
            "oriented_axis_projection": oriented_projection,
            "nearest_prototype_prediction": predictions,
        }
    )

    summary = {
        "mean_matching_correlation": _run_and_label_balanced_mean(
            matching, labels, run_ids
        ),
        "mean_nonmatching_correlation": _run_and_label_balanced_mean(
            nonmatching, labels, run_ids
        ),
        "matching_correlation_advantage": _run_and_label_balanced_mean(
            advantage, labels, run_ids
        ),
        "fisher_z_matching_advantage": _run_and_label_balanced_mean(
            fisher_advantage, labels, run_ids
        ),
        "odor1_matching_correlation_advantage": _run_balanced_mean(
            advantage[labels == 0],
            run_ids[labels == 0],
        ),
        "odor2_matching_correlation_advantage": _run_balanced_mean(
            advantage[labels == 1],
            run_ids[labels == 1],
        ),
        "cue_nearest_prototype_accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "cue_nearest_prototype_balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "cue_nearest_prototype_auc": float(
            roc_auc_score(labels, projection)
        ),
    }
    return trial_metrics, summary


def _passive_leave_one_run_out_metrics(
    patterns: np.ndarray,
    labels: np.ndarray,
    run_ids: np.ndarray,
) -> dict[str, float]:
    trial_frames = []
    for held_out_run in np.unique(run_ids):
        train = run_ids != held_out_run
        test = ~train
        prototypes, _ = _build_run_balanced_prototypes(
            patterns[train], labels[train], run_ids[train]
        )
        trial_metrics, _ = _score_against_prototypes(
            patterns[test], labels[test], run_ids[test], prototypes
        )
        trial_metrics["run"] = run_ids[test]
        trial_metrics["label"] = labels[test]
        trial_frames.append(trial_metrics)

    trials = pd.concat(trial_frames, ignore_index=True)
    labels_all = trials["label"].to_numpy(dtype=int)
    runs_all = trials["run"].to_numpy(dtype=int)
    predictions = trials["nearest_prototype_prediction"].to_numpy(dtype=int)
    projection = trials["odor2_minus_odor1_axis_projection"].to_numpy()
    advantage = trials["matching_correlation_advantage"].to_numpy()
    return {
        "passive_looro_matching_correlation_advantage": (
            _run_and_label_balanced_mean(advantage, labels_all, runs_all)
        ),
        "passive_looro_accuracy": float(
            accuracy_score(labels_all, predictions)
        ),
        "passive_looro_balanced_accuracy": float(
            balanced_accuracy_score(labels_all, predictions)
        ),
        "passive_looro_auc": float(roc_auc_score(labels_all, projection)),
    }


def _axis_run_similarity(run_prototypes: dict[int, np.ndarray]) -> float:
    runs = sorted(run_prototypes)
    axes = np.vstack(
        [
            run_prototypes[run][1] - run_prototypes[run][0]
            for run in runs
        ]
    )
    normalized_axes = _center_and_unit_normalize(axes)
    similarities = [
        float(normalized_axes[first] @ normalized_axes[second])
        for first, second in combinations(range(len(runs)), 2)
    ]
    if not similarities:
        return np.nan
    return float(np.mean(similarities))


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
                diagnostics_dir, sub, cue_subset
            )
        )
        labels, label_source = decoding._cue_labels(
            sub, samples, sub14_original_events_dir
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
        _,
        passive_affine,
    ) = decoding._load_passive_odor_patterns(
        base_dir, sub, odor_runs, reference_voxel_indices
    )

    hemisphere_mask, midline_mask = decoding._hemisphere_voxel_mask(
        reference_voxel_indices, passive_affine, hemisphere
    )
    passive_selected = passive_raw[:, hemisphere_mask]
    cue_selected = {
        cue: values["raw"][:, hemisphere_mask]
        for cue, values in cue_inputs.items()
    }
    finite_voxels = np.isfinite(passive_selected).all(axis=0)
    for patterns in cue_selected.values():
        finite_voxels &= np.isfinite(patterns).all(axis=0)
    if not finite_voxels.any():
        raise ValueError("No jointly finite passive/cue ROI voxels remain.")

    passive_used = passive_selected[:, finite_voxels]
    passive_patterns = decoding._runwise_zscore(
        passive_used, passive_runs
    )
    prototypes, run_prototypes = _build_run_balanced_prototypes(
        passive_patterns, passive_labels, passive_runs
    )
    passive_metrics = _passive_leave_one_run_out_metrics(
        passive_patterns, passive_labels, passive_runs
    )
    prototype_correlation = float(
        _pattern_correlations(prototypes[:1], prototypes)[0, 1]
    )
    axis_similarity = _axis_run_similarity(run_prototypes)

    n_voxels_bilateral = int(len(hemisphere_mask))
    n_voxels_raw = int(hemisphere_mask.sum())
    n_voxels_used = int(finite_voxels.sum())
    rows = []
    trial_outputs = {}
    for cue_subset in cue_subsets:
        values = cue_inputs[cue_subset]
        samples = values["samples"].copy().reset_index(drop=True)
        labels = values["labels"]
        cue_runs = samples["run"].astype(int).to_numpy()
        cue_patterns = decoding._runwise_zscore(
            cue_selected[cue_subset][:, finite_voxels], cue_runs
        )
        trial_metrics, cue_metrics = _score_against_prototypes(
            cue_patterns, labels, cue_runs, prototypes
        )
        trial_output = pd.concat([samples, trial_metrics], axis=1)
        trial_output["roi"] = roi_name
        trial_output["hemisphere"] = hemisphere
        trial_output["cue_subset"] = cue_subset
        trial_output["cue_label_source"] = values["label_source"]
        trial_outputs[cue_subset] = trial_output

        row = {
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
            "normalization": "within_task_run_voxelwise_zscore",
            "prototype_weighting": "equal_passive_run_weight",
            "primary_similarity": "pearson_correlation",
            "n_voxels_bilateral_raw": n_voxels_bilateral,
            "n_voxels_midline": int(midline_mask.sum()),
            "n_voxels_excluded_by_hemisphere": (
                n_voxels_bilateral - n_voxels_raw
            ),
            "n_voxels_raw": n_voxels_raw,
            "n_voxels_used": n_voxels_used,
            "n_voxels_dropped_nonfinite": n_voxels_raw - n_voxels_used,
            "n_passive_samples": int(len(passive_labels)),
            "n_passive_odor1": int((passive_labels == 0).sum()),
            "n_passive_odor2": int((passive_labels == 1).sum()),
            "n_passive_runs": int(len(np.unique(passive_runs))),
            "passive_runs": ",".join(map(str, sorted(np.unique(passive_runs)))),
            "n_cue_samples": int(len(labels)),
            "n_cue_odor1": int((labels == 0).sum()),
            "n_cue_odor2": int((labels == 1).sum()),
            "n_cue_runs": int(len(np.unique(cue_runs))),
            "cue_runs": ",".join(map(str, sorted(np.unique(cue_runs)))),
            "passive_prototype_correlation": prototype_correlation,
            "passive_axis_run_similarity": axis_similarity,
            **passive_metrics,
            **cue_metrics,
        }
        rows.append(row)

    return rows, trial_outputs


def _scope_output_dir(
    output_root: Path,
    roi_name: str,
    hemisphere: str,
) -> Path:
    return output_root / f"roi-{roi_name}_{hemisphere}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build passive odor prototypes and quantify their alignment "
            "with conditioned and preconditioned first-image cue patterns."
        )
    )
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--diagnostics_dir", type=Path, required=True)
    parser.add_argument("--roi_name", required=True)
    parser.add_argument(
        "--hemispheres",
        nargs="+",
        choices=HEMISPHERES,
        default=list(HEMISPHERES),
        help="ROI scopes. Default: bilateral left right.",
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=CUE_SUBSETS,
        default=list(CUE_SUBSETS),
        help="Cue subsets analyzed separately. Default: both.",
    )
    parser.add_argument(
        "--odor_runs",
        nargs="+",
        type=int,
        default=list(DEFAULT_ODOR_RUNS),
    )
    parser.add_argument("--sub14_original_events_dir", type=Path)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help=(
            "Default: <project_root>/roi_template_alignment/"
            "passive_odor_to_premeal_cues."
        ),
    )
    parser.add_argument(
        "--skip_trial_outputs",
        action="store_true",
        help="Do not save trial-level matching correlations.",
    )
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    diagnostics_dir = args.diagnostics_dir.expanduser().resolve()
    roi_name = decoding._sanitize_label(args.roi_name)
    project_root = SCRIPT_DIR.parents[1]
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "roi_template_alignment"
        / "passive_odor_to_premeal_cues"
    )
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
    print("Passive odor -> premeal cue ROI template alignment")
    print("=" * 80)
    print(f"SPCD data root:       {base_dir}")
    print(f"RSA diagnostics:      {diagnostics_dir}")
    print(f"ROI:                  {roi_name}")
    print(f"Hemispheres:          {', '.join(args.hemispheres)}")
    print(f"Cue subsets:          {', '.join(cue_subsets)} (separate)")
    print(f"Subjects:             {len(subjects)}")
    print(f"Output root:          {output_root}")
    print("Primary metric:       matching correlation - nonmatching correlation")
    print("=" * 80)

    for hemisphere in dict.fromkeys(args.hemispheres):
        scope_dir = _scope_output_dir(output_root, roi_name, hemisphere)
        rows = []
        failures = []
        for sub in subjects:
            try:
                subject_rows, trial_outputs = _analyze_subject(
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
                if not args.skip_trial_outputs:
                    subject_dir = scope_dir / "trial_alignment" / sub
                    subject_dir.mkdir(parents=True, exist_ok=True)
                    for cue_subset, trial_df in trial_outputs.items():
                        trial_df.to_csv(
                            subject_dir / f"{cue_subset}.csv", index=False
                        )
                summary_text = ", ".join(
                    f"{row['cue_subset']}={row['matching_correlation_advantage']:.5f}"
                    for row in subject_rows
                )
                print(f"[OK] {sub} {hemisphere}: {summary_text}")
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
        subject_csv = scope_dir / "subject_template_alignment.csv"
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
