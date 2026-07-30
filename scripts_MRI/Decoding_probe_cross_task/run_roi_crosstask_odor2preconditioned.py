#!/usr/bin/env python3
"""Train an odor classifier on passive smelling and test premeal cue patterns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.svm import SVC


ODOR1_CONDITIONS = {"A11", "A12", "B11", "B12"}
ODOR2_CONDITIONS = {"A21", "A22", "B21", "B22"}
CUE_SUBSET_FILES = {
    "preconditioned_odor_only": "preconditioned_odor_only",
    "conditioned_odor_only": "conditioned_odor_only",
}
DEFAULT_ODOR_RUNS = (1, 2)
SUB14 = "sub-14"


def _sanitize_label(label: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in label
    )


def _discover_subjects(diagnostics_dir: Path) -> list[str]:
    if not diagnostics_dir.is_dir():
        raise FileNotFoundError(
            f"RSA diagnostics directory not found: {diagnostics_dir}"
        )
    subjects = sorted(
        path.name
        for path in diagnostics_dir.iterdir()
        if path.is_dir() and path.name.startswith("sub-")
    )
    if not subjects:
        raise RuntimeError(
            f"No subject diagnostic directories found under: {diagnostics_dir}"
        )
    return subjects


def _load_cue_diagnostics(
    diagnostics_dir: Path,
    sub: str,
    cue_subset: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, Path]:
    stem = CUE_SUBSET_FILES[cue_subset]
    subject_dir = diagnostics_dir / sub
    pattern_path = subject_dir / f"{stem}_roi_patterns.npz"
    sample_path = subject_dir / f"{stem}_samples.csv"

    if not pattern_path.is_file():
        raise FileNotFoundError(f"Cue ROI patterns not found: {pattern_path}")
    if not sample_path.is_file():
        raise FileNotFoundError(f"Cue sample metadata not found: {sample_path}")

    with np.load(pattern_path, allow_pickle=False) as diagnostic:
        required = {"raw_patterns", "run_ids", "conditions"}
        missing = required - set(diagnostic.files)
        if missing:
            raise ValueError(
                f"Diagnostic NPZ missing keys {sorted(missing)}: {pattern_path}"
            )

        patterns = np.asarray(diagnostic["raw_patterns"], dtype=float)
        run_ids = np.asarray(diagnostic["run_ids"], dtype=int)
        conditions = np.asarray(diagnostic["conditions"], dtype=str)
        if "raw_roi_voxel_indices" in diagnostic.files:
            voxel_indices = np.asarray(
                diagnostic["raw_roi_voxel_indices"],
                dtype=int,
            )
        elif "roi_voxel_indices" in diagnostic.files:
            voxel_indices = np.asarray(
                diagnostic["roi_voxel_indices"],
                dtype=int,
            )
        else:
            raise ValueError(
                "Diagnostic NPZ has no raw ROI voxel coordinates: "
                f"{pattern_path}"
            )

    samples = pd.read_csv(sample_path)
    required_columns = {
        "sample_index",
        "sub",
        "run",
        "trial",
        "img_index",
        "condition",
    }
    missing_columns = required_columns - set(samples.columns)
    if missing_columns:
        raise ValueError(
            f"Cue sample CSV missing columns {sorted(missing_columns)}: "
            f"{sample_path}"
        )
    samples = samples.sort_values("sample_index").reset_index(drop=True)

    n_samples = patterns.shape[0]
    if patterns.ndim != 2:
        raise ValueError(
            f"Expected 2D cue pattern matrix, got {patterns.shape}: "
            f"{pattern_path}"
        )
    if voxel_indices.shape != (patterns.shape[1], 3):
        raise ValueError(
            "Raw ROI coordinates do not match cue pattern columns: "
            f"patterns={patterns.shape}, coordinates={voxel_indices.shape}"
        )
    if len(samples) != n_samples or len(run_ids) != n_samples:
        raise ValueError(
            "Cue diagnostics have inconsistent sample counts: "
            f"patterns={n_samples}, CSV={len(samples)}, runs={len(run_ids)}"
        )
    if not np.array_equal(
        samples["condition"].astype(str).to_numpy(),
        conditions,
    ):
        raise ValueError(
            f"Cue conditions disagree between NPZ and CSV: {subject_dir}"
        )
    if not np.array_equal(samples["run"].astype(int).to_numpy(), run_ids):
        raise ValueError(
            f"Cue run IDs disagree between NPZ and CSV: {subject_dir}"
        )
    if set(samples["sub"].astype(str)) != {sub}:
        raise ValueError(
            f"Cue sample CSV contains an unexpected subject: {sample_path}"
        )
    if not np.all(samples["img_index"].astype(int).to_numpy() == 1):
        raise ValueError(
            f"Cue diagnostics are not first-image-only: {sample_path}"
        )

    return patterns, voxel_indices, samples, pattern_path


def _labels_from_conditions(conditions: pd.Series) -> np.ndarray:
    labels = []
    for condition in conditions.astype(str):
        if condition in ODOR1_CONDITIONS:
            labels.append(0)
        elif condition in ODOR2_CONDITIONS:
            labels.append(1)
        else:
            raise ValueError(
                f"Condition is not an odor-predictive A/B cue: {condition}"
            )
    return np.asarray(labels, dtype=int)


def _load_sub14_original_cue_labels(
    samples: pd.DataFrame,
    original_events_dir: Path,
) -> np.ndarray:
    """Match sub-14 cue samples to the pre-correction premeal event files."""
    event_rows = []
    for run in sorted(samples["run"].astype(int).unique()):
        event_path = (
            original_events_dir
            / f"{SUB14}_task-premeal_run-{run}_events_orig.tsv"
        )
        if not event_path.is_file():
            raise FileNotFoundError(
                f"Original sub-14 premeal events not found: {event_path}"
            )
        events = pd.read_csv(event_path, sep="\t")
        required = {
            "trial_type",
            "run",
            "trial",
            "img_index",
            "cue_index",
            "odor_index",
        }
        missing = required - set(events.columns)
        if missing:
            raise ValueError(
                f"Original sub-14 TSV missing columns {sorted(missing)}: "
                f"{event_path}"
            )
        events = events.loc[
            events["trial_type"].isin(["cue_img1", "cue_img2"]),
            ["run", "trial", "img_index", "cue_index", "odor_index"],
        ].copy()
        event_rows.append(events)

    originals = pd.concat(event_rows, ignore_index=True)
    key = ["run", "trial", "img_index"]
    if originals.duplicated(key).any():
        raise ValueError(
            "Original sub-14 premeal TSVs contain duplicate cue keys "
            "(run, trial, img_index)."
        )

    sample_keys = samples.loc[:, key].copy()
    sample_keys["_sample_order"] = np.arange(len(sample_keys))
    matched = sample_keys.merge(
        originals,
        on=key,
        how="left",
        validate="many_to_one",
    ).sort_values("_sample_order")

    if matched["odor_index"].isna().any():
        missing_rows = matched.loc[matched["odor_index"].isna(), key]
        raise ValueError(
            "Could not match cue samples to original sub-14 events:\n"
            + missing_rows.to_string(index=False)
        )

    odor_index = matched["odor_index"].astype(int).to_numpy()
    if not np.all(np.isin(odor_index, [1, 2])):
        bad = sorted(set(odor_index) - {1, 2})
        raise ValueError(
            "Original sub-14 labels include non-odor outcomes for selected "
            f"cue samples: {bad}"
        )
    return odor_index - 1


def _cue_labels(
    sub: str,
    samples: pd.DataFrame,
    sub14_original_events_dir: Path,
) -> tuple[np.ndarray, str]:
    if sub == SUB14:
        return (
            _load_sub14_original_cue_labels(
                samples,
                sub14_original_events_dir,
            ),
            "original_premeal_events",
        )
    return _labels_from_conditions(samples["condition"]), "rsa_condition"


def _passive_event_path(base_dir: Path, sub: str, run: int) -> Path:
    return (
        base_dir
        / "nii"
        / sub
        / "func_PAID"
        / "odor_scan"
        / f"{sub}_task-odor_run-{run}_events.tsv"
    )


def _passive_beta_path(
    base_dir: Path,
    sub: str,
    run: int,
    trial_number: int,
) -> Path:
    stem = (
        base_dir
        / "GLM"
        / "odor_scan_trialwise_2mm"
        / sub
        / f"beta_run{run}_trial{trial_number:02d}"
    )
    for suffix in (".nii", ".nii.gz"):
        candidate = Path(str(stem) + suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Passive beta not found with .nii or .nii.gz suffix: {stem}"
    )


def _extract_voxels(
    image_path: Path,
    voxel_indices: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray]:
    image = nib.load(str(image_path))
    shape = tuple(int(value) for value in image.shape[:3])
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D beta image: {image_path}")
    if np.any(voxel_indices < 0):
        raise ValueError("ROI diagnostics contain negative voxel coordinates.")
    upper = np.asarray(shape, dtype=int)
    if np.any(voxel_indices >= upper):
        raise ValueError(
            "ROI coordinates exceed passive beta dimensions: "
            f"coordinates max={voxel_indices.max(axis=0).tolist()}, "
            f"beta shape={shape}, beta={image_path}"
        )
    data = np.asanyarray(image.dataobj, dtype=float)
    values = data[
        voxel_indices[:, 0],
        voxel_indices[:, 1],
        voxel_indices[:, 2],
    ]
    return values, shape, np.asarray(image.affine, dtype=float)


def _load_passive_odor_patterns(
    base_dir: Path,
    sub: str,
    runs: tuple[int, ...],
    voxel_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    labels = []
    run_ids = []
    metadata = []
    reference_shape = None
    reference_affine = None

    for run in runs:
        event_path = _passive_event_path(base_dir, sub, run)
        if not event_path.is_file():
            raise FileNotFoundError(f"Passive events not found: {event_path}")
        events = pd.read_csv(event_path, sep="\t")
        if "trial_type" not in events.columns:
            raise ValueError(
                f"Passive events TSV has no trial_type column: {event_path}"
            )

        # Beta numbering follows every row in the unfiltered passive design.
        events = events.copy()
        events["_beta_trial_number"] = np.arange(1, len(events) + 1)
        odor_events = events.loc[
            events["trial_type"].isin(["odor_1", "odor_2"])
        ].copy()
        if odor_events.empty:
            raise ValueError(f"No odor_1/odor_2 events found: {event_path}")

        for _, event in odor_events.iterrows():
            beta_trial = int(event["_beta_trial_number"])
            beta_path = _passive_beta_path(
                base_dir,
                sub,
                run,
                beta_trial,
            )
            values, shape, affine = _extract_voxels(
                beta_path,
                voxel_indices,
            )
            if reference_shape is None:
                reference_shape = shape
                reference_affine = affine
            elif shape != reference_shape or not np.allclose(
                affine,
                reference_affine,
                atol=1e-5,
            ):
                raise ValueError(
                    "Passive beta images do not share one image space: "
                    f"{beta_path}"
                )

            trial_type = str(event["trial_type"])
            label = 0 if trial_type == "odor_1" else 1
            rows.append(values)
            labels.append(label)
            run_ids.append(run)
            metadata.append(
                {
                    "run": run,
                    "beta_trial_number": beta_trial,
                    "trial_type": trial_type,
                    "label": label,
                    "beta_path": str(beta_path),
                }
            )

    patterns = np.vstack(rows)
    y = np.asarray(labels, dtype=int)
    groups = np.asarray(run_ids, dtype=int)
    if set(y) != {0, 1}:
        raise ValueError(
            f"Passive training data do not contain both odor classes: {sub}"
        )
    if len(np.unique(groups)) < 2:
        raise ValueError(
            f"Passive training data require at least two runs: {sub}"
        )
    return patterns, y, groups, pd.DataFrame(metadata)


def _runwise_zscore(
    patterns: np.ndarray,
    run_ids: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(patterns, dtype=float).copy()
    for run in np.unique(run_ids):
        run_mask = run_ids == run
        if int(run_mask.sum()) < 2:
            raise ValueError(
                f"Need at least two samples to normalize run {run}."
            )
        run_patterns = normalized[run_mask]
        mean = run_patterns.mean(axis=0, keepdims=True)
        std = run_patterns.std(axis=0, ddof=0, keepdims=True)
        std[std < 1e-8] = 1.0
        normalized[run_mask] = (run_patterns - mean) / std
    return normalized


def _drop_nonfinite_voxels(
    train_patterns: np.ndarray,
    test_patterns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(train_patterns).all(axis=0)
    finite &= np.isfinite(test_patterns).all(axis=0)
    if not finite.any():
        raise ValueError(
            "No ROI voxels remain after joint passive/cue finite-value QC."
        )
    return train_patterns[:, finite], test_patterns[:, finite], finite


def _decision_distance(model: SVC, patterns: np.ndarray) -> np.ndarray:
    decision = np.asarray(model.decision_function(patterns), dtype=float)
    weight_norm = float(np.linalg.norm(model.coef_))
    if not np.isfinite(weight_norm) or weight_norm <= 0:
        raise ValueError("Linear SVM has a zero or nonfinite weight norm.")
    return decision / weight_norm


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_distance: np.ndarray,
) -> dict[str, float]:
    signed_distance = (2 * y_true - 1) * decision_distance
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "auc": float(roc_auc_score(y_true, decision_distance)),
        "mean_signed_distance": float(np.mean(signed_distance)),
    }


def _passive_leave_one_run_out(
    patterns: np.ndarray,
    labels: np.ndarray,
    run_ids: np.ndarray,
    svm_c: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    predictions = np.empty_like(labels)
    distances = np.empty(labels.shape, dtype=float)

    for held_out_run in np.unique(run_ids):
        test_mask = run_ids == held_out_run
        train_mask = ~test_mask
        if set(labels[train_mask]) != {0, 1}:
            raise ValueError(
                "A passive training fold does not contain both odor classes: "
                f"held-out run {held_out_run}"
            )
        model = SVC(kernel="linear", C=svm_c)
        model.fit(patterns[train_mask], labels[train_mask])
        predictions[test_mask] = model.predict(patterns[test_mask])
        distances[test_mask] = _decision_distance(
            model,
            patterns[test_mask],
        )

    return (
        _binary_metrics(labels, predictions, distances),
        predictions,
        distances,
    )


def _analyze_subject(
    *,
    base_dir: Path,
    diagnostics_dir: Path,
    sub: str,
    roi_name: str,
    cue_subset: str,
    odor_runs: tuple[int, ...],
    svm_c: float,
    sub14_original_events_dir: Path,
    prediction_dir: Path | None,
) -> dict:
    (
        cue_raw,
        voxel_indices,
        cue_samples,
        pattern_path,
    ) = _load_cue_diagnostics(
        diagnostics_dir,
        sub,
        cue_subset,
    )
    cue_y, cue_label_source = _cue_labels(
        sub,
        cue_samples,
        sub14_original_events_dir,
    )
    passive_raw, passive_y, passive_runs, passive_meta = (
        _load_passive_odor_patterns(
            base_dir,
            sub,
            odor_runs,
            voxel_indices,
        )
    )
    cue_runs = cue_samples["run"].astype(int).to_numpy()

    passive_used, cue_used, finite_voxel_mask = _drop_nonfinite_voxels(
        passive_raw,
        cue_raw,
    )
    passive_patterns = _runwise_zscore(passive_used, passive_runs)
    cue_patterns = _runwise_zscore(cue_used, cue_runs)

    passive_metrics, passive_pred, passive_distance = (
        _passive_leave_one_run_out(
            passive_patterns,
            passive_y,
            passive_runs,
            svm_c,
        )
    )

    model = SVC(kernel="linear", C=svm_c)
    model.fit(passive_patterns, passive_y)
    cue_pred = model.predict(cue_patterns)
    cue_distance = _decision_distance(model, cue_patterns)
    cue_metrics = _binary_metrics(cue_y, cue_pred, cue_distance)

    if prediction_dir is not None:
        subject_dir = prediction_dir / sub
        subject_dir.mkdir(parents=True, exist_ok=True)

        cue_predictions = cue_samples.copy()
        cue_predictions["true_label"] = cue_y
        cue_predictions["predicted_label"] = cue_pred
        cue_predictions["decision_distance_odor2"] = cue_distance
        cue_predictions["signed_distance"] = (
            (2 * cue_y - 1) * cue_distance
        )
        cue_predictions["label_source"] = cue_label_source
        cue_predictions.to_csv(
            subject_dir / f"{cue_subset}_cue_predictions.csv",
            index=False,
        )

        passive_predictions = passive_meta.copy()
        passive_predictions["cv_predicted_label"] = passive_pred
        passive_predictions["cv_decision_distance_odor2"] = (
            passive_distance
        )
        passive_predictions["cv_signed_distance"] = (
            (2 * passive_y - 1) * passive_distance
        )
        passive_predictions.to_csv(
            subject_dir / f"{cue_subset}_passive_cv_predictions.csv",
            index=False,
        )

    n_voxels_raw = int(cue_raw.shape[1])
    n_voxels_used = int(finite_voxel_mask.sum())
    row = {
        "sub": sub,
        "roi": roi_name,
        "cue_subset": cue_subset,
        "cue_label_source": cue_label_source,
        "cue_diagnostic_npz": str(pattern_path),
        "svm_kernel": "linear",
        "svm_c": svm_c,
        "normalization": "within_task_run_voxelwise_zscore",
        "n_voxels_raw": n_voxels_raw,
        "n_voxels_used": n_voxels_used,
        "n_voxels_dropped_nonfinite": n_voxels_raw - n_voxels_used,
        "n_passive_samples": int(len(passive_y)),
        "n_passive_odor1": int((passive_y == 0).sum()),
        "n_passive_odor2": int((passive_y == 1).sum()),
        "n_passive_runs": int(len(np.unique(passive_runs))),
        "passive_runs": ",".join(
            str(run) for run in sorted(np.unique(passive_runs))
        ),
        "n_cue_samples": int(len(cue_y)),
        "n_cue_odor1": int((cue_y == 0).sum()),
        "n_cue_odor2": int((cue_y == 1).sum()),
        "n_cue_runs": int(len(np.unique(cue_runs))),
        "cue_runs": ",".join(
            str(run) for run in sorted(np.unique(cue_runs))
        ),
        "passive_cv_accuracy": passive_metrics["accuracy"],
        "passive_cv_balanced_accuracy": passive_metrics[
            "balanced_accuracy"
        ],
        "passive_cv_auc": passive_metrics["auc"],
        "passive_cv_mean_signed_distance": passive_metrics[
            "mean_signed_distance"
        ],
        "cue_accuracy": cue_metrics["accuracy"],
        "cue_balanced_accuracy": cue_metrics["balanced_accuracy"],
        "cue_auc": cue_metrics["auc"],
        "cue_mean_signed_distance": cue_metrics["mean_signed_distance"],
        "n_support_odor1": int(model.n_support_[0]),
        "n_support_odor2": int(model.n_support_[1]),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a linear SVM on passive odor-scan ROI patterns and test "
            "the exact first-image cue patterns saved by the ROI RSA."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="SPCD data root containing nii/ and GLM/.",
    )
    parser.add_argument(
        "--diagnostics_dir",
        type=Path,
        required=True,
        help=(
            "One ROI's RSA diagnostics directory containing sub-XX folders."
        ),
    )
    parser.add_argument(
        "--roi_name",
        required=True,
        help="ROI label written to output filenames and rows.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs. Default: discover from diagnostics_dir.",
    )
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=sorted(CUE_SUBSET_FILES),
        default=["preconditioned_odor_only"],
        help=(
            "Saved RSA cue subsets to test. Default: "
            "preconditioned_odor_only."
        ),
    )
    parser.add_argument(
        "--odor_runs",
        nargs="+",
        type=int,
        default=list(DEFAULT_ODOR_RUNS),
        help="Passive odor runs used for training. Default: 1 2.",
    )
    parser.add_argument(
        "--svm_c",
        type=float,
        default=1.0,
        help="Fixed linear SVM C. Default: 1.0.",
    )
    parser.add_argument(
        "--sub14_original_events_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing sub-14 *_events_orig.tsv files. Default: "
            "<base_dir>/nii/sub-14/func_PAID/premeal."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Subject-results CSV. Default: <base_dir>/roi_decoding/"
            "probe_crosstask_odor2preconditioned/roi-<name>_subject_results.csv."
        ),
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save trial-level cue and passive-CV prediction CSVs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first failed subject/subset instead of recording it.",
    )
    args = parser.parse_args()

    if args.svm_c <= 0:
        raise ValueError("--svm_c must be positive.")
    odor_runs = tuple(dict.fromkeys(args.odor_runs))
    if len(odor_runs) < 2:
        raise ValueError("--odor_runs must contain at least two runs.")

    base_dir = args.base_dir.expanduser().resolve()
    diagnostics_dir = args.diagnostics_dir.expanduser().resolve()
    roi_name = _sanitize_label(args.roi_name)
    sub14_original_events_dir = (
        args.sub14_original_events_dir.expanduser().resolve()
        if args.sub14_original_events_dir is not None
        else base_dir / "nii" / SUB14 / "func_PAID" / "premeal"
    )
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else (
            base_dir
            / "roi_decoding"
            / "probe_crosstask_odor2preconditioned"
            / f"roi-{roi_name}_subject_results.csv"
        )
    )
    prediction_dir = (
        output_csv.parent / f"roi-{roi_name}_predictions"
        if args.save_predictions
        else None
    )

    subjects = (
        args.subjects
        if args.subjects is not None
        else _discover_subjects(diagnostics_dir)
    )
    subjects = list(dict.fromkeys(subjects))

    print("=" * 80)
    print("ROI passive-odor -> premeal-cue cross-task decoding")
    print("=" * 80)
    print(f"SPCD data root:        {base_dir}")
    print(f"RSA diagnostics:       {diagnostics_dir}")
    print(f"ROI:                   {roi_name}")
    print(f"Subjects:              {len(subjects)}")
    print(f"Cue subsets:           {', '.join(args.cue_subsets)}")
    print(f"Passive runs:          {', '.join(map(str, odor_runs))}")
    print(f"Classifier:            linear SVM, C={args.svm_c:g}")
    print("Normalization:         separate within-run voxelwise z-score")
    print(f"Sub-14 original TSVs:  {sub14_original_events_dir}")
    print(f"Output CSV:            {output_csv}")
    print("=" * 80)

    rows = []
    failures = []
    for sub in subjects:
        for cue_subset in args.cue_subsets:
            try:
                row = _analyze_subject(
                    base_dir=base_dir,
                    diagnostics_dir=diagnostics_dir,
                    sub=sub,
                    roi_name=roi_name,
                    cue_subset=cue_subset,
                    odor_runs=odor_runs,
                    svm_c=args.svm_c,
                    sub14_original_events_dir=sub14_original_events_dir,
                    prediction_dir=prediction_dir,
                )
                rows.append(row)
                print(
                    f"[OK] {sub} {cue_subset}: "
                    f"cue balanced accuracy="
                    f"{row['cue_balanced_accuracy']:.3f}, "
                    f"passive CV="
                    f"{row['passive_cv_balanced_accuracy']:.3f}, "
                    f"voxels={row['n_voxels_used']}/"
                    f"{row['n_voxels_raw']}"
                )
            except Exception as exc:
                if args.strict:
                    raise
                failures.append(
                    {
                        "sub": sub,
                        "roi": roi_name,
                        "cue_subset": cue_subset,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"[FAIL] {sub} {cue_subset}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    if not rows:
        raise RuntimeError("No subject analyses completed successfully.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved subject results: {output_csv}")

    failure_csv = output_csv.with_name(
        output_csv.stem + "_failures.csv"
    )
    failure_columns = [
        "sub",
        "roi",
        "cue_subset",
        "error_type",
        "error",
    ]
    pd.DataFrame(failures, columns=failure_columns).to_csv(
        failure_csv,
        index=False,
    )
    print(f"Saved failures ({len(failures)}): {failure_csv}")


if __name__ == "__main__":
    main()
