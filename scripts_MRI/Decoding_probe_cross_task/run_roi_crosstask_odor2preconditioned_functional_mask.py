#!/usr/bin/env python3
"""Decode preconditioned premeal cues in one group functional mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.svm import SVC

import run_roi_crosstask_odor2preconditioned as established_decoder


CUE_SUBSET = "preconditioned_odor_only"
PRECONDITIONED_CONDITIONS = {
    1: "A11",
    3: "A21",
    7: "A12",
    9: "A22",
}
DEFAULT_ODOR_RUNS = (1, 2)
DEFAULT_PROBE_RUNS = (1, 2, 3)
SUB14 = "sub-14"


def _sanitize_label(label: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in label
    )


def _neuroimaging_stem(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".hdr", ".img"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _discover_subjects(base_dir: Path) -> list[str]:
    glm_dir = base_dir / "GLM" / "premeal_trialwise_2mm"
    if not glm_dir.is_dir():
        raise FileNotFoundError(f"Premeal GLM directory not found: {glm_dir}")
    subjects = sorted(
        path.name
        for path in glm_dir.iterdir()
        if path.is_dir() and path.name.startswith("sub-")
    )
    if not subjects:
        raise RuntimeError(f"No subject directories found under: {glm_dir}")
    return subjects


def _load_functional_mask(
    mask_path: Path,
) -> tuple[np.ndarray, tuple[int, int, int], np.ndarray]:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Functional mask not found: {mask_path}")
    image = nib.load(str(mask_path))
    if image.ndim != 3:
        raise ValueError(
            f"Functional mask must be 3D, got {image.shape}: {mask_path}"
        )
    data = np.asanyarray(image.dataobj, dtype=float)
    mask = np.isfinite(data) & (data > 0)
    if not mask.any():
        raise ValueError(
            f"Functional mask has no finite positive voxels: {mask_path}"
        )
    voxel_indices = np.column_stack(np.where(mask)).astype(int)
    shape = tuple(int(value) for value in image.shape)
    affine = np.asarray(image.affine, dtype=float)
    return voxel_indices, shape, affine


def _validate_image_space(
    image,
    *,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
    image_path: Path,
) -> None:
    shape = tuple(int(value) for value in image.shape[:3])
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D beta image: {image_path}")
    if shape != reference_shape or not np.allclose(
        image.affine,
        reference_affine,
        atol=1e-5,
    ):
        raise ValueError(
            "Functional mask and beta image are not in the same space: "
            f"mask shape={reference_shape}, beta shape={shape}, beta={image_path}"
        )


def _extract_mask_values(
    image_path: Path,
    voxel_indices: np.ndarray,
    reference_shape: tuple[int, int, int],
    reference_affine: np.ndarray,
) -> np.ndarray:
    if not image_path.is_file():
        raise FileNotFoundError(f"Probe beta not found: {image_path}")
    image = nib.load(str(image_path))
    _validate_image_space(
        image,
        reference_shape=reference_shape,
        reference_affine=reference_affine,
        image_path=image_path,
    )
    data = np.asanyarray(image.dataobj, dtype=float)
    return data[
        voxel_indices[:, 0],
        voxel_indices[:, 1],
        voxel_indices[:, 2],
    ]


def _probe_beta_index_path(base_dir: Path, sub: str, run: int) -> Path:
    return (
        base_dir
        / "GLM"
        / "premeal_trialwise_2mm"
        / sub
        / f"{sub}_premeal_run-{run}_beta_index.csv"
    )


def _selected_probe_rows(index_path: Path, sub: str, run: int) -> pd.DataFrame:
    table = pd.read_csv(index_path)
    required = {"beta_file", "run", "trial", "img_index", "cue_index"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Probe beta index missing columns {sorted(missing)}: {index_path}"
        )

    numeric_columns = ("run", "trial", "img_index", "cue_index")
    converted = {}
    for column in numeric_columns:
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Probe beta index has invalid {column} values: {index_path}"
            )
        converted[column] = values.astype(int)

    table = table.copy()
    for column, values in converted.items():
        table[column] = values
    if set(table["run"]) != {run}:
        raise ValueError(
            f"Probe beta index run column does not equal run {run}: {index_path}"
        )

    selected = table.loc[
        (table["img_index"] == 1)
        & table["cue_index"].isin(PRECONDITIONED_CONDITIONS)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No first-image preconditioned cues found: {index_path}"
        )
    if "trial_type" in selected.columns:
        unexpected = sorted(
            set(selected["trial_type"].astype(str)) - {"cue_img1"}
        )
        if unexpected:
            raise ValueError(
                "Selected first-image rows have unexpected trial types "
                f"{unexpected}: {index_path}"
            )
    selected["condition"] = selected["cue_index"].map(
        PRECONDITIONED_CONDITIONS
    )
    selected["sub"] = sub
    return selected


def _load_probe_patterns(
    *,
    base_dir: Path,
    sub: str,
    runs: tuple[int, ...],
    voxel_indices: np.ndarray,
    mask_shape: tuple[int, int, int],
    mask_affine: np.ndarray,
    sub14_original_events_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, str]:
    rows = []
    sample_tables = []

    for run in runs:
        index_path = _probe_beta_index_path(base_dir, sub, run)
        if not index_path.is_file():
            continue
        selected = _selected_probe_rows(index_path, sub, run)
        subject_dir = index_path.parent

        for _, sample in selected.iterrows():
            beta_path = subject_dir / str(sample["beta_file"])
            rows.append(
                _extract_mask_values(
                    beta_path,
                    voxel_indices,
                    mask_shape,
                    mask_affine,
                )
            )

        selected = selected.copy()
        selected["beta_path"] = selected["beta_file"].map(
            lambda name: str(subject_dir / str(name))
        )
        sample_tables.append(selected)

    if not rows:
        raise RuntimeError(
            f"No preconditioned first-image probe data found for {sub}, runs={runs}"
        )

    samples = pd.concat(sample_tables, ignore_index=True)
    samples.insert(0, "sample_index", np.arange(len(samples)))
    patterns = np.vstack(rows)
    labels, label_source = established_decoder._cue_labels(
        sub,
        samples,
        sub14_original_events_dir,
    )
    run_ids = samples["run"].astype(int).to_numpy()
    if set(labels) != {0, 1}:
        raise ValueError(
            f"Probe data do not contain both odor classes for {sub}: {set(labels)}"
        )
    if len(np.unique(run_ids)) < 2:
        raise ValueError(f"Probe data require at least two runs for {sub}.")
    return patterns, labels, run_ids, samples, label_source


def _validate_passive_space(
    passive_meta: pd.DataFrame,
    passive_affine: np.ndarray,
    mask_shape: tuple[int, int, int],
    mask_affine: np.ndarray,
) -> None:
    if passive_meta.empty:
        raise RuntimeError("No passive beta metadata were returned.")
    first_path = Path(str(passive_meta.iloc[0]["beta_path"]))
    first_image = nib.load(str(first_path))
    _validate_image_space(
        first_image,
        reference_shape=mask_shape,
        reference_affine=mask_affine,
        image_path=first_path,
    )
    if not np.allclose(passive_affine, mask_affine, atol=1e-5):
        raise ValueError(
            "Functional mask and passive odor betas have different affines."
        )


def _analyze_subject(
    *,
    base_dir: Path,
    sub: str,
    roi_name: str,
    mask_path: Path,
    voxel_indices: np.ndarray,
    mask_shape: tuple[int, int, int],
    mask_affine: np.ndarray,
    odor_runs: tuple[int, ...],
    probe_runs: tuple[int, ...],
    svm_c: float,
    sub14_original_events_dir: Path,
    prediction_dir: Path | None,
) -> dict:
    (
        cue_raw,
        cue_y,
        cue_runs,
        cue_samples,
        cue_label_source,
    ) = _load_probe_patterns(
        base_dir=base_dir,
        sub=sub,
        runs=probe_runs,
        voxel_indices=voxel_indices,
        mask_shape=mask_shape,
        mask_affine=mask_affine,
        sub14_original_events_dir=sub14_original_events_dir,
    )
    (
        passive_raw,
        passive_y,
        passive_runs,
        passive_meta,
        passive_affine,
    ) = established_decoder._load_passive_odor_patterns(
        base_dir,
        sub,
        odor_runs,
        voxel_indices,
    )
    _validate_passive_space(
        passive_meta,
        passive_affine,
        mask_shape,
        mask_affine,
    )

    passive_used, cue_used, finite_voxel_mask = (
        established_decoder._drop_nonfinite_voxels(passive_raw, cue_raw)
    )
    passive_patterns = established_decoder._runwise_zscore(
        passive_used,
        passive_runs,
    )
    cue_patterns = established_decoder._runwise_zscore(cue_used, cue_runs)

    passive_metrics, passive_pred, passive_distance = (
        established_decoder._passive_leave_one_run_out(
            passive_patterns,
            passive_y,
            passive_runs,
            svm_c,
        )
    )
    model = SVC(kernel="linear", C=svm_c)
    model.fit(passive_patterns, passive_y)
    cue_pred = model.predict(cue_patterns)
    cue_distance = established_decoder._decision_distance(model, cue_patterns)
    cue_metrics = established_decoder._binary_metrics(
        cue_y,
        cue_pred,
        cue_distance,
    )

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
            subject_dir / f"{CUE_SUBSET}_cue_predictions.csv",
            index=False,
        )

        passive_predictions = passive_meta.copy()
        passive_predictions["cv_predicted_label"] = passive_pred
        passive_predictions["cv_decision_distance_odor2"] = passive_distance
        passive_predictions["cv_signed_distance"] = (
            (2 * passive_y - 1) * passive_distance
        )
        passive_predictions.to_csv(
            subject_dir / f"{CUE_SUBSET}_passive_cv_predictions.csv",
            index=False,
        )

    n_voxels_raw = int(len(voxel_indices))
    n_voxels_used = int(finite_voxel_mask.sum())
    world_x = nib.affines.apply_affine(mask_affine, voxel_indices)[:, 0]
    return {
        "sub": sub,
        "roi": roi_name,
        "hemisphere": "mask_defined",
        "hemisphere_rule": "functional_mask_as_provided",
        "cue_subset": CUE_SUBSET,
        "cue_label_source": cue_label_source,
        "roi_source": "group_functional_mask",
        "roi_mask": str(mask_path),
        "mask_threshold_rule": "finite_and_gt_0",
        "analysis_scope": "exploratory_post_selection",
        "svm_kernel": "linear",
        "svm_c": svm_c,
        "normalization": "within_task_run_voxelwise_zscore",
        "n_voxels_mask": n_voxels_raw,
        "n_voxels_left": int((world_x < 0).sum()),
        "n_voxels_midline": int(np.isclose(world_x, 0.0, atol=1e-5).sum()),
        "n_voxels_right": int((world_x > 0).sum()),
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
        "cue_runs": ",".join(str(run) for run in sorted(np.unique(cue_runs))),
        "passive_cv_accuracy": passive_metrics["accuracy"],
        "passive_cv_balanced_accuracy": passive_metrics["balanced_accuracy"],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary odor decoder on passive odor scans and test "
            "preconditioned first-image premeal cues within one functional mask."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="SPCD data root containing GLM/ and nii/.",
    )
    parser.add_argument(
        "--roi_mask",
        type=Path,
        required=True,
        help="Binary functional mask; finite values greater than zero are used.",
    )
    parser.add_argument(
        "--roi_name",
        default=None,
        help="Output ROI label. Default: mask filename without its extension.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs. Default: discover premeal GLM subjects.",
    )
    parser.add_argument(
        "--odor_runs",
        nargs="+",
        type=int,
        default=list(DEFAULT_ODOR_RUNS),
        help="Passive odor training runs. Default: 1 2.",
    )
    parser.add_argument(
        "--probe_runs",
        nargs="+",
        type=int,
        default=list(DEFAULT_PROBE_RUNS),
        help="Premeal probe test runs. Default: 1 2 3.",
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
            "Subject results path. Default: <base_dir>/roi_decoding/"
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
        help="Stop on the first failed subject instead of recording it.",
    )
    args = parser.parse_args()

    if args.svm_c <= 0:
        raise ValueError("--svm_c must be positive.")
    odor_runs = tuple(dict.fromkeys(args.odor_runs))
    probe_runs = tuple(dict.fromkeys(args.probe_runs))
    if len(odor_runs) < 2:
        raise ValueError("--odor_runs must contain at least two runs.")
    if len(probe_runs) < 2:
        raise ValueError("--probe_runs must contain at least two runs.")

    base_dir = args.base_dir.expanduser().resolve()
    mask_path = args.roi_mask.expanduser().resolve()
    voxel_indices, mask_shape, mask_affine = _load_functional_mask(mask_path)
    roi_name = _sanitize_label(
        args.roi_name if args.roi_name else _neuroimaging_stem(mask_path)
    )
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
        list(dict.fromkeys(args.subjects))
        if args.subjects is not None
        else _discover_subjects(base_dir)
    )

    print("=" * 80)
    print("Functional-mask passive-odor -> preconditioned-cue decoding")
    print("=" * 80)
    print(f"SPCD data root:        {base_dir}")
    print(f"Functional mask:       {mask_path}")
    print(f"ROI:                   {roi_name}")
    print(f"Mask voxels:           {len(voxel_indices)}")
    print(f"Subjects:              {len(subjects)}")
    print(f"Cue subset:            {CUE_SUBSET}")
    print(f"Passive runs:          {', '.join(map(str, odor_runs))}")
    print(f"Probe runs:            {', '.join(map(str, probe_runs))}")
    print(f"Classifier:            linear SVM, C={args.svm_c:g}")
    print("Normalization:         separate within-run voxelwise z-score")
    print("Inference scope:       exploratory/post-selection")
    print(f"Output CSV:            {output_csv}")
    print("=" * 80)

    rows = []
    failures = []
    for sub in subjects:
        try:
            row = _analyze_subject(
                base_dir=base_dir,
                sub=sub,
                roi_name=roi_name,
                mask_path=mask_path,
                voxel_indices=voxel_indices,
                mask_shape=mask_shape,
                mask_affine=mask_affine,
                odor_runs=odor_runs,
                probe_runs=probe_runs,
                svm_c=args.svm_c,
                sub14_original_events_dir=sub14_original_events_dir,
                prediction_dir=prediction_dir,
            )
            rows.append(row)
            print(
                f"[OK] {sub}: cue balanced accuracy="
                f"{row['cue_balanced_accuracy']:.3f}, passive CV="
                f"{row['passive_cv_balanced_accuracy']:.3f}, voxels="
                f"{row['n_voxels_used']}/{row['n_voxels_raw']}"
            )
        except Exception as exc:
            if args.strict:
                raise
            failures.append(
                {
                    "sub": sub,
                    "roi": roi_name,
                    "cue_subset": CUE_SUBSET,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(
                f"[FAIL] {sub}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    if not rows:
        raise RuntimeError("No subject analyses completed successfully.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved subject results: {output_csv}")

    failure_csv = output_csv.with_name(output_csv.stem + "_failures.csv")
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
