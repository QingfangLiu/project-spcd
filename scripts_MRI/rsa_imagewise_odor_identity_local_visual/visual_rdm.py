"""Local visual-image RDM helpers for searchlight-matched control models."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from .models import base_condition
from .neural_rdm import compute_correlation_rdm


CUE_SUBSET_CONDITIONS = {
    "conditioned_odor_only": ("B11", "B21", "B12", "B22"),
    "preconditioned_odor_only": ("A11", "A21", "A12", "A22"),
}

CONDITION_TO_LOCALIZER_IMAGE = {
    "A11": 1,
    "A21": 3,
    "A12": 7,
    "A22": 9,
    "B11": 2,
    "B21": 4,
    "B12": 8,
    "B22": 10,
}


def _subject_info_path(base_dir: Path) -> Path:
    candidates = [
        base_dir / "task_info" / "subject_info.csv",
        base_dir / "subject_info.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "subject_info.csv not found. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def get_pairing_for_subject(base_dir: str | Path, sub: str) -> str:
    path = _subject_info_path(Path(base_dir).expanduser().resolve())
    df = pd.read_csv(path)
    required = {"subject", "pairing"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"subject_info.csv missing columns {sorted(missing)}: {path}")

    row = df.loc[df["subject"] == sub]
    if row.empty:
        raise ValueError(f"Subject {sub} not found in subject_info.csv: {path}")

    pairing = str(row.iloc[0]["pairing"]).strip().lower()
    if pairing not in ("c", "p"):
        raise ValueError(f"Unexpected pairing code for {sub}: {pairing}")
    return pairing


def localizer_category_for_subset(pairing: str, cue_subset: str) -> str:
    if cue_subset not in CUE_SUBSET_CONDITIONS:
        raise ValueError(f"Unknown cue_subset: {cue_subset}")
    if pairing not in ("c", "p"):
        raise ValueError(f"Unexpected pairing code: {pairing}")

    if cue_subset == "conditioned_odor_only":
        return "character" if pairing == "c" else "place"
    if cue_subset == "preconditioned_odor_only":
        return "place" if pairing == "c" else "character"
    raise ValueError(f"Unknown cue_subset: {cue_subset}")


def _events_path(base_dir: Path, sub: str, stim_type: str, run: int) -> Path:
    return (
        base_dir
        / "nii"
        / sub
        / "func_PAID"
        / stim_type
        / f"{sub}_task-{stim_type}_run-{run}_events.tsv"
    )


def _beta_path(base_dir: Path, sub: str, stim_type: str, run: int, trial_num: int) -> Path:
    return (
        base_dir
        / "GLM"
        / f"{stim_type}_trialwise_2mm"
        / sub
        / f"beta_run{run}_trial{trial_num:02d}.nii"
    )


def _parse_img_index(trial_type) -> int | None:
    if not isinstance(trial_type, str):
        return None
    if not trial_type.startswith("img_"):
        return None
    try:
        return int(trial_type.split("_", 1)[1])
    except ValueError:
        return None


def build_localizer_images_for_local_visual_rsa(
    sub: str,
    base_dir: str | Path,
    sample_labels: Sequence[str],
    cue_subset: str,
    localizer_runs: Sequence[int] = (1, 2, 3, 4),
) -> tuple[list, np.ndarray, np.ndarray, tuple[str, ...], list[dict], dict]:
    """
    Load independent localizer beta images for local visual-RDM searchlights.

    The returned image list includes all independent localizer trials in the
    selected character/place category. Rows that match the current subset's A/B
    cue images receive their base-cue condition label; other rows receive an
    empty label and are used only for within-run voxelwise z-scoring.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    if cue_subset not in CUE_SUBSET_CONDITIONS:
        raise ValueError(f"Unknown cue_subset: {cue_subset}")

    pairing = get_pairing_for_subject(base_dir, sub)
    stim_type = localizer_category_for_subset(pairing, cue_subset)

    sample_conditions = [base_condition(label) for label in sample_labels]
    allowed_conditions = set(CUE_SUBSET_CONDITIONS[cue_subset])
    invalid_conditions = sorted(set(sample_conditions) - allowed_conditions)
    if invalid_conditions:
        raise ValueError(
            f"Sample labels include conditions outside {cue_subset}: {invalid_conditions}"
        )

    condition_order = tuple(
        cond
        for cond in CUE_SUBSET_CONDITIONS[cue_subset]
        if cond in set(sample_conditions)
    )
    if len(condition_order) < 2:
        raise ValueError(
            f"Need at least 2 base conditions in sample labels for {cue_subset}."
        )

    img_index_to_condition = {
        CONDITION_TO_LOCALIZER_IMAGE[cond]: cond
        for cond in condition_order
    }

    imgs = []
    run_ids = []
    condition_labels = []
    rows = []

    for run in localizer_runs:
        events_path = _events_path(base_dir, sub, stim_type, int(run))
        if not events_path.exists():
            continue

        events = pd.read_csv(events_path, sep="\t").reset_index(drop=True)
        if "trial_type" not in events.columns:
            raise ValueError(f"Events TSV missing trial_type column: {events_path}")

        for row_idx, row in events.iterrows():
            trial_num = row_idx + 1
            beta_path = _beta_path(base_dir, sub, stim_type, int(run), trial_num)
            if not beta_path.exists():
                raise FileNotFoundError(f"Localizer beta not found: {beta_path}")

            img_index = _parse_img_index(row["trial_type"])
            condition = ""
            if img_index in img_index_to_condition:
                condition = img_index_to_condition[int(img_index)]

            imgs.append(nib.load(str(beta_path)))
            run_ids.append(int(run))
            condition_labels.append(condition)
            rows.append(
                {
                    "sub": sub,
                    "run": int(run),
                    "trial_num": trial_num,
                    "trial_type": row["trial_type"],
                    "img_index": "" if img_index is None else int(img_index),
                    "condition": condition,
                    "stim_type": stim_type,
                    "beta_path": str(beta_path),
                }
            )

    if not imgs:
        raise RuntimeError(
            f"No independent {stim_type} localizer images found for {sub} {cue_subset}."
        )

    run_ids = np.asarray(run_ids, dtype=int)
    condition_labels = np.asarray(condition_labels, dtype=object)
    missing_conditions = [
        cond for cond in condition_order if not np.any(condition_labels == cond)
    ]
    if missing_conditions:
        raise RuntimeError(
            f"Missing localizer rows for {sub} {cue_subset}: {missing_conditions}"
        )

    summary = {
        "sub": sub,
        "cue_subset": cue_subset,
        "pairing": pairing,
        "localizer_category": stim_type,
        "n_localizer_rows_total": int(len(rows)),
        "n_localizer_rows_selected": int(np.sum(condition_labels != "")),
        "localizer_runs": ",".join(str(run) for run in sorted(set(run_ids.tolist()))),
        "condition_order": ",".join(condition_order),
    }
    return imgs, run_ids, condition_labels, condition_order, rows, summary


def _zscore_by_run(
    patterns: np.ndarray,
    run_ids: np.ndarray,
    eps: float,
) -> np.ndarray:
    out = patterns.astype(float, copy=True)
    for run in np.unique(run_ids):
        idx = np.where(run_ids == run)[0]
        if idx.size == 0:
            continue
        run_mean = np.nanmean(out[idx], axis=0, keepdims=True)
        run_std = np.nanstd(out[idx], axis=0, keepdims=True)
        run_std[~np.isfinite(run_std) | (run_std < eps)] = 1.0
        out[idx] = (out[idx] - run_mean) / run_std
    return out


def _row_norms_after_center(patterns: np.ndarray) -> np.ndarray:
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    return np.linalg.norm(centered, axis=1)


def create_local_visual_image_rdm_from_patterns(
    localizer_patterns: np.ndarray,
    localizer_run_ids: Sequence[int],
    localizer_condition_labels: Sequence[str],
    condition_order: Sequence[str],
    sample_labels: Sequence[str],
    zscore_across_trials: bool = True,
    eps: float = 1e-8,
) -> np.ndarray | None:
    """
    Create a sample-level visual RDM from localizer patterns in one voxel set.

    Returns None for degenerate voxel sets where the local visual RDM cannot be
    estimated reliably.
    """
    X = np.asarray(localizer_patterns, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 3:
        return None

    run_ids = np.asarray(localizer_run_ids, dtype=int)
    condition_labels = np.asarray(localizer_condition_labels, dtype=object)
    if X.shape[0] != run_ids.shape[0] or X.shape[0] != condition_labels.shape[0]:
        raise ValueError("Localizer pattern rows must match run and condition labels.")

    if not np.isfinite(X).all():
        finite_voxels = np.isfinite(X).all(axis=0)
        if finite_voxels.sum() < 3:
            return None
        X = X[:, finite_voxels]

    if zscore_across_trials:
        X = _zscore_by_run(X, run_ids=run_ids, eps=eps)

    base_patterns = []
    for condition in condition_order:
        idx = np.where(condition_labels == condition)[0]
        if idx.size == 0:
            return None
        base_patterns.append(X[idx].mean(axis=0))

    base_patterns = np.vstack(base_patterns)
    if np.any(_row_norms_after_center(base_patterns) < eps):
        return None

    base_rdm = compute_correlation_rdm(base_patterns)
    if not np.isfinite(base_rdm).all():
        return None

    sample_conditions = [base_condition(label) for label in sample_labels]
    cond_index = {cond: idx for idx, cond in enumerate(condition_order)}
    if any(cond not in cond_index for cond in sample_conditions):
        raise ValueError("Sample labels do not match local visual condition order.")

    sample_rdm = np.zeros((len(sample_conditions), len(sample_conditions)), dtype=float)
    for i, cond_i in enumerate(sample_conditions):
        for j, cond_j in enumerate(sample_conditions):
            sample_rdm[i, j] = base_rdm[cond_index[cond_i], cond_index[cond_j]]
    np.fill_diagonal(sample_rdm, 0.0)
    return sample_rdm
