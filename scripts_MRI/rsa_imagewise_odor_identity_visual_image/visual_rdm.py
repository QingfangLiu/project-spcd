"""Subject-specific visual-image RDMs from independent image-scan betas."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from .models import base_condition
from .neural_rdm import compute_correlation_rdm


DEFAULT_VISUAL_MASK = (
    Path("masks")
    / "independent_image_decoding_masks"
    / "visual_union_p001_visual_only_gm0p1.nii"
)

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


def resolve_under_base(path_value: str | Path, base_dir: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path(base_dir).expanduser().resolve() / path


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


def _same_space(img, ref_img, atol: float = 1e-5) -> bool:
    return img.shape[:3] == ref_img.shape[:3] and np.allclose(
        img.affine, ref_img.affine, atol=atol
    )


def _load_mask(mask_path: Path):
    if not mask_path.exists():
        raise FileNotFoundError(f"Visual mask not found: {mask_path}")
    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    mask = np.isfinite(data) & (data > 0)
    if not np.any(mask):
        raise ValueError(f"Visual mask has no nonzero voxels: {mask_path}")
    return img, mask


def _load_run_patterns(
    base_dir: Path,
    sub: str,
    stim_type: str,
    run: int,
    mask_img,
    mask: np.ndarray,
    img_index_to_condition: dict[int, str],
    zscore_across_trials: bool,
    eps: float,
) -> tuple[list[np.ndarray], list[dict]]:
    events_path = _events_path(base_dir, sub, stim_type, run)
    if not events_path.exists():
        return [], []

    events = pd.read_csv(events_path, sep="\t").reset_index(drop=True)
    if "trial_type" not in events.columns:
        raise ValueError(f"Events TSV missing trial_type column: {events_path}")

    beta_patterns = []
    beta_paths = []
    for row_idx in range(len(events)):
        trial_num = row_idx + 1
        path = _beta_path(base_dir, sub, stim_type, run, trial_num)
        if not path.exists():
            raise FileNotFoundError(f"Localizer beta not found: {path}")

        beta_img = nib.load(str(path))
        if not _same_space(beta_img, mask_img):
            raise ValueError(
                "Localizer beta and visual mask are not in the same space:\n"
                f"Beta: {path}\nMask: {mask_img.get_filename()}"
            )
        beta_patterns.append(np.asanyarray(beta_img.dataobj, dtype=float)[mask])
        beta_paths.append(path)

    run_matrix = np.vstack(beta_patterns)
    if zscore_across_trials:
        run_mean = np.nanmean(run_matrix, axis=0, keepdims=True)
        run_std = np.nanstd(run_matrix, axis=0, keepdims=True)
        run_std[~np.isfinite(run_std) | (run_std < eps)] = 1.0
        run_matrix = (run_matrix - run_mean) / run_std

    selected_patterns: list[np.ndarray] = []
    selected_rows: list[dict] = []
    for row_idx, row in events.iterrows():
        img_index = _parse_img_index(row["trial_type"])
        if img_index not in img_index_to_condition:
            continue

        condition = img_index_to_condition[int(img_index)]
        selected_patterns.append(run_matrix[row_idx, :])
        selected_rows.append(
            {
                "sub": sub,
                "run": run,
                "trial_num": row_idx + 1,
                "trial_type": row["trial_type"],
                "img_index": int(img_index),
                "condition": condition,
                "stim_type": stim_type,
                "beta_path": str(beta_paths[row_idx]),
            }
        )

    return selected_patterns, selected_rows


def _write_visual_diagnostics(
    diagnostics_dir: Path,
    sub: str,
    cue_subset: str,
    sample_labels: Sequence[str],
    sample_rdm: np.ndarray,
    condition_order: Sequence[str],
    base_rdm: np.ndarray,
    localizer_rows: Sequence[dict],
    summary: dict,
) -> None:
    sub_dir = diagnostics_dir / sub
    sub_dir.mkdir(parents=True, exist_ok=True)
    prefix = cue_subset

    pd.DataFrame(localizer_rows).to_csv(
        sub_dir / f"{prefix}_visual_localizer_samples.csv",
        index=False,
        na_rep="NaN",
    )
    pd.DataFrame(base_rdm, index=condition_order, columns=condition_order).to_csv(
        sub_dir / f"{prefix}_visual_base_rdm.csv",
        na_rep="NaN",
    )
    pd.DataFrame(sample_rdm, index=sample_labels, columns=sample_labels).to_csv(
        sub_dir / f"{prefix}_visual_image_rdm.csv",
        na_rep="NaN",
    )
    pd.DataFrame([summary]).to_csv(
        sub_dir / f"{prefix}_visual_rdm_summary.csv",
        index=False,
        na_rep="NaN",
    )


def create_visual_image_rdm(
    sub: str,
    base_dir: str | Path,
    sample_labels: Sequence[str],
    cue_subset: str,
    visual_mask: str | Path = DEFAULT_VISUAL_MASK,
    localizer_runs: Sequence[int] = (1, 2, 3, 4),
    zscore_across_trials: bool = True,
    diagnostics_dir: str | Path | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Build a sample-level visual-image RDM from independent image-scan betas.

    The localizer category is selected from subject_info.csv and the cue subset.
    For the current subset-specific analyses, B cues use the conditioned image
    category and A cues use the preconditioned image category. Localizer betas
    are z-scored voxelwise within run across all localizer trials before the
    relevant cue images are selected and averaged by base cue.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    sample_labels = np.asarray(sample_labels, dtype=object)
    if sample_labels.size < 2:
        raise ValueError("Need at least 2 samples to create a visual-image RDM.")
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

    condition_order = [
        cond
        for cond in CUE_SUBSET_CONDITIONS[cue_subset]
        if cond in set(sample_conditions)
    ]
    if len(condition_order) < 2:
        raise ValueError(
            f"Need at least 2 base conditions in sample labels for {cue_subset}."
        )

    img_index_to_condition = {
        CONDITION_TO_LOCALIZER_IMAGE[cond]: cond
        for cond in condition_order
    }

    visual_mask_path = resolve_under_base(visual_mask, base_dir)
    mask_img, mask = _load_mask(visual_mask_path)

    all_patterns: list[np.ndarray] = []
    localizer_rows: list[dict] = []
    for run in localizer_runs:
        run_patterns, run_rows = _load_run_patterns(
            base_dir=base_dir,
            sub=sub,
            stim_type=stim_type,
            run=int(run),
            mask_img=mask_img,
            mask=mask,
            img_index_to_condition=img_index_to_condition,
            zscore_across_trials=zscore_across_trials,
            eps=eps,
        )
        all_patterns.extend(run_patterns)
        localizer_rows.extend(run_rows)

    if not all_patterns:
        raise RuntimeError(
            f"No independent {stim_type} localizer rows found for {sub} {cue_subset}."
        )

    pattern_matrix = np.vstack(all_patterns)
    finite_voxel_mask = np.isfinite(pattern_matrix).all(axis=0)
    n_nonfinite_voxels = int((~finite_voxel_mask).sum())
    if not np.any(finite_voxel_mask):
        raise ValueError(
            f"No finite visual-mask voxels remain for {sub} {cue_subset}."
        )
    pattern_matrix = pattern_matrix[:, finite_voxel_mask]

    row_conditions = np.asarray(
        [row["condition"] for row in localizer_rows],
        dtype=object,
    )
    missing_conditions = [
        cond for cond in condition_order if not np.any(row_conditions == cond)
    ]
    if missing_conditions:
        raise RuntimeError(
            f"Missing localizer rows for {sub} {cue_subset}: {missing_conditions}"
        )

    base_patterns = np.vstack(
        [
            pattern_matrix[row_conditions == cond].mean(axis=0)
            for cond in condition_order
        ]
    )
    base_rdm = compute_correlation_rdm(base_patterns)

    cond_index = {cond: idx for idx, cond in enumerate(condition_order)}
    sample_rdm = np.zeros((len(sample_conditions), len(sample_conditions)), dtype=float)
    for i, cond_i in enumerate(sample_conditions):
        for j, cond_j in enumerate(sample_conditions):
            sample_rdm[i, j] = base_rdm[cond_index[cond_i], cond_index[cond_j]]
    np.fill_diagonal(sample_rdm, 0.0)

    summary = {
        "sub": sub,
        "cue_subset": cue_subset,
        "pairing": pairing,
        "localizer_category": stim_type,
        "visual_mask": str(visual_mask_path),
        "zscore_across_trials": bool(zscore_across_trials),
        "n_mask_voxels": int(mask.sum()),
        "n_visual_voxels_used": int(finite_voxel_mask.sum()),
        "n_visual_voxels_dropped_nonfinite": n_nonfinite_voxels,
        "n_localizer_rows": int(len(localizer_rows)),
        "localizer_runs": ",".join(
            str(run) for run in sorted({row["run"] for row in localizer_rows})
        ),
        "condition_order": ",".join(condition_order),
    }

    if diagnostics_dir is not None:
        _write_visual_diagnostics(
            diagnostics_dir=Path(diagnostics_dir).expanduser().resolve(),
            sub=sub,
            cue_subset=cue_subset,
            sample_labels=sample_labels,
            sample_rdm=sample_rdm,
            condition_order=condition_order,
            base_rdm=base_rdm,
            localizer_rows=localizer_rows,
            summary=summary,
        )

    return sample_rdm
