#!/usr/bin/env python3
"""
ROI RSA for premeal odor identity in odor-predictive image-wise betas.

This script uses the same sample definitions and model RDMs as the
whole-brain searchlight variant, but extracts one pattern matrix from a
predefined ROI and writes scalar RSA betas to CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rsa_imagewise_odor_identity.models import (
        create_odor_identity_rdm,
        create_image_identity_rdm,
        create_button_presence_rdm,
        create_button_side_rdm,
    )
    from rsa_imagewise_odor_identity.neural_rdm import compute_correlation_rdm
    from rsa_imagewise_odor_identity.rsa_stats import (
        rsa_multiple_regression,
        vectorize_lower_triangle,
    )
else:
    from .models import (
        create_odor_identity_rdm,
        create_image_identity_rdm,
        create_button_presence_rdm,
        create_button_side_rdm,
    )
    from .neural_rdm import compute_correlation_rdm
    from .rsa_stats import rsa_multiple_regression, vectorize_lower_triangle


CONDITION_MAP = {
    (1, 1): "A11",
    (3, 2): "A21",
    (5, 0): "C1",
    (2, 1): "B11",
    (4, 2): "B21",
    (6, 0): "D1",
    (7, 1): "A12",
    (9, 2): "A22",
    (11, 0): "C2",
    (8, 1): "B12",
    (10, 2): "B22",
    (12, 0): "D2",
}

CUE_SUBSETS = {
    "conditioned_odor_only": {"B11", "B21", "B12", "B22"},
    "preconditioned_odor_only": {"A11", "A21", "A12", "A22"},
}

MODEL_NAMES = (
    "odor_identity",
    "image_identity",
    "button_presence",
    "button_side",
)

BETA_INDEX_TEMPLATE = "{sub}_{session_tag}_run-{run}_beta_index.csv"
EVENTS_TEMPLATE = "{sub}_task-{session_tag}_run-{run}_events.tsv"


def _sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def _resolve_under_base(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def _same_space(img, ref_img, atol: float = 1e-5) -> bool:
    return img.shape[:3] == ref_img.shape[:3] and np.allclose(
        img.affine, ref_img.affine, atol=atol
    )


def _load_mask(mask_path: Path, gm_mask_path: Path | None = None):
    roi_img = nib.load(str(mask_path))
    roi_data = np.asanyarray(roi_img.dataobj)
    roi_mask_raw = np.isfinite(roi_data) & (roi_data > 0)
    roi_mask = roi_mask_raw.copy()

    gm_voxels = None
    if gm_mask_path is not None and gm_mask_path.exists():
        gm_img = nib.load(str(gm_mask_path))
        if not _same_space(gm_img, roi_img):
            raise ValueError(
                "ROI mask and gray-matter mask are not in the same space:\n"
                f"ROI: {mask_path}\nGM:  {gm_mask_path}"
            )
        gm_data = np.asanyarray(gm_img.dataobj)
        gm_mask = np.isfinite(gm_data) & (gm_data > 0)
        gm_voxels = int(gm_mask.sum())
        roi_mask &= gm_mask

    n_raw = int(roi_mask_raw.sum())
    n_final = int(roi_mask.sum())
    if n_final == 0:
        raise ValueError(f"ROI mask has no nonzero voxels after GM intersection: {mask_path}")

    return roi_img, roi_mask, n_raw, n_final, gm_voxels


def _discover_subjects(base_dir: Path) -> list[str]:
    glm_dir = base_dir / "GLM" / "premeal_trialwise_2mm"
    if not glm_dir.is_dir():
        raise FileNotFoundError(f"GLM folder not found: {glm_dir}")
    subs = sorted(
        p.name for p in glm_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )
    if not subs:
        raise RuntimeError(f"No subjects found under: {glm_dir}")
    return subs


def _normalize_button_label(label) -> str:
    if pd.isna(label):
        return "none"
    label = str(label).strip().lower()
    if label in ("left", "right"):
        return label
    return "none"


def _load_trial_button_map(base_dir: Path, sub: str, session_tag: str, run: int) -> dict[int, str]:
    events_path = (
        base_dir
        / "nii"
        / sub
        / "func_PAID"
        / session_tag
        / EVENTS_TEMPLATE.format(sub=sub, session_tag=session_tag, run=run)
    )
    if not events_path.exists():
        raise FileNotFoundError(f"Events TSV not found: {events_path}")

    ev = pd.read_csv(events_path, sep="\t").copy()
    required = {"trial", "trial_type", "response_hand"}
    missing = required - set(ev.columns)
    if missing:
        raise ValueError(f"Events TSV missing columns {sorted(missing)}: {events_path}")

    button_rows = ev.loc[ev["trial_type"] == "button", ["trial", "response_hand"]]
    return {
        int(row["trial"]): _normalize_button_label(row["response_hand"])
        for _, row in button_rows.iterrows()
    }


def _load_run_beta_meta(base_dir: Path, sub: str, session_tag: str, run: int) -> pd.DataFrame:
    session_folder = f"{session_tag}_trialwise_2mm"
    sub_dir = base_dir / "GLM" / session_folder / sub
    csv_path = sub_dir / BETA_INDEX_TEMPLATE.format(
        sub=sub,
        session_tag=session_tag,
        run=run,
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    required = {"beta_file", "trial", "img_index", "cue_index", "odor_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Beta index CSV missing columns {sorted(missing)}: {csv_path}")

    df["beta_path"] = df["beta_file"].apply(lambda name: str(sub_dir / str(name)))
    return df


def _build_samples(
    base_dir: Path,
    sub: str,
    cue_subset: str,
    button_scope: str,
    session_tag: str = "premeal",
) -> list[dict]:
    if cue_subset not in CUE_SUBSETS:
        raise ValueError(f"Unknown cue_subset: {cue_subset}")
    if button_scope not in ("both_images", "second_image_only"):
        raise ValueError("button_scope must be 'both_images' or 'second_image_only'")

    samples = []
    for run in (1, 2, 3):
        try:
            meta = _load_run_beta_meta(base_dir, sub, session_tag, run)
            trial_to_button = _load_trial_button_map(base_dir, sub, session_tag, run)
        except FileNotFoundError as exc:
            print(f"[SKIP] {sub} run {run}: {exc}")
            continue

        for _, row in meta.iterrows():
            cue_key = (int(row["cue_index"]), int(row["odor_index"]))
            condition = CONDITION_MAP.get(cue_key)
            if condition not in CUE_SUBSETS[cue_subset]:
                continue

            img_index = int(row["img_index"])
            if img_index not in (1, 2):
                raise ValueError(f"Unexpected img_index={img_index} for {sub} run {run}")

            trial = int(row["trial"])
            trial_button = trial_to_button.get(trial, "none")
            if button_scope == "second_image_only" and img_index == 1:
                button_label = "none"
            else:
                button_label = trial_button

            samples.append(
                {
                    "sub": sub,
                    "session": session_tag,
                    "run": run,
                    "trial": trial,
                    "img_index": img_index,
                    "condition": condition,
                    "sample_label": f"{condition}_img{img_index}",
                    "button_label": button_label,
                    "beta_path": str(row["beta_path"]),
                }
            )

    return samples


def _load_roi_patterns(samples: list[dict], roi_img, roi_mask: np.ndarray) -> np.ndarray:
    rows = []
    for sample in samples:
        beta_path = Path(sample["beta_path"])
        if not beta_path.exists():
            raise FileNotFoundError(f"Beta image not found: {beta_path}")
        beta_img = nib.load(str(beta_path))
        if not _same_space(beta_img, roi_img):
            raise ValueError(
                "Beta image and ROI mask are not in the same space:\n"
                f"Beta: {beta_path}\nROI:  {roi_img.get_filename()}"
            )
        data = np.asanyarray(beta_img.dataobj, dtype=float)
        rows.append(data[roi_mask])
    return np.vstack(rows)


def _center_within_run(patterns: np.ndarray, run_ids: np.ndarray) -> np.ndarray:
    centered = patterns.astype(float, copy=True)
    for run in np.unique(run_ids):
        idx = np.where(run_ids == run)[0]
        centered[idx] -= centered[idx].mean(axis=0, keepdims=True)
    return centered


def _cross_run_pair_mask(run_ids: np.ndarray) -> np.ndarray:
    n = len(run_ids)
    mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i):
            mat[i, j] = run_ids[i] != run_ids[j]
    return mat[np.tril_indices(n, k=-1)]


def _run_roi_rsa_for_subset(
    base_dir: Path,
    sub: str,
    cue_subset: str,
    roi_img,
    roi_mask: np.ndarray,
    roi_name: str,
    roi_mask_path: Path,
    button_scope: str,
    center_within_run: bool,
) -> dict:
    samples = _build_samples(
        base_dir=base_dir,
        sub=sub,
        cue_subset=cue_subset,
        button_scope=button_scope,
    )
    if not samples:
        raise RuntimeError(f"No samples found for {sub} {cue_subset}")

    run_ids = np.asarray([sample["run"] for sample in samples], dtype=int)
    if np.unique(run_ids).size < 2:
        raise RuntimeError(f"Need at least 2 runs for cross-run RSA: {sub} {cue_subset}")

    patterns = _load_roi_patterns(samples, roi_img, roi_mask)
    if center_within_run:
        patterns = _center_within_run(patterns, run_ids)

    sample_labels = np.asarray([sample["sample_label"] for sample in samples], dtype=object)
    button_labels = np.asarray([sample["button_label"] for sample in samples], dtype=object)

    model_rdms = {
        "odor_identity": create_odor_identity_rdm(sample_labels),
        "image_identity": create_image_identity_rdm(sample_labels),
        "button_presence": create_button_presence_rdm(button_labels),
        "button_side": create_button_side_rdm(button_labels),
    }

    neural_rdm = compute_correlation_rdm(patterns)
    pair_mask = _cross_run_pair_mask(run_ids)
    betas = rsa_multiple_regression(neural_rdm, model_rdms, pair_mask=pair_mask)

    odor_vec = vectorize_lower_triangle(model_rdms["odor_identity"], pair_mask=pair_mask)
    image_vec = vectorize_lower_triangle(model_rdms["image_identity"], pair_mask=pair_mask)

    row = {
        "sub": sub,
        "session": "premeal",
        "cue_subset": cue_subset,
        "roi": roi_name,
        "roi_mask": str(roi_mask_path),
        "n_voxels": int(roi_mask.sum()),
        "n_samples": len(samples),
        "n_runs": int(np.unique(run_ids).size),
        "runs": ",".join(str(run) for run in sorted(np.unique(run_ids))),
        "n_pairs": int(pair_mask.sum()),
        "same_odor_pairs": int((odor_vec == 0).sum()),
        "diff_odor_pairs": int((odor_vec == 1).sum()),
        "same_image_pairs": int((image_vec == 0).sum()),
        "diff_image_pairs": int((image_vec == 1).sum()),
        "button_scope": button_scope,
        "center_within_run": bool(center_within_run),
    }
    for name in MODEL_NAMES:
        row[f"{name}_beta"] = betas[name]
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run local ROI RSA for premeal odor identity in odor-predictive cues."
    )
    parser.add_argument("--base_dir", required=True, help="Base directory containing GLM, nii, and masks folders")
    parser.add_argument("--roi_mask", required=True, help="ROI mask path, absolute or relative to base_dir")
    parser.add_argument("--roi_name", default="lateral_ofc", help="Label used in output CSVs")
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs; defaults to discovered premeal GLM subjects")
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=sorted(CUE_SUBSETS),
        default=sorted(CUE_SUBSETS),
        help="Cue subsets to run",
    )
    parser.add_argument(
        "--button_scope",
        default="second_image_only",
        choices=["both_images", "second_image_only"],
    )
    parser.add_argument("--center_within_run", action="store_true")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <base_dir>/rsa_roi/probe_imagewise_odor_identity/premeal",
    )
    parser.add_argument("--strict", action="store_true", help="Fail immediately instead of skipping failed subject/subset runs")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    roi_mask_path = _resolve_under_base(args.roi_mask, base_dir).resolve()
    roi_name = _sanitize_label(args.roi_name)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else base_dir / "rsa_roi" / "probe_imagewise_odor_identity" / "premeal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"roi-{roi_name}_subject_betas.csv"

    gm_mask_path = base_dir / "masks" / "gm_0.1.nii"
    roi_img, roi_mask, n_roi_raw, n_roi_final, n_gm = _load_mask(
        roi_mask_path,
        gm_mask_path=gm_mask_path,
    )

    subjects = sorted(args.subjects) if args.subjects else _discover_subjects(base_dir)

    print("=" * 80)
    print("Premeal odor-identity ROI RSA")
    print("=" * 80)
    print(f"Base dir:              {base_dir}")
    print(f"ROI mask:              {roi_mask_path}")
    print(f"ROI label:             {roi_name}")
    print(f"ROI voxels raw/final:  {n_roi_raw} / {n_roi_final}")
    if n_gm is not None:
        print(f"GM mask voxels:        {n_gm}")
    print(f"Subjects:              {len(subjects)}")
    print(f"Cue subsets:           {', '.join(args.cue_subsets)}")
    print(f"Button scope:          {args.button_scope}")
    print(f"Center within run:     {args.center_within_run}")
    print(f"Output CSV:            {output_csv}")
    print("=" * 80)

    rows = []
    failures = []
    for sub in subjects:
        for cue_subset in args.cue_subsets:
            try:
                row = _run_roi_rsa_for_subset(
                    base_dir=base_dir,
                    sub=sub,
                    cue_subset=cue_subset,
                    roi_img=roi_img,
                    roi_mask=roi_mask,
                    roi_name=roi_name,
                    roi_mask_path=roi_mask_path,
                    button_scope=args.button_scope,
                    center_within_run=args.center_within_run,
                )
                rows.append(row)
                print(
                    f"[OK] {sub} {cue_subset}: "
                    f"beta={row['odor_identity_beta']:.6f}, "
                    f"samples={row['n_samples']}, pairs={row['n_pairs']}"
                )
            except Exception as exc:
                message = f"{sub} {cue_subset}: {exc}"
                failures.append(message)
                print(f"[FAIL] {message}")
                if args.strict:
                    raise

    if not rows:
        raise RuntimeError("No ROI RSA rows were produced.")

    out = pd.DataFrame(rows)
    out = out.sort_values(["sub", "cue_subset"]).reset_index(drop=True)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved subject ROI RSA betas: {output_csv}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
