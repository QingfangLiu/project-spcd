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
    from rsa_imagewise_odor_identity_first_image.models import (
        create_odor_identity_rdm,
        create_image_identity_rdm,
    )
    from rsa_imagewise_odor_identity_first_image.neural_rdm import compute_correlation_rdm
    from rsa_imagewise_odor_identity_first_image.rsa_stats import (
        rsa_multiple_regression,
        vectorize_lower_triangle,
    )
else:
    from .models import (
        create_odor_identity_rdm,
        create_image_identity_rdm,
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
)

DIAGNOSTIC_MODEL_NAMES = MODEL_NAMES

BETA_INDEX_TEMPLATE = "{sub}_{session_tag}_run-{run}_beta_index.csv"


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


def _load_mask(mask_path: Path, gm_mask_path: Path | None = None, roi_threshold: float = 0.0):
    roi_img = nib.load(str(mask_path))
    roi_data = np.asanyarray(roi_img.dataobj)
    if roi_threshold <= 0:
        roi_mask_raw = np.isfinite(roi_data) & (roi_data > 0)
    else:
        roi_mask_raw = np.isfinite(roi_data) & (roi_data >= roi_threshold)
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
    session_tag: str = "premeal",
) -> list[dict]:
    if cue_subset not in CUE_SUBSETS:
        raise ValueError(f"Unknown cue_subset: {cue_subset}")

    samples = []
    for run in (1, 2, 3):
        try:
            meta = _load_run_beta_meta(base_dir, sub, session_tag, run)
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
            if img_index != 1:
                continue

            trial = int(row["trial"])

            samples.append(
                {
                    "sub": sub,
                    "session": session_tag,
                    "run": run,
                    "trial": trial,
                    "img_index": img_index,
                    "condition": condition,
                    "sample_label": f"{condition}_img1",
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


def _prepare_analysis_patterns(
    raw_patterns: np.ndarray,
    run_ids: np.ndarray,
    center_within_run: bool,
    drop_nonfinite_voxels: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    finite_voxel_mask = np.isfinite(raw_patterns).all(axis=0)
    n_voxels_raw = int(raw_patterns.shape[1])
    n_voxels_nonfinite = int((~finite_voxel_mask).sum())

    if drop_nonfinite_voxels:
        used_voxel_mask = finite_voxel_mask
    else:
        used_voxel_mask = np.ones(n_voxels_raw, dtype=bool)

    n_voxels_used = int(used_voxel_mask.sum())
    if n_voxels_used == 0:
        raise ValueError(
            "No ROI voxels remain after dropping nonfinite voxels "
            f"from {n_voxels_raw} raw ROI voxels."
        )

    raw_used_patterns = raw_patterns[:, used_voxel_mask]
    analysis_patterns = raw_used_patterns
    if center_within_run:
        analysis_patterns = _center_within_run(raw_used_patterns, run_ids)

    voxel_qc = {
        "n_voxels_raw": n_voxels_raw,
        "n_voxels_used": n_voxels_used,
        "n_voxels_nonfinite": n_voxels_nonfinite,
        "n_voxels_dropped_nonfinite": n_voxels_raw - n_voxels_used,
    }
    return raw_used_patterns, analysis_patterns, used_voxel_mask, voxel_qc


def _cross_run_pair_mask(run_ids: np.ndarray) -> np.ndarray:
    n = len(run_ids)
    mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i):
            mat[i, j] = run_ids[i] != run_ids[j]
    return mat[np.tril_indices(n, k=-1)]


def _cross_run_pair_table(samples: list[dict], run_ids: np.ndarray, pair_mask: np.ndarray) -> pd.DataFrame:
    rows = []
    n = len(samples)
    full_pair_idx = 0
    kept_pair_idx = 0
    for i in range(n):
        for j in range(i):
            if pair_mask[full_pair_idx]:
                rows.append(
                    {
                        "pair_index": kept_pair_idx,
                        "sample_i": i,
                        "sample_j": j,
                        "run_i": int(run_ids[i]),
                        "run_j": int(run_ids[j]),
                        "trial_i": samples[i]["trial"],
                        "trial_j": samples[j]["trial"],
                        "condition_i": samples[i]["condition"],
                        "condition_j": samples[j]["condition"],
                        "sample_label_i": samples[i]["sample_label"],
                        "sample_label_j": samples[j]["sample_label"],
                    }
                )
                kept_pair_idx += 1
            full_pair_idx += 1
    return pd.DataFrame(rows)


def _sample_nan_report(
    samples: list[dict],
    raw_patterns: np.ndarray,
    analysis_patterns: np.ndarray,
) -> pd.DataFrame:
    sample_df = pd.DataFrame(samples).copy()
    sample_df.insert(0, "sample_index", np.arange(len(sample_df), dtype=int))

    raw_finite = np.isfinite(raw_patterns)
    analysis_finite = np.isfinite(analysis_patterns)
    report = sample_df[
        [
            "sample_index",
            "sub",
            "session",
            "run",
            "trial",
            "img_index",
            "condition",
            "sample_label",
            "beta_path",
        ]
    ].copy()
    report["n_voxels_raw"] = raw_patterns.shape[1]
    report["n_voxels_analysis"] = analysis_patterns.shape[1]
    report["n_voxels_dropped_nonfinite"] = raw_patterns.shape[1] - analysis_patterns.shape[1]
    report["raw_finite_voxels"] = raw_finite.sum(axis=1).astype(int)
    report["raw_nan_voxels"] = np.isnan(raw_patterns).sum(axis=1).astype(int)
    report["raw_nonfinite_voxels"] = (~raw_finite).sum(axis=1).astype(int)
    report["analysis_finite_voxels"] = analysis_finite.sum(axis=1).astype(int)
    report["analysis_nan_voxels"] = np.isnan(analysis_patterns).sum(axis=1).astype(int)
    report["analysis_nonfinite_voxels"] = (~analysis_finite).sum(axis=1).astype(int)
    return report


def _write_roi_diagnostics(
    diagnostics_root: Path,
    sub: str,
    cue_subset: str,
    samples: list[dict],
    raw_patterns: np.ndarray,
    raw_used_patterns: np.ndarray,
    analysis_patterns: np.ndarray,
    run_ids: np.ndarray,
    roi_mask: np.ndarray,
    used_voxel_mask: np.ndarray,
    voxel_qc: dict,
    neural_rdm: np.ndarray | None = None,
    pair_mask: np.ndarray | None = None,
    model_rdms: dict | None = None,
) -> None:
    sub_dir = diagnostics_root / sub
    sub_dir.mkdir(parents=True, exist_ok=True)

    prefix = _sanitize_label(cue_subset)
    sample_df = pd.DataFrame(samples).copy()
    sample_df.insert(0, "sample_index", np.arange(len(sample_df), dtype=int))
    sample_df.to_csv(sub_dir / f"{prefix}_samples.csv", index=False, na_rep="NaN")

    nan_report = _sample_nan_report(samples, raw_patterns, analysis_patterns)
    nan_report.to_csv(sub_dir / f"{prefix}_nan_report.csv", index=False, na_rep="NaN")

    roi_voxel_indices = np.column_stack(np.where(roi_mask))
    used_roi_voxel_indices = roi_voxel_indices[used_voxel_mask]
    np.savez_compressed(
        sub_dir / f"{prefix}_roi_patterns.npz",
        raw_patterns=raw_patterns,
        raw_used_patterns=raw_used_patterns,
        analysis_patterns=analysis_patterns,
        run_ids=run_ids,
        sample_labels=np.asarray([sample["sample_label"] for sample in samples], dtype=str),
        conditions=np.asarray([sample["condition"] for sample in samples], dtype=str),
        beta_paths=np.asarray([sample["beta_path"] for sample in samples], dtype=str),
        roi_voxel_indices=roi_voxel_indices,
        raw_roi_voxel_indices=roi_voxel_indices,
        analysis_roi_voxel_indices=used_roi_voxel_indices,
        used_voxel_mask=used_voxel_mask,
        n_voxels_raw=voxel_qc["n_voxels_raw"],
        n_voxels_used=voxel_qc["n_voxels_used"],
        n_voxels_nonfinite=voxel_qc["n_voxels_nonfinite"],
        n_voxels_dropped_nonfinite=voxel_qc["n_voxels_dropped_nonfinite"],
        raw_nan_by_voxel=np.isnan(raw_patterns).sum(axis=0).astype(int),
        raw_nonfinite_by_voxel=(~np.isfinite(raw_patterns)).sum(axis=0).astype(int),
        analysis_nan_by_voxel=np.isnan(analysis_patterns).sum(axis=0).astype(int),
        analysis_nonfinite_by_voxel=(~np.isfinite(analysis_patterns)).sum(axis=0).astype(int),
    )

    if neural_rdm is None or pair_mask is None:
        return

    pair_df = _cross_run_pair_table(samples, run_ids, pair_mask)
    pair_df["neural_y"] = vectorize_lower_triangle(neural_rdm, pair_mask=pair_mask)
    if model_rdms is not None:
        for name in DIAGNOSTIC_MODEL_NAMES:
            pair_df[name] = vectorize_lower_triangle(model_rdms[name], pair_mask=pair_mask)
    pair_df.to_csv(sub_dir / f"{prefix}_neural_y.csv", index=False, na_rep="NaN")


def _run_roi_rsa_for_subset(
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
    samples = _build_samples(
        base_dir=base_dir,
        sub=sub,
        cue_subset=cue_subset,
    )
    if not samples:
        raise RuntimeError(f"No samples found for {sub} {cue_subset}")

    run_ids = np.asarray([sample["run"] for sample in samples], dtype=int)
    if np.unique(run_ids).size < 2:
        raise RuntimeError(f"Need at least 2 runs for cross-run RSA: {sub} {cue_subset}")

    raw_patterns = _load_roi_patterns(samples, roi_img, roi_mask)
    raw_used_patterns, patterns, used_voxel_mask, voxel_qc = _prepare_analysis_patterns(
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
                f"using {voxel_qc['n_voxels_used']}/{voxel_qc['n_voxels_raw']} voxels"
            )
        else:
            print(
                f"[QC] {sub} {cue_subset}: found "
                f"{voxel_qc['n_voxels_nonfinite']} nonfinite ROI voxels; "
                "rerun with --drop_nonfinite_voxels to omit them"
            )

    sample_labels = np.asarray([sample["sample_label"] for sample in samples], dtype=object)

    model_rdms = {
        "odor_identity": create_odor_identity_rdm(sample_labels),
        "image_identity": create_image_identity_rdm(sample_labels),
    }

    neural_rdm = compute_correlation_rdm(patterns)
    pair_mask = _cross_run_pair_mask(run_ids)
    if diagnostics_root is not None:
        _write_roi_diagnostics(
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
            model_rdms=model_rdms,
        )

    if not np.isfinite(neural_rdm).all():
        neural_vec = vectorize_lower_triangle(neural_rdm, pair_mask=pair_mask)
        raise ValueError(
            "Neural RDM contains nonfinite values "
            f"({int((~np.isfinite(neural_vec)).sum())}/{len(neural_vec)} cross-run pairs). "
            f"Raw ROI voxels={voxel_qc['n_voxels_raw']}; "
            f"nonfinite ROI voxels={voxel_qc['n_voxels_nonfinite']}. "
            "Use --drop_nonfinite_voxels to drop bad ROI voxels before centering/RDM."
        )

    betas = rsa_multiple_regression(neural_rdm, model_rdms, pair_mask=pair_mask)

    odor_vec = vectorize_lower_triangle(model_rdms["odor_identity"], pair_mask=pair_mask)
    image_vec = vectorize_lower_triangle(model_rdms["image_identity"], pair_mask=pair_mask)

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
        "n_voxels_dropped_nonfinite": voxel_qc["n_voxels_dropped_nonfinite"],
        "n_samples": len(samples),
        "n_runs": int(np.unique(run_ids).size),
        "runs": ",".join(str(run) for run in sorted(np.unique(run_ids))),
        "n_pairs": int(pair_mask.sum()),
        "same_odor_pairs": int((odor_vec == 0).sum()),
        "diff_odor_pairs": int((odor_vec == 1).sum()),
        "same_image_pairs": int((image_vec == 0).sum()),
        "diff_image_pairs": int((image_vec == 1).sum()),
        "center_within_run": bool(center_within_run),
        "drop_nonfinite_voxels": bool(drop_nonfinite_voxels),
        "roi_threshold": roi_threshold,
    }
    for name in MODEL_NAMES:
        row[f"{name}_beta"] = betas[name]
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Run local first-image-only ROI RSA for premeal odor identity in odor-predictive cues."
    )
    parser.add_argument("--base_dir", required=True, help="Base directory containing GLM, nii, and masks folders")
    parser.add_argument("--roi_mask", required=True, help="ROI mask path, absolute or relative to base_dir")
    parser.add_argument("--roi_name", default="lateral_ofc", help="Label used in output CSVs")
    parser.add_argument(
        "--roi_threshold",
        type=float,
        default=0.0,
        help="ROI voxel cutoff. Default 0 preserves the old mask > 0 behavior; positive values use mask >= threshold.",
    )
    parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs; defaults to discovered premeal GLM subjects")
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=sorted(CUE_SUBSETS),
        default=sorted(CUE_SUBSETS),
        help="Cue subsets to run",
    )
    parser.add_argument("--center_within_run", action="store_true")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Default: <base_dir>/rsa_roi/probe_imagewise_odor_identity_first_image/premeal",
    )
    parser.add_argument(
        "--save_diagnostics",
        action="store_true",
        help="Save per-subject ROI patterns, sample metadata, NaN reports, and neural y vectors",
    )
    parser.add_argument(
        "--drop_nonfinite_voxels",
        action="store_true",
        help="Drop ROI voxels with any nonfinite subject/subset sample before centering and RDM computation",
    )
    parser.add_argument("--strict", action="store_true", help="Fail immediately instead of skipping failed subject/subset runs")
    args = parser.parse_args()
    if args.roi_threshold < 0:
        raise ValueError("--roi_threshold must be nonnegative")

    base_dir = Path(args.base_dir).expanduser().resolve()
    roi_mask_path = _resolve_under_base(args.roi_mask, base_dir).resolve()
    roi_name = _sanitize_label(args.roi_name)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else base_dir / "rsa_roi" / "probe_imagewise_odor_identity_first_image" / "premeal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"roi-{roi_name}_subject_betas.csv"
    diagnostics_root = output_dir / "diagnostics" if args.save_diagnostics else None

    gm_mask_path = base_dir / "masks" / "gm_0.1.nii"
    roi_img, roi_mask, n_roi_raw, n_roi_final, n_gm = _load_mask(
        roi_mask_path,
        gm_mask_path=gm_mask_path,
        roi_threshold=args.roi_threshold,
    )

    subjects = sorted(args.subjects) if args.subjects else _discover_subjects(base_dir)

    print("=" * 80)
    print("Premeal odor-identity ROI RSA")
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
    print("Button model:          disabled")
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
                row = _run_roi_rsa_for_subset(
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
                    f"beta={row['odor_identity_beta']:.6f}, "
                    f"samples={row['n_samples']}, pairs={row['n_pairs']}, "
                    f"voxels={row['n_voxels_used']}/{row['n_voxels_raw']}, "
                    f"dropped_nonfinite={row['n_voxels_dropped_nonfinite']}"
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

    dropped_rows = out.loc[out["n_voxels_dropped_nonfinite"] > 0]
    if len(dropped_rows):
        max_dropped = int(dropped_rows["n_voxels_dropped_nonfinite"].max())
        print(
            "\nNonfinite voxel handling:"
            f"\n  Rows with dropped voxels: {len(dropped_rows)} / {len(out)}"
            f"\n  Max dropped in one row:   {max_dropped}"
        )
    else:
        print("\nNonfinite voxel handling: no ROI voxels were dropped in saved rows.")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
