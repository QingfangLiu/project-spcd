#!/usr/bin/env python3
"""Run combined pre/post value RSA in independent vmPFC and two bilateral OFC ROIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from scipy import ndimage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse the established ROI I/O/QC and the current searchlight's model/statistics.
from rsa_imagewise_odor_identity.run_roi_rsa import (
    CONDITION_MAP, _cross_run_pair_mask, _load_roi_patterns, _load_run_beta_meta,
    _load_trial_button_map, _prepare_analysis_patterns, _same_space,
)
from rsa_imagewise.models import (
    create_odor_value_rdm_by_phase, create_cue_type_rdm, create_image_identity_rdm,
    create_button_presence_rdm, create_button_side_rdm,
)
from rsa_imagewise.neural_rdm import compute_correlation_rdm
from rsa_imagewise.ratings import load_subject_pleasantness_ratings
from rsa_imagewise.rsa_stats import rsa_multiple_regression, vectorize_lower_triangle
from rsa_imagewise_value_roi.aggregate_roi_rsa import (
    DEFAULT_SKIP_SUBS, ROI_NAMES, summarize_subject_betas,
)


PHASES = ("premeal", "postmeal")
MODELS = ("odor_value", "cue_type", "image_identity", "button_presence", "button_side")
BARTRA_SOURCE = "masks/nifti_roi-masks_fig03/fig03D/C_POSvsNEG_kda10_clust_tstat1_bin.nii.gz"
OFC_MASKS = (
    "mask-OFC_K2_cluster1_medial_posterior_lateral_gm0p1.nii.gz",
    "mask-OFC_K2_cluster2_central_anterior_lateral_gm0p1.nii.gz",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_masks(project_root: Path, gm_path: Path, destination: Path) -> tuple[dict, list]:
    """Select Bartra's independent vmPFC component, then align all masks to GM."""
    gm_img = nib.load(gm_path)
    gm_values = np.asarray(gm_img.dataobj)
    gm = np.isfinite(gm_values) & (gm_values > 0)
    if gm.ndim != 3 or not gm.any():
        raise ValueError("The gray-matter mask must be a nonempty 3D image.")
    bartra_path = project_root / BARTRA_SOURCE
    bartra = nib.load(bartra_path)
    labels, _ = ndimage.label(np.asarray(bartra.dataobj) > 0,
                             structure=np.ones((3, 3, 3)))
    seed = np.rint(nib.affines.apply_affine(np.linalg.inv(bartra.affine),
                                          [0, 44, -6])).astype(int)
    if np.any(seed < 0) or np.any(seed >= labels.shape):
        raise ValueError("Bartra vmPFC seed is outside the supplied image.")
    component = labels[tuple(seed)]
    if component == 0 or int((labels == component).sum()) != 1245:
        raise ValueError("Expected the independent Bartra vmPFC component (1,245 native voxels).")
    sources = [(bartra_path, nib.Nifti1Image((labels == component).astype(np.uint8),
                                           bartra.affine))]
    for filename in OFC_MASKS:
        path = project_root / "masks/OFC_clusterMap_K2_gm0p1" / filename
        sources.append((path, nib.load(path)))
    masks, manifest = {}, []
    destination.mkdir(parents=True, exist_ok=True)
    for name, (source, image) in zip(ROI_NAMES, sources):
        aligned = image if _same_space(image, gm_img) else resample_from_to(image, gm_img, order=0)
        values = np.asarray(aligned.dataobj)
        raw = np.isfinite(values) & (values > 0)
        mask = raw & gm
        if mask.sum() < 3:
            raise ValueError(f"ROI has fewer than three voxels after GM intersection: {name}")
        output = destination / f"roi-{name}_gm0p1.nii.gz"
        roi_img = nib.Nifti1Image(mask.astype(np.uint8), gm_img.affine)
        nib.save(roi_img, output)
        masks[name] = (roi_img, mask)
        manifest.append(dict(roi=name, source=str(source), source_sha256=_sha256(source),
                             mask=str(output), mask_sha256=_sha256(output),
                             n_voxels_before_gm=int(raw.sum()), n_voxels=int(mask.sum()),
                             gm_mask=str(gm_path), gm_sha256=_sha256(gm_path)))
    return masks, manifest


def build_samples(base_dir: Path, sub: str) -> tuple[list, list]:
    """Adapt existing ROI sample loading to all cues and both phases.

    Only absent beta-index CSVs count as unavailable runs; broken present
    runs fail explicitly rather than silently changing the selected samples.
    """
    samples, missing_runs = [], []
    for phase_idx, phase in enumerate(PHASES):
        phase_start = len(samples)
        for run in (1, 2, 3):
            csv_path = (base_dir / "GLM" / f"{phase}_trialwise_2mm" / sub
                        / f"{sub}_{phase}_run-{run}_beta_index.csv")
            if not csv_path.is_file():
                missing_runs.append(f"{phase}:{run}")
                continue
            meta = _load_run_beta_meta(base_dir, sub, phase, run)
            buttons = _load_trial_button_map(base_dir, sub, phase, run)
            run_start = len(samples)
            for _, row in meta.iterrows():
                key = (int(row["cue_index"]), int(row["odor_index"]))
                if key not in CONDITION_MAP:
                    continue
                img_index = int(row["img_index"])
                if img_index not in (1, 2):
                    raise ValueError(f"Unexpected img_index={img_index}: {csv_path}")
                condition, trial = CONDITION_MAP[key], int(row["trial"])
                samples.append(dict(
                    sub=sub, session=phase, run=run + phase_idx * 100, run_in_phase=run,
                    trial=trial, img_index=img_index, condition=condition,
                    sample_label=f"{condition}_img{img_index}",
                    button_label="none" if img_index == 1 else buttons.get(trial, "none"),
                    beta_path=str(row["beta_path"]),
                ))
            if len(samples) == run_start:
                raise ValueError(f"No valid cue presentations in {csv_path}")
        if len(samples) == phase_start:
            raise ValueError(f"No usable {phase} samples for {sub}; both phases are required.")
    return samples, missing_runs


def build_models(samples: list, ratings: dict) -> tuple[dict, np.ndarray, np.ndarray, int]:
    labels = [s["sample_label"] for s in samples]
    phases = [s["session"] for s in samples]
    buttons = [s["button_label"] for s in samples]
    models = dict(
        odor_value=create_odor_value_rdm_by_phase(labels, phases, ratings),
        cue_type=create_cue_type_rdm(labels), image_identity=create_image_identity_rdm(labels),
        button_presence=create_button_presence_rdm(buttons),
        button_side=create_button_side_rdm(buttons),
    )
    run_ids = np.asarray([s["run"] for s in samples])
    pairs = _cross_run_pair_mask(run_ids)
    design = np.column_stack([vectorize_lower_triangle(models[m], pairs) for m in MODELS])
    design = np.column_stack([np.ones(len(design)), design])
    if not np.isfinite(design).all():
        raise ValueError("Model RDMs contain nonfinite values.")
    rank = int(np.linalg.matrix_rank(design))
    nuisance_rank = np.linalg.matrix_rank(design[:, [0, 2, 3, 4, 5]])
    if rank != nuisance_rank + 1:
        raise ValueError("Value effect is not identifiable given the nuisance RDMs.")
    return models, run_ids, pairs, rank


def analyze_roi(samples, models, run_ids, pairs, roi_img, mask, center_within_run):
    raw = _load_roi_patterns(samples, roi_img, mask)
    _, patterns, used, qc = _prepare_analysis_patterns(
        raw, run_ids, center_within_run=center_within_run, drop_nonfinite_voxels=True)
    if patterns.shape[1] < 3:
        raise ValueError("Fewer than three finite ROI voxels remain.")
    if np.any(np.linalg.norm(patterns - patterns.mean(axis=1, keepdims=True), axis=1) == 0):
        raise ValueError("A sample has a constant spatial pattern; correlation is undefined.")
    neural = compute_correlation_rdm(patterns)
    if not np.isfinite(neural).all():
        raise ValueError("Nonfinite neural RDM.")
    betas = rsa_multiple_regression(neural, models, pair_mask=pairs)
    if not all(np.isfinite(list(betas.values()))):
        raise ValueError("Nonfinite RSA coefficients.")
    qc["voxel_loss_fraction"] = qc["n_voxels_dropped_nonfinite"] / qc["n_voxels_raw"]
    qc["qc_high_voxel_loss"] = qc["voxel_loss_fraction"] >= 0.20
    diagnostics = dict(raw_patterns=raw, analysis_patterns=patterns, used_voxel_mask=used,
                       roi_voxel_indices=np.column_stack(np.where(mask)), neural_rdm=neural)
    return betas, qc, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, required=True, help="External data root containing GLM/ and nii/.")
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output_dir", type=Path, help="Default: repo/rsa_roi/probe_imagewise_value/both.")
    parser.add_argument("--gm_mask", type=Path, help="Default: base_dir/masks/gm_0.1.nii.")
    parser.add_argument("--ratings_file", type=Path, help="Optional compiled wide ratings CSV.")
    parser.add_argument("--subjects", nargs="+", help="Default: participant folders in repo/beh_processed_subs.")
    parser.add_argument("--skip_subs", nargs="*", default=list(DEFAULT_SKIP_SUBS))
    parser.add_argument("--center_within_run", action="store_true",
                        help="Sensitivity option; default off, matching the current value searchlight.")
    parser.add_argument("--save_diagnostics", action="store_true")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Summarize successful subjects even when other eligible subjects fail.")
    args = parser.parse_args()
    project, base = args.project_root.expanduser().resolve(), args.base_dir.expanduser().resolve()
    if args.center_within_run and args.output_dir is None:
        output_tag = "both_centered"
    else:
        output_tag = "both"
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              project / "rsa_roi/probe_imagewise_value" / output_tag)
    # A fresh destination keeps old summaries/diagnostics from masquerading as a new run.
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh --output_dir: {output}")
    gm_path = (args.gm_mask.expanduser().resolve() if args.gm_mask else base / "masks/gm_0.1.nii")
    ratings_file = str(args.ratings_file.expanduser().resolve()) if args.ratings_file else None
    subject_root = project / "beh_processed_subs"
    subjects = sorted(set(args.subjects)) if args.subjects else sorted(
        p.name for p in subject_root.glob("sub-*") if p.is_dir())
    if not subjects:
        raise ValueError("No participant folders found; provide --subjects or a complete project clone.")
    if any(not sub.startswith("sub-") or Path(sub).name != sub for sub in subjects):
        raise ValueError("Subject IDs must be folder names such as sub-01.")
    masks, mask_manifest = prepare_masks(project, gm_path, output / "masks")
    pd.DataFrame(mask_manifest).to_csv(output / "mask_manifest.csv", index=False)
    config = dict(base_dir=str(base), project_root=str(project), output_dir=str(output),
                  subjects=subjects, excluded_subjects=args.skip_subs, session="both",
                  cue_subset="all", image_positions=[1, 2], button_scope="second_image_only",
                  center_within_run=args.center_within_run, drop_nonfinite_voxels=True,
                  save_diagnostics=args.save_diagnostics, allow_partial=args.allow_partial,
                  models=list(MODELS), phase_regressor=False, odor_identity_regressor=False,
                  roi_names=list(ROI_NAMES), masks=mask_manifest, ratings_file=ratings_file,
                  python=sys.version)
    rows, inclusion, failures = [], [], []
    for sub in subjects:
        report = dict(sub=sub, status="", reason="")
        if sub in args.skip_subs:
            report.update(status="excluded_paper_rule", reason="Current paper participant exclusion")
            inclusion.append(report)
            continue
        try:
            # Specify the repo explicitly rather than relying on the imported module's location.
            subject_ratings = ratings_file or str(subject_root / sub / "pleasantness_summary.csv")
            ratings = {phase: load_subject_pleasantness_ratings(sub, phase, subject_ratings)
                       for phase in PHASES}
            if any(r is None for r in ratings.values()):
                report.update(status="excluded_missing_ratings", reason="Complete pre/post ratings required")
                inclusion.append(report)
                continue
            if not np.isfinite([v for phase in PHASES for v in ratings[phase].values()]).all():
                raise ValueError("Nonfinite pleasantness ratings; provide complete ratings.")
            report["ratings_source"] = subject_ratings
            report["ratings_sha256"] = _sha256(Path(subject_ratings))
            samples, missing = build_samples(base, sub)
            models, run_ids, pairs, rank = build_models(samples, ratings)
            report.update(n_samples=len(samples), n_pairs=int(pairs.sum()),
                          missing_runs=",".join(missing), qc_incomplete_runs=bool(missing))
            pending = []
            for name, (roi_img, mask) in masks.items():
                betas, qc, diagnostics = analyze_roi(samples, models, run_ids, pairs,
                                                     roi_img, mask, args.center_within_run)
                row = dict(sub=sub, roi=name, session="both", cue_subset="all",
                           n_samples=len(samples), n_pairs=int(pairs.sum()),
                           n_runs=int(len(np.unique(run_ids))), missing_runs=",".join(missing),
                           qc_incomplete_runs=bool(missing), design_rank=rank,
                           center_within_run=args.center_within_run, **qc)
                row.update({f"{m}_beta": betas[m] for m in MODELS})
                pending.append(row)
                if args.save_diagnostics:
                    folder = output / "diagnostics" / sub / name
                    folder.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(samples).to_csv(folder / "samples.csv", index=False)
                    np.savez_compressed(folder / "rsa.npz", **diagnostics, run_ids=run_ids,
                                        pair_mask=pairs, **{f"model_{m}": models[m] for m in MODELS})
            rows.extend(pending)
            report.update(status="included", reason="Successful in all three ROIs")
            print(f"[OK] {sub}: " + "; ".join(f"{r['roi']}={r['odor_value_beta']:.6g}" for r in pending))
        except Exception as exc:
            report.update(status="failed", reason=str(exc))
            failures.append(f"{sub}: {exc}")
            print(f"[FAIL] {sub}: {exc}")
        inclusion.append(report)
        # Preserve completed subjects and inclusion decisions even if interrupted later.
        pd.DataFrame(inclusion).to_csv(output / "subject_inclusion.csv", index=False)
        pd.DataFrame(rows).to_csv(output / "subject_betas.csv", index=False)

    pd.DataFrame(inclusion).to_csv(output / "subject_inclusion.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "subject_betas.csv", index=False)
    config.update(included_subjects=[r["sub"] for r in inclusion if r["status"] == "included"],
                  failures=failures)
    (output / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    if failures and not args.allow_partial:
        raise RuntimeError("Eligible participants failed. See subject_inclusion.csv; no group summary written. "
                           "Fix the inputs and rerun in a fresh folder, or explicitly use --allow_partial.")
    summary = summarize_subject_betas(pd.DataFrame(rows))
    summary["excluded_subjects"] = ",".join(args.skip_subs)
    summary["failed_subjects"] = ",".join(r["sub"] for r in inclusion if r["status"] == "failed")
    summary["missing_ratings_subjects"] = ",".join(
        r["sub"] for r in inclusion if r["status"] == "excluded_missing_ratings")
    summary["allow_partial"] = args.allow_partial
    summary.to_csv(output / "group_summary.csv", index=False)
    print(f"Saved results for {len(config['included_subjects'])} common participants: {output}")


if __name__ == "__main__":
    main()
