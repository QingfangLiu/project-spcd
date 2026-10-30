#!/usr/bin/env python3
"""Export unprocessed ROI beta patterns for offline RSA and coverage diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsa_imagewise.models import condition_to_group
from rsa_imagewise.ratings import load_subject_pleasantness_ratings
from rsa_imagewise_value_roi.run_roi_rsa import (
    CONDITION_MAP, DEFAULT_SKIP_SUBS, PHASES, _load_run_beta_meta,
    _load_trial_button_map, _same_space, _sha256, prepare_masks,
)


SCHEMA_VERSION = 1
ISSUE_COLUMNS = ["sub", "session", "run_in_phase", "kind", "source", "reason"]
SAMPLE_COLUMNS = [
    "sample_index", "sub", "session", "run", "run_in_phase", "trial", "img_index",
    "condition", "sample_label", "cue_index", "odor_index", "odor_group", "cue_type",
    "cue_set", "trial_button_label", "button_label", "button_metadata_ok", "beta_path",
    "beta_index_csv", "beta_index_row", "beta_index_sha256",
]


def collect_samples(base: Path, sub: str) -> tuple[pd.DataFrame, list, list]:
    """Keep every interpretable beta-index row, including samples with unreadable images."""
    samples, missing_runs, issues = [], [], []
    for phase_idx, phase in enumerate(PHASES):
        for run in (1, 2, 3):
            path = (base / "GLM" / f"{phase}_trialwise_2mm" / sub
                    / f"{sub}_{phase}_run-{run}_beta_index.csv")
            context = dict(sub=sub, session=phase, run_in_phase=run)
            if not path.is_file():
                missing_runs.append(f"{phase}:{run}")
                continue
            try:
                meta = _load_run_beta_meta(base, sub, phase, run)
                source_hash = _sha256(path)
            except Exception as exc:
                issues.append(dict(context, kind="beta_index", source=str(path), reason=str(exc)))
                continue
            buttons_ok = True
            try:
                buttons = _load_trial_button_map(base, sub, phase, run)
            except Exception as exc:
                buttons, buttons_ok = {}, False
                events_path = (base / "nii" / sub / "func_PAID" / phase
                               / f"{sub}_task-{phase}_run-{run}_events.tsv")
                issues.append(dict(context, kind="events", source=str(events_path), reason=str(exc)))
            for row_number, row in meta.iterrows():
                try:
                    cue, odor = int(row["cue_index"]), int(row["odor_index"])
                    condition = CONDITION_MAP[(cue, odor)]
                    position, trial = int(row["img_index"]), int(row["trial"])
                    if position not in (1, 2):
                        raise ValueError(f"Unexpected img_index={position}")
                    trial_button = buttons.get(trial, "none") if buttons_ok else "unknown"
                    samples.append(dict(
                        sample_index=len(samples), sub=sub, session=phase,
                        run=run + 100 * phase_idx, run_in_phase=run, trial=trial,
                        img_index=position, condition=condition,
                        sample_label=f"{condition}_img{position}", cue_index=cue,
                        odor_index=odor, odor_group=condition_to_group(condition),
                        cue_type="preconditioned" if condition.startswith(("A", "C")) else "conditioned",
                        cue_set=int(condition[-1]), trial_button_label=trial_button,
                        button_label="none" if position == 1 else trial_button,
                        button_metadata_ok=buttons_ok, beta_path=str(row["beta_path"]),
                        beta_index_csv=str(path), beta_index_row=int(row_number),
                        beta_index_sha256=source_hash,
                    ))
                except Exception as exc:
                    issues.append(dict(context, kind="sample_metadata", source=str(path),
                                       reason=f"Row {row_number}: {exc}"))
    return pd.DataFrame(samples, columns=SAMPLE_COLUMNS), missing_runs, issues


def extract_patterns(samples: pd.DataFrame, masks: dict) -> tuple[dict, pd.DataFrame, list]:
    """Read each image once, preserving NaN/Inf and every ROI voxel column."""
    matrices = {name: np.full((len(samples), int(mask.sum())), np.nan, dtype=np.float64)
                for name, (_, mask) in masks.items()}
    reference = next(iter(masks.values()))[0]
    samples = samples.copy()
    samples["beta_read_status"] = "ok"
    samples["beta_read_error"] = ""
    issues = []
    for index, sample in samples.iterrows():
        try:
            image = nib.load(sample["beta_path"])
            if len(image.shape) != 3 or not _same_space(image, reference):
                raise ValueError("Beta image must be 3D and match the ROI analysis grid.")
            values = np.asarray(image.dataobj, dtype=np.float64)
            for name, (_, mask) in masks.items():
                matrices[name][index] = values[mask]
        except Exception as exc:
            # NaN placeholders keep row alignment explicit; status distinguishes
            # unreadable images from genuinely nonfinite values in readable images.
            samples.loc[index, "beta_read_status"] = "error"
            samples.loc[index, "beta_read_error"] = str(exc)
            issues.append(dict(sub=sample["sub"], session=sample["session"],
                               run_in_phase=sample["run_in_phase"], kind="beta_image",
                               source=sample["beta_path"], reason=str(exc)))
    return matrices, samples, issues


def coverage_reports(raw: np.ndarray, samples: pd.DataFrame, roi: str) -> tuple[pd.DataFrame, list]:
    finite = np.isfinite(raw)
    per_sample = samples[["sample_index", "sub", "session", "run", "run_in_phase", "trial",
                          "img_index", "condition", "beta_path", "beta_read_status"]].copy()
    per_sample["roi"] = roi
    per_sample["n_voxels"] = raw.shape[1]
    per_sample["n_finite"] = finite.sum(axis=1)
    per_sample["n_nan"] = np.isnan(raw).sum(axis=1)
    per_sample["n_posinf"] = np.isposinf(raw).sum(axis=1)
    per_sample["n_neginf"] = np.isneginf(raw).sum(axis=1)
    per_sample["all_nonfinite"] = ~finite.any(axis=1)

    selections = [("all", "all", "all", "all", np.ones(len(samples), dtype=bool))]
    for phase in PHASES:
        selections.append(("phase", phase, "all", "all", samples.session == phase))
        for run in (1, 2, 3):
            selections.append(("phase_run", phase, run, "all",
                               (samples.session == phase) & (samples.run_in_phase == run)))
        for position in (1, 2):
            selections.append(("phase_image", phase, "all", position,
                               (samples.session == phase) & (samples.img_index == position)))
    for position in (1, 2):
        selections.append(("image", "all", "all", position, samples.img_index == position))
    rows = []
    for scope, phase, run, position, selection in selections:
        chosen = np.asarray(selection, dtype=bool)
        if not chosen.any():
            continue
        valid = finite[chosen]
        rows.append(dict(
            sub=samples.iloc[0]["sub"], roi=roi, scope=scope, session=phase,
            run_in_phase=run, img_index=position, n_samples=int(chosen.sum()),
            n_voxels=raw.shape[1], n_voxels_finite_all_samples=int(valid.all(axis=0).sum()),
            min_finite_voxels_per_sample=int(valid.sum(axis=1).min()),
            n_all_nonfinite_samples=int((~valid.any(axis=1)).sum()),
            n_input_error_samples=int((samples.loc[chosen, "beta_read_status"] != "ok").sum()),
        ))
    return per_sample, rows


def export_ratings(project: Path, sub: str, ratings_file: Path | None) -> tuple[list, dict]:
    source = ratings_file or project / "beh_processed_subs" / sub / "pleasantness_summary.csv"
    values, errors = {}, []
    for phase in PHASES:
        try:
            loaded = (load_subject_pleasantness_ratings(sub, phase, str(source))
                      if source.is_file() else None)
            values[phase] = loaded or {}
        except Exception as exc:
            values[phase] = {}
            errors.append(f"{phase}: {exc}")
    rows = []
    for phase in PHASES:
        for odor_index, group in ((0, "clean"), (1, "odor1"), (2, "odor2")):
            value = values[phase].get(group, np.nan)
            rows.append(dict(sub=sub, session=phase, odor_index=odor_index, odor_group=group,
                             pleasantness=value, rating_available=bool(np.isfinite(value))))
    complete = all(row["rating_available"] for row in rows)
    return rows, dict(ratings_complete=complete, ratings_source=str(source),
                      ratings_sha256=_sha256(source) if source.is_file() else "",
                      ratings_error="; ".join(errors))


def save_roi_archive(path: Path, raw: np.ndarray, samples: pd.DataFrame, image, mask, roi: str):
    ijk = np.column_stack(np.where(mask))
    np.savez_compressed(
        path, schema_version=np.array(SCHEMA_VERSION), raw_patterns=raw,
        sub=np.array(samples.iloc[0]["sub"]), roi=np.array(roi),
        voxel_ijk=ijk, voxel_mni=nib.affines.apply_affine(image.affine, ijk),
        affine=image.affine, sample_index=samples.sample_index.to_numpy(dtype=int),
        run_ids=samples.run.to_numpy(dtype=int), phase_labels=samples.session.to_numpy(dtype=str),
        sample_labels=samples.sample_label.to_numpy(dtype=str),
        img_index=samples.img_index.to_numpy(dtype=int), odor_index=samples.odor_index.to_numpy(dtype=int),
        cue_index=samples.cue_index.to_numpy(dtype=int),
        button_labels=samples.button_label.to_numpy(dtype=str),
        beta_read_status=samples.beta_read_status.to_numpy(dtype=str),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output_dir", type=Path,
                        help="Default: repo/rsa_roi/probe_imagewise_value/raw_patterns.")
    parser.add_argument("--gm_mask", type=Path)
    parser.add_argument("--ratings_file", type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--skip_subs", nargs="*", default=list(DEFAULT_SKIP_SUBS))
    args = parser.parse_args()
    project, base = args.project_root.expanduser().resolve(), args.base_dir.expanduser().resolve()
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              project / "rsa_roi/probe_imagewise_value/raw_patterns")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a fresh --output_dir: {output}")
    subjects = sorted(set(args.subjects)) if args.subjects else sorted(
        p.name for p in (project / "beh_processed_subs").glob("sub-*") if p.is_dir())
    if not subjects or any(not s.startswith("sub-") or Path(s).name != s for s in subjects):
        raise ValueError("Provide valid participant folders such as sub-01.")
    gm = args.gm_mask.expanduser().resolve() if args.gm_mask else base / "masks/gm_0.1.nii"
    ratings_file = args.ratings_file.expanduser().resolve() if args.ratings_file else None
    masks, mask_manifest = prepare_masks(project, gm, output / "masks")
    pd.DataFrame(mask_manifest).to_csv(output / "mask_manifest.csv", index=False)
    config = dict(schema_version=SCHEMA_VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
                  base_dir=str(base), project_root=str(project), output_dir=str(output),
                  subjects=subjects, excluded_subjects=args.skip_subs, roi_names=list(masks),
                  phases=list(PHASES), dtype="float64", centered=False, averaged=False,
                  nonfinite_voxels_removed=False, requires_complete_ratings=False,
                  ratings_file=str(ratings_file) if ratings_file else None,
                  exporter_sha256=_sha256(Path(__file__)), python=sys.version,
                  numpy=np.__version__, pandas=pd.__version__, nibabel=nib.__version__)
    (output / "export_config.json").write_text(json.dumps(config, indent=2) + "\n")
    reports, issues, ratings_rows, coverage = [], [], [], []

    def checkpoint():
        pd.DataFrame(reports).to_csv(output / "subject_export.csv", index=False)
        pd.DataFrame(issues, columns=ISSUE_COLUMNS).to_csv(output / "input_issues.csv", index=False)
        pd.DataFrame(ratings_rows, columns=["sub", "session", "odor_index", "odor_group",
                                          "pleasantness", "rating_available"]).to_csv(
                                              output / "ratings.csv", index=False)
        pd.DataFrame(coverage).to_csv(output / "coverage_summary.csv", index=False)

    for sub in subjects:
        report = dict(sub=sub, status="", reason="", n_samples=0, n_rois_exported=0)
        if sub in args.skip_subs:
            report.update(status="excluded_paper_rule", reason="Current paper participant exclusion")
            reports.append(report)
            checkpoint()
            continue
        roi_files = []
        try:
            subject_ratings, ratings_report = export_ratings(project, sub, ratings_file)
            ratings_rows.extend(subject_ratings)
            report.update(ratings_report)
            samples, missing_runs, sample_issues = collect_samples(base, sub)
            issues.extend(sample_issues)
            report.update(n_samples=len(samples), missing_runs=",".join(missing_runs),
                          n_samples_premeal=int((samples.session == "premeal").sum()),
                          n_samples_postmeal=int((samples.session == "postmeal").sum()))
            if samples.empty:
                raise ValueError("No interpretable beta-index rows found; no pattern matrices exported.")
            matrices, samples, image_issues = extract_patterns(samples, masks)
            issues.extend(image_issues)
            folder = output / sub
            folder.mkdir(parents=True, exist_ok=True)
            samples.to_csv(folder / "samples.csv", index=False)
            for roi, (image, mask) in masks.items():
                raw = matrices[roi]
                archive = folder / f"roi-{roi}_patterns.npz"
                # Raw patterns are saved even when every voxel fails an RSA QC criterion.
                save_roi_archive(archive, raw, samples, image, mask, roi)
                roi_files.append(str(archive.relative_to(output)))
                report["n_rois_exported"] += 1
                per_sample, roi_coverage = coverage_reports(raw, samples, roi)
                per_sample.to_csv(folder / f"roi-{roi}_sample_coverage.csv", index=False)
                coverage.extend(roi_coverage)
                print(f"[EXPORTED] {sub} {roi}: {raw.shape[0]} samples x {raw.shape[1]} voxels; "
                      f"{int(np.isfinite(raw).all(axis=0).sum())} finite across all samples", flush=True)
            report.update(status="exported_with_input_errors" if sample_issues or image_issues else "exported",
                          reason="; ".join(i["reason"] for i in sample_issues + image_issues))
        except Exception as exc:
            report.update(status="failed", reason=str(exc))
            issues.append(dict(sub=sub, session="", run_in_phase="", kind="participant_export",
                               source="", reason=str(exc)))
            print(f"[FAIL] {sub}: {exc}", flush=True)
        report["roi_files"] = ";".join(roi_files)
        reports.append(report)
        checkpoint()
    failed = [r["sub"] for r in reports if r["status"] in ("failed", "exported_with_input_errors")]
    config.update(completed=True, input_error_subjects=failed,
                  exported_subjects=[r["sub"] for r in reports if r["n_rois_exported"] == len(masks)])
    (output / "export_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Export saved to {output}", flush=True)
    if failed:
        raise RuntimeError("Export finished with input errors for " + ", ".join(failed)
                           + "; available patterns were saved. See input_issues.csv.")


if __name__ == "__main__":
    main()
