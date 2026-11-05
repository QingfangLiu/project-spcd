#!/usr/bin/env python3
"""Prepare, run, and validate the updated combined-phase value searchlights.

The estimator, models, sample loader, and map writer are the existing
rsa_imagewise implementation. This file supplies cohort and execution control.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shlex
import sys
import traceback

import nibabel as nib
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from rsa_imagewise.ratings import load_subject_pleasantness_ratings

SUBSETS = ("all", "conditioned_only", "preconditioned_only")
PHASES = ("premeal", "postmeal")
EXCLUDED = ("sub-19", "sub-31", "sub-33")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_plan(folder):
    folder = Path(folder).resolve()
    plan = json.loads((folder / "analysis_config.json").read_text())
    for relative, expected in plan["source_sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError(f"Source changed since preparation: {relative}. Prepare a fresh run.")
    for name, expected in plan["input_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Prepared input changed: {name}. Prepare a fresh run.")
    return folder, plan


def prepare(args):
    folder = args.output_dir.resolve()
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {folder}")
    base = args.base_dir.resolve()
    gm_path = base / "masks/gm_0.1.nii"
    gm_image = nib.load(gm_path)
    gm = np.asarray(gm_image.dataobj)
    if gm.ndim != 3 or not np.isfinite(gm).all() or not (gm > 0).any():
        raise ValueError("Expected a finite, nonempty 3D gray-matter mask.")
    candidates = sorted(set(args.subjects or [p.name for p in
                        (PROJECT / "beh_processed_subs").glob("sub-*") if p.is_dir()]))
    if not candidates or any(not re.fullmatch(r"sub-\d{2}", s) for s in candidates):
        raise ValueError("Provide participant IDs such as sub-01, or a complete repository clone.")
    reports, ratings, subjects = [], [], []
    for sub in candidates:
        row = dict(sub=sub, status="", reason="")
        if sub in EXCLUDED:
            row.update(status="excluded_paper_rule", reason="Current paper exclusion")
        else:
            source = (args.ratings_file.resolve() if args.ratings_file else
                      PROJECT / "beh_processed_subs" / sub / "pleasantness_summary.csv")
            row.update(ratings_source=str(source),
                       ratings_sha256=sha256(source) if source.is_file() else "")
            # Malformed existing ratings fail explicitly; absence is an eligibility rule.
            by_phase = {p: load_subject_pleasantness_ratings(sub, p, str(source)) for p in PHASES}
            if any(r is None for r in by_phase.values()):
                row.update(status="excluded_missing_ratings", reason="Complete pre/post ratings required")
            else:
                values = np.array([by_phase[p][g] for p in PHASES
                                   for g in ("odor1", "odor2", "clean")])
                if not np.isfinite(values).all():
                    row.update(status="excluded_missing_ratings", reason="Nonfinite ratings")
                elif values.std() == 0:
                    row.update(status="failed_ratings", reason="All six ratings are identical")
                else:
                    subjects.append(sub)
                    ratings.append(dict(sub=sub, **{f"{prefix}_{g}": by_phase[p][g]
                        for p, prefix in zip(PHASES, ("pre", "post"))
                        for g in ("odor1", "odor2", "clean")}))
                    row.update(status="planned", reason="Data validation occurs on execution")
        reports.append(row)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reports).to_csv(folder / "subject_inclusion.csv", index=False)
    if any(r["status"] == "failed_ratings" for r in reports) or not subjects:
        raise ValueError("Unidentifiable ratings or no eligible participants; see subject_inclusion.csv.")
    pd.DataFrame(ratings).to_csv(folder / "ratings.csv", index=False)
    nib.save(gm_image, folder / "gm_mask.nii")
    sources = sorted((SCRIPTS / "rsa_imagewise").glob("*.py")) + [
        Path(__file__).resolve(), Path(__file__).with_name("aggregate_group.m")]
    plan = dict(schema_version=1, base_dir=str(base), subjects=subjects,
                excluded_paper_subjects=list(EXCLUDED), subsets=list(SUBSETS), session="both",
                radius_mm=args.radius, n_jobs=args.n_jobs, button_scope="second_image_only",
                center_within_run=False, target="odor_value", group_smoothing_fwhm_mm=6,
                model_note="Original five regressors; cue_type dropped within single cue classes; intercept; no phase or odor-identity regressor",
                ratings_note="Six phase-by-outcome values jointly z-scored within participant",
                source_sha256={str(p.relative_to(PROJECT)): sha256(p) for p in sources},
                input_sha256={p: sha256(folder / p) for p in
                              ("ratings.csv", "gm_mask.nii", "subject_inclusion.csv")},
                gm_source=str(gm_path), gm_source_sha256=sha256(gm_path))
    write_json(folder / "analysis_config.json", plan)
    runner = Path(__file__).resolve()
    commands = [shlex.join([sys.executable, "-B", str(runner), "run", "--run_dir", str(folder),
                           "--subjects", sub]) for sub in subjects]
    (folder / "jobs.swarm").write_text("\n".join(commands) + "\n")
    print(f"Prepared {len(subjects)} participants × 3 analyses: {folder}")
    print("sub-10 included:", "sub-10" in subjects)


def same_grid(image, reference):
    return image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-5)


def validate_inputs(base, sub, mask_image, report_dir):
    """Check complete runs before the legacy loader can silently skip errors.

    Nonfinite beta values inside the analysis mask are reported and block this
    participant. They are never filled, resampled, or turned into an exclusion.
    """
    from rsa_imagewise.io_imagewise import condition_map, _load_trial_button_map
    mask = np.asarray(mask_image.dataobj) > 0
    rows, problems, samples = [], [], []
    for phase_idx, phase in enumerate(PHASES):
        for run in (1, 2, 3):
            context = dict(sub=sub, phase=phase, run=run)
            path = base / "GLM" / f"{phase}_trialwise_2mm" / sub / f"{sub}_{phase}_run-{run}_beta_index.csv"
            try:
                meta = pd.read_csv(path)
                required = {"trial", "img_index", "cue_index", "odor_index", "beta_file"}
                if not required.issubset(meta.columns):
                    raise ValueError(f"Missing index columns: {sorted(required - set(meta.columns))}")
                if len(meta) != 48:
                    raise ValueError(f"Expected 48 presentations, found {len(meta)}")
                for key in ("trial", "img_index", "cue_index", "odor_index"):
                    vals = pd.to_numeric(meta[key], errors="raise")
                    if not np.isfinite(vals).all() or not (vals == np.floor(vals)).all():
                        raise ValueError(f"Invalid integer labels in {key}")
                if meta.duplicated(["trial", "img_index"]).any() or meta.trial.nunique() != 24:
                    raise ValueError("Expected 24 unique trials, each with two presentations")
                if not meta.groupby("trial").img_index.apply(lambda x: set(x) == {1, 2}).all():
                    raise ValueError("Every trial must contain img_index 1 and 2")
                # Use the same events validation and response mapping as the original loader.
                _load_trial_button_map(str(base), sub, phase, run)
                for i, record in meta.iterrows():
                    condition = condition_map[(int(record.cue_index), int(record.odor_index))]
                    beta = path.parent / str(record.beta_file)
                    qc = dict(context, index_row=int(i), trial=int(record.trial),
                              img_index=int(record.img_index), condition=condition, beta_path=str(beta))
                    try:
                        image = nib.load(beta)
                        if not same_grid(image, mask_image):
                            raise ValueError("Beta and gray-matter mask grids differ")
                        values = np.asarray(image.dataobj)[mask]
                        qc.update(n_mask_voxels=int(mask.sum()),
                                  n_nonfinite_voxels=int((~np.isfinite(values)).sum()))
                        if qc["n_nonfinite_voxels"]:
                            raise ValueError("Nonfinite values inside gray-matter mask; inspect before rerunning")
                        if np.ptp(values) == 0:
                            raise ValueError("Constant spatial pattern inside gray-matter mask")
                        qc["status"] = "ok"
                    except Exception as exc:
                        qc.update(status="failed", reason=str(exc))
                        problems.append(dict(qc))
                    rows.append(qc)
                    samples.append(dict(phase=phase, run=run + phase_idx * 100,
                                        label=f"{condition}_img{int(record.img_index)}"))
            except Exception as exc:
                problems.append(dict(context, source=str(path), reason=str(exc)))
    pd.DataFrame(rows).to_csv(report_dir / "input_coverage.csv", index=False)
    write_json(report_dir / "input_problems.json", problems)
    if problems:
        raise ValueError(f"{len(problems)} input problems; see input_problems.json and input_coverage.csv")
    # Guard sample balance across classes, phases, and runs for paired interpretation.
    for run in (1, 2, 3, 101, 102, 103):
        selected = [s for s in samples if s["run"] == run]
        expected_labels = {f"{condition}_img{position}" for condition in condition_map.values()
                           for position in (1, 2)}
        counts_by_label = {label: sum(s["label"] == label for s in selected)
                           for label in expected_labels}
        if any(count != 2 for count in counts_by_label.values()):
            raise ValueError(f"Run {run}: expected two presentations per cue identity and position; {counts_by_label}")
        counts = [sum(s["label"].startswith(prefixes) for s in selected)
                  for prefixes in (("B", "D"), ("A", "C"))]
        if counts != [24, 24]:
            raise ValueError(f"Run {run}: expected 24 presentations per cue class, found {counts}")
    return {"n_samples": len(samples), "n_runs": 6, "samples_per_subset": {"all": 288,
            "conditioned_only": 144, "preconditioned_only": 144}}


def runtime_versions():
    return dict(python=sys.version, **{name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "nibabel", "scikit-learn", "nilearn")})


def map_name(sub, subset):
    return f"{sub}_model-odor_value_subset-{subset}.nii.gz"


def last_attempt(subject_dir):
    attempts = sorted(subject_dir.glob("attempt-*"))
    return attempts[-1] if attempts else None


def validate_map(path, mask_image):
    image = nib.load(path)
    if not same_grid(image, mask_image):
        raise ValueError(f"Map grid mismatch: {path}")
    values = np.asarray(image.dataobj)[np.asarray(mask_image.dataobj) > 0]
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite searchlight coefficients: {path}")


def run_subject(folder, plan, sub):
    subject_dir = folder / "subjects" / sub
    subject_dir.mkdir(parents=True, exist_ok=True)
    lock = subject_dir / ".running"
    lock.mkdir()  # Refuse overlapping jobs for the same participant.
    try:
        previous = last_attempt(subject_dir)
        if previous and (previous / "status.json").exists():
            if json.loads((previous / "status.json").read_text())["status"] == "success":
                print(f"{sub}: already succeeded; check validates existing maps")
                return True
        attempt = subject_dir / f"attempt-{len(list(subject_dir.glob('attempt-*'))) + 1:03d}"
        attempt.mkdir()
        status = dict(sub=sub, status="running", maps={})
        write_json(attempt / "status.json", status)
        with (attempt / "run.log").open("w", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                status["versions"] = runtime_versions()
                mask_image = nib.load(folder / "gm_mask.nii")
                if sha256(Path(plan["base_dir"]) / "masks/gm_0.1.nii") != plan["gm_source_sha256"]:
                    raise ValueError("Data-machine GM mask changed since preparation")
                status["input_summary"] = validate_inputs(Path(plan["base_dir"]), sub, mask_image, attempt)
                from rsa_imagewise.searchlight import searchlight_rsa_single_subject
                ratings = {p: load_subject_pleasantness_ratings(sub, p, str(folder / "ratings.csv"))
                           for p in PHASES}
                for subset in SUBSETS:
                    print(f"Running {sub}: {subset}", flush=True)
                    searchlight_rsa_single_subject(
                        sub=sub, base_dir=plan["base_dir"], session_tag=PHASES,
                        radius=plan["radius_mm"], n_jobs=plan["n_jobs"],
                        button_scope="second_image_only", cue_subset=subset,
                        center_within_run=False, output_dir=str(attempt / "maps"),
                        targets=["odor_value"], ratings_by_phase=ratings)
                    path = attempt / "maps" / map_name(sub, subset)
                    validate_map(path, mask_image)
                    status["maps"][subset] = dict(path=str(path.resolve()), sha256=sha256(path))
                status["status"] = "success"
            except Exception as exc:
                traceback.print_exc()
                status.update(status="failed", reason=str(exc))
            write_json(attempt / "status.json", status)
        print(f"{sub}: {status['status']} ({attempt / 'run.log'})", flush=True)
        return status["status"] == "success"
    finally:
        lock.rmdir()


def run(args):
    folder, plan = load_plan(args.run_dir)
    subjects = args.subjects or plan["subjects"]
    if len(subjects) != len(set(subjects)) or set(subjects) - set(plan["subjects"]):
        raise ValueError("Requested subjects must be unique members of the prepared cohort")
    ok = True
    for sub in subjects:
        # Continue after a participant fails so every failure can be inspected.
        try:
            succeeded = run_subject(folder, plan, sub)
        except Exception:
            traceback.print_exc()
            succeeded = False
        ok = succeeded and ok
    return 0 if ok else 1


def check(args):
    folder, plan = load_plan(args.run_dir)
    ready = folder / "group_inputs.json"
    if ready.exists():
        ready.unlink()  # Never leave a stale group-ready manifest after failed validation.
    reports, maps, versions = [], {s: [] for s in SUBSETS}, []
    mask_image = nib.load(folder / "gm_mask.nii")
    for sub in plan["subjects"]:
        subject_dir = folder / "subjects" / sub
        attempt = last_attempt(subject_dir)
        row = dict(sub=sub, status="failed", reason="")
        try:
            if (subject_dir / ".running").exists():
                raise ValueError("Participant job is still running")
            if not attempt:
                raise ValueError("Not run")
            status = json.loads((attempt / "status.json").read_text())
            if status["status"] != "success":
                raise ValueError(status.get("reason", status["status"]))
            for subset in SUBSETS:
                path = attempt / "maps" / map_name(sub, subset)
                validate_map(path, mask_image)
                if sha256(path) != status["maps"][subset]["sha256"]:
                    raise ValueError(f"Map changed after execution: {path}")
                maps[subset].append(str(path.resolve()))
            versions.append(status["versions"])
            row.update(status="success", attempt=str(attempt))
        except Exception as exc:
            row["reason"] = str(exc)
        reports.append(row)
    pd.DataFrame(reports).to_csv(folder / "check_report.csv", index=False)
    problems = [r for r in reports if r["status"] != "success"]
    if problems or len(plan["subjects"]) < 2:
        print(f"Group analysis blocked: {len(problems)} incomplete participants. See {folder / 'check_report.csv'}")
        return 1
    if any(v != versions[0] for v in versions):
        raise ValueError("Runtime/library versions differ across participants; use one environment")
    payload = dict(subjects=plan["subjects"], subsets=list(SUBSETS), maps=maps,
                   gm_mask=str((folder / "gm_mask.nii").resolve()),
                   analysis_config_sha256=sha256(folder / "analysis_config.json"),
                   smoothing_fwhm_mm=6, versions=versions[0],
                   map_sha256={p: sha256(p) for paths in maps.values() for p in paths})
    write_json(ready, payload)
    print(f"All three analyses ready for SPM: n={len(plan['subjects'])}; {ready}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Freeze cohort, ratings, settings, and job commands")
    p.add_argument("--base_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True, help="Fresh run directory on data computer")
    p.add_argument("--ratings_file", type=Path)
    p.add_argument("--subjects", nargs="+")
    p.add_argument("--radius", type=float, default=8.)
    p.add_argument("--n_jobs", type=int, default=4)
    for name, description in (("run", "Run all or selected planned participants"),
                              ("check", "Validate the complete shared cohort for SPM")):
        p = commands.add_parser(name, help=description)
        p.add_argument("--run_dir", type=Path, required=True)
        if name == "run":
            p.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.radius <= 0 or not np.isfinite(args.radius) or args.n_jobs < 1:
            parser.error("Radius must be positive and finite; n_jobs must be at least 1")
        prepare(args)
        return 0
    return run(args) if args.command == "run" else check(args)


if __name__ == "__main__":
    sys.exit(main())
