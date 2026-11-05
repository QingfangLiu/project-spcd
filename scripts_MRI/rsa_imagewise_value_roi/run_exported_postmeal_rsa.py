#!/usr/bin/env python3
"""Explore postmeal value RSA from existing raw ROI exports, without editing figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsa_imagewise.neural_rdm import compute_correlation_rdm
from rsa_imagewise.rsa_stats import rsa_multiple_regression, vectorize_lower_triangle
from rsa_imagewise_odor_identity.run_roi_rsa import _center_within_run
from rsa_imagewise_value_roi.aggregate_roi_rsa import (
    DEFAULT_SKIP_SUBS, ROI_NAMES, summarize_subject_betas,
)
from rsa_imagewise_value_roi.run_roi_rsa import MODELS, build_models


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite p values in correction family")
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def fit_phase(raw, meta, phase, ratings):
    """Use one phase's ratings, finite voxels, and run means only."""
    selected = meta.loc[meta.session == phase].copy()
    if len(selected) != 144 or selected.groupby("run").size().tolist() != [48] * 3:
        raise ValueError(f"Expected three complete {phase} runs with 144 presentations")
    if set(selected.img_index) != {1, 2} or selected.condition.nunique() != 12:
        raise ValueError("Expected both presentation positions and all 12 cue identities")
    if not selected.button_metadata_ok.all() or not selected.beta_read_status.eq("ok").all():
        raise ValueError("Invalid selected sample metadata")
    models, runs, pairs, rank = build_models(selected.to_dict("records"), {phase: ratings})
    if int(pairs.sum()) != 6912:
        raise ValueError("Expected 6,912 unique cross-run pairs")
    selected_raw = raw[selected.index.to_numpy()]
    finite = np.isfinite(selected_raw).all(axis=0)
    qc = dict(n_samples=len(selected), n_runs=3, n_pairs=int(pairs.sum()),
              n_voxels_raw=raw.shape[1], n_voxels_used=int(finite.sum()),
              n_voxels_dropped=int((~finite).sum()),
              n_all_nonfinite_samples=int((~np.isfinite(selected_raw).any(axis=1)).sum()),
              design_rank=rank)
    if finite.sum() < 3:
        return None, qc, None
    patterns = _center_within_run(selected_raw[:, finite], runs)
    if np.any(np.linalg.norm(patterns - patterns.mean(axis=1, keepdims=True), axis=1) == 0):
        raise ValueError("Constant spatial pattern: correlation distance is undefined")
    neural = compute_correlation_rdm(patterns)
    np.testing.assert_allclose(neural, squareform(pdist(patterns, metric="correlation")),
                               rtol=0, atol=1e-12)
    if not np.isfinite(neural).all():
        raise ValueError("Nonfinite neural RDM")
    betas = rsa_multiple_regression(neural, models, pair_mask=pairs)
    design = np.column_stack([np.ones(int(pairs.sum()))] + [
        vectorize_lower_triangle(models[m], pairs) for m in MODELS])
    response = vectorize_lower_triangle(neural, pairs)
    independent = np.linalg.lstsq(design, response, rcond=None)[0][1:]
    np.testing.assert_allclose([betas[m] for m in MODELS], independent, rtol=0, atol=1e-12)
    diagnostics = dict(
        independent_ols_max_error=float(np.max(np.abs(
            independent - np.array([betas[m] for m in MODELS])))),
        model_design_condition_number=float(np.linalg.cond(design)),
    )
    return betas, qc, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    root = project / "rsa_roi/probe_imagewise_value"
    raw_root = root / "raw_patterns"
    output = (args.output_dir.resolve() if args.output_dir else
              root / "exploratory_postmeal_v1")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")

    export = json.loads((raw_root / "export_config.json").read_text())
    ratings_table = pd.read_csv(raw_root / "ratings.csv")
    reference = pd.read_csv(root / "exploratory_premeal_v1/subject_betas.csv")
    reference = reference.loc[reference.design == "centered_baseline"].set_index(["sub", "roi"])
    cohort = sorted(pd.read_csv(root / "both/subject_betas.csv")["sub"].unique())
    if len(cohort) != 29 or set(cohort) & set(DEFAULT_SKIP_SUBS):
        raise ValueError("Expected the established 29-participant cohort")

    rows, inclusion, coverage, validations, sources = [], [], [], [], {}
    for sub in export["subjects"]:
        report = dict(sub=sub, included=False, reason="")
        if sub in DEFAULT_SKIP_SUBS:
            report["reason"] = "paper_exclusion"
            inclusion.append(report)
            continue
        rt = ratings_table.loc[(ratings_table["sub"] == sub) &
                               (ratings_table.session == "postmeal")]
        if (len(rt) != 3 or set(rt.odor_group) != {"odor1", "odor2", "clean"} or
                not rt.rating_available.all() or not np.isfinite(rt.pleasantness).all()):
            report["reason"] = "missing_postmeal_ratings"
            inclusion.append(report)
            continue
        ratings = dict(zip(rt.odor_group, rt.pleasantness))
        meta_path = raw_root / sub / "samples.csv"
        meta = pd.read_csv(meta_path)
        if not np.array_equal(meta.sample_index, np.arange(len(meta))):
            raise ValueError(f"Nonsequential sample indices: {sub}")
        sources[str(meta_path.relative_to(project))] = sha256(meta_path)
        pending, errors = [], []
        for roi in ROI_NAMES:
            archive_path = raw_root / sub / f"roi-{roi}_patterns.npz"
            sources[str(archive_path.relative_to(project))] = sha256(archive_path)
            with np.load(archive_path, allow_pickle=False) as archive:
                if str(archive["sub"]) != sub or str(archive["roi"]) != roi:
                    raise ValueError(f"Archive identity mismatch: {archive_path}")
                for key, column in [("sample_index", "sample_index"), ("run_ids", "run"),
                                    ("phase_labels", "session"), ("sample_labels", "sample_label"),
                                    ("img_index", "img_index"), ("button_labels", "button_label")]:
                    np.testing.assert_array_equal(archive[key], meta[column].to_numpy())
                raw = archive["raw_patterns"]
            betas, qc, diagnostic = fit_phase(raw, meta, "postmeal", ratings)
            coverage.append(dict(sub=sub, roi=roi, session="postmeal", **qc))
            if betas is None:
                errors.append(f"{roi}: fewer than three finite postmeal voxels")
                continue
            pending.append(dict(sub=sub, roi=roi, session="postmeal", design="centered_baseline",
                                cue_subset="all", center_within_run=True, **qc, **diagnostic,
                                **{m + "_beta": betas[m] for m in MODELS}))
            # Reproduce the already-saved premeal analysis using the same machinery.
            pre = ratings_table.loc[(ratings_table["sub"] == sub) &
                                    (ratings_table.session == "premeal")]
            pre_betas, _, _ = fit_phase(raw, meta, "premeal", dict(zip(pre.odor_group, pre.pleasantness)))
            expected = reference.loc[(sub, roi), "odor_value_beta"]
            error = abs(pre_betas["odor_value"] - expected)
            if error > 1e-12:
                raise ValueError(f"Saved premeal coefficient does not reproduce: {sub}, {roi}")
            validations.append(dict(sub=sub, roi=roi, premeal_reference_error=error,
                                    **diagnostic))
        if errors:
            report["reason"] = "; ".join(errors)
        else:
            report.update(included=True, reason="eligible_postmeal_all_three_rois")
            rows.extend(pending)
        inclusion.append(report)

    coefficients = pd.DataFrame(rows)
    if sorted(coefficients["sub"].unique()) != cohort:
        raise ValueError("Postmeal cohort differs from the approved fixed cohort; inspect QC")
    summary = summarize_subject_betas(coefficients)
    summary["session"] = "postmeal"  # Shared summarizer otherwise labels combined-phase analyses.
    summary["design"] = "centered_baseline"
    summary["exploratory"] = True
    for _, row in summary.iterrows():
        values = coefficients.loc[coefficients.roi == row.roi, "odor_value_beta"]
        check = stats.ttest_1samp(values, 0, alternative="greater")
        np.testing.assert_allclose([row.t, row.p_greater_than_zero],
                                   [check.statistic, check.pvalue], rtol=0, atol=1e-12)

    family_path = root / "retained_models_first_cue_v1/cumulative_group_tests.csv"
    family = pd.read_csv(family_path)
    previous_family_size = len(family)
    if previous_family_size != 111:
        raise ValueError("Expected the existing 111-test exploratory family")
    family = family.drop(columns=[c for c in family if c.startswith("p_holm_all_")])
    new_tests = summary[["roi", "n", "p_greater_than_zero"]].rename(
        columns={"p_greater_than_zero": "p_one_sided"})
    new_tests = new_tests.assign(phase="postmeal", cohort="common_29",
                                 design="centered_baseline", method="t_test")
    family = pd.concat([family, new_tests], ignore_index=True)
    keys = ["phase", "cohort", "design", "roi", "method"]
    if family.duplicated(keys).any():
        raise ValueError("Duplicate tests in cumulative family")
    correction_name = f"p_holm_all_{len(family)}_tests"
    family[correction_name] = holm(family.p_one_sided)
    summary[correction_name] = family.tail(3)[correction_name].to_numpy()

    config = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), exploratory=True,
        objective="General expected value across all cues after devaluation",
        source_raw_export=str(raw_root), subjects=cohort, n_subjects=len(cohort),
        excluded_paper_subjects=list(DEFAULT_SKIP_SUBS), roi_names=list(ROI_NAMES),
        phase="postmeal", image_positions=[1, 2], clean_air_included=True,
        rating_standardization="Three postmeal outcome ratings within participant; z-score ddof=0",
        centering="Voxel mean over all 48 selected presentations within each postmeal run",
        finite_voxels="Finite across all selected postmeal samples, before centering",
        averaging=False, neural_distance="1 - Pearson correlation",
        models=list(MODELS), phase_regressor=False, odor_identity_regressor=False,
        regression="OLS with intercept", pairs="Unique different-run postmeal pairs only; 6912 per subject",
        group_test="One-sided one-sample t test of participant value coefficients against zero",
        confidence_intervals="Two-sided 95% t intervals for group mean coefficients",
        correction="Holm within these three ROIs; also retain the full 114-test exploratory family",
        verification="Archive/sample alignment, independent correlation distances and OLS, saved premeal baseline reproduction, independent group t tests",
        source_hashes={**sources, str((raw_root / "ratings.csv").relative_to(project)): sha256(raw_root / "ratings.csv"),
                       str((raw_root / "export_config.json").relative_to(project)): sha256(raw_root / "export_config.json"),
                       str((root / "exploratory_premeal_v1/subject_betas.csv").relative_to(project)): sha256(root / "exploratory_premeal_v1/subject_betas.csv"),
                       str((root / "both/subject_betas.csv").relative_to(project)): sha256(root / "both/subject_betas.csv"),
                       str(family_path.relative_to(project)): sha256(family_path)},
        script_sha256=sha256(__file__),
        versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__),
    )
    output.mkdir(parents=True)
    coefficients.to_csv(output / "subject_betas.csv", index=False)
    summary.to_csv(output / "group_summary.csv", index=False)
    pd.DataFrame(inclusion).to_csv(output / "subject_inclusion.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "coverage_qc.csv", index=False)
    pd.DataFrame(validations).to_csv(output / "verification.csv", index=False)
    family.to_csv(output / "cumulative_group_tests.csv", index=False)
    (output / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary[["roi", "n", "mean", "t", "p_greater_than_zero",
                   "p_greater_than_zero_holm_3_rois", correction_name]].to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
