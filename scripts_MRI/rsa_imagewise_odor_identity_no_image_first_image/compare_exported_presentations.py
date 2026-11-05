#!/usr/bin/env python3
"""Compare first-only and both-position premeal identity RSA from raw ROI exports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.spatial.distance import pdist, squareform

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsa_imagewise_odor_identity.aggregate_roi_rsa import _one_sample_summary
from rsa_imagewise_odor_identity_no_image_first_image.models import create_odor_identity_rdm
from rsa_imagewise_odor_identity_no_image_first_image.neural_rdm import compute_correlation_rdm
from rsa_imagewise_odor_identity_no_image_first_image.run_roi_rsa import (
    _cross_run_pair_mask, _prepare_analysis_patterns, vectorize_lower_triangle,
)
from rsa_imagewise_odor_identity_no_image_first_image.run_roi_rsa_spearman import (
    SPEARMAN_COL, compute_rank_rsa_statistics,
)


PAPER_EXCLUSIONS = ("sub-19", "sub-31", "sub-33")
ROIS = ("OFC_K2_cluster1_medial_posterior_lateral",
        "OFC_K2_cluster2_central_anterior_lateral")
SUBSETS = {"conditioned_odor_only": "conditioned",
           "preconditioned_odor_only": "preconditioned"}
SCOPES = ("first_only", "both_images")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(values):
    p = np.asarray(values, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite p values")
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return result


def analyze(raw, meta, cue_type, scope):
    selected = meta.loc[(meta.session == "premeal") & (meta.cue_type == cue_type)
                        & meta.odor_group.isin(["odor1", "odor2"])].copy()
    if scope == "first_only":
        selected = selected.loc[selected.img_index == 1]
    samples_per_run = 8 if scope == "first_only" else 16
    n_runs = selected.run.nunique()
    if n_runs not in (2, 3) or not selected.groupby("run").size().eq(samples_per_run).all():
        raise ValueError("Unexpected sample/run counts")
    if selected.condition.nunique() != 4 or not selected.beta_read_status.eq("ok").all():
        raise ValueError("Missing cue identities or failed beta reads")
    # Balance each run across both odors and, when selected, both positions.
    expected_per_cell = 4
    if not selected.groupby(["run", "odor_group", "img_index"]).size().eq(expected_per_cell).all():
        raise ValueError("Unbalanced odor/presentation samples")
    runs = selected.run.to_numpy()
    raw_selected = raw[selected.index.to_numpy()]
    _, patterns, _, qc = _prepare_analysis_patterns(
        raw_selected, runs, center_within_run=True, drop_nonfinite_voxels=True)
    if patterns.shape[1] < 3 or np.any(np.linalg.norm(
            patterns - patterns.mean(axis=1, keepdims=True), axis=1) == 0):
        raise ValueError("Insufficient voxels or constant spatial pattern")
    neural = compute_correlation_rdm(patterns)
    np.testing.assert_allclose(neural, squareform(pdist(patterns, metric="correlation")),
                               rtol=0, atol=1e-12)
    model = create_odor_identity_rdm(selected.sample_label)
    groups = selected.odor_group.to_numpy()
    np.testing.assert_array_equal(model, (groups[:, None] != groups[None, :]).astype(float))
    pairs = _cross_run_pair_mask(runs)
    expected_pairs = (n_runs * (n_runs - 1) // 2) * samples_per_run ** 2
    if int(pairs.sum()) != expected_pairs:
        raise ValueError("Unexpected cross-run pair count")
    result = compute_rank_rsa_statistics(neural, model, pairs)
    independent_rho = stats.spearmanr(vectorize_lower_triangle(neural, pairs),
                                     vectorize_lower_triangle(model, pairs)).statistic
    np.testing.assert_allclose(result[SPEARMAN_COL], independent_rho, rtol=0, atol=1e-12)
    result.update(n_samples=len(selected), n_runs=int(n_runs), n_pairs=int(pairs.sum()),
                  runs=",".join(str(r) for r in np.unique(runs)), **qc,
                  independent_spearman_error=abs(result[SPEARMAN_COL] - independent_rho))
    return result


def summarize(values, contrast):
    row = _one_sample_summary(values, "one_sample", "odor_identity_spearman", contrast)
    half = stats.t.ppf(.975, row["df"]) * row["se"]
    row.update(ci95_low=row["mean"] - half, ci95_high=row["mean"] + half)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    raw_root = project / "rsa_roi/probe_imagewise_value/raw_patterns"
    reference_root = project / ("rsa_roi/probe_imagewise_odor_identity_no_image_first_image/"
                                "premeal_ofc_k2_spearman")
    output = (args.output_dir.resolve() if args.output_dir else project /
              "rsa_roi/odor_identity_presentation_comparison/premeal_ofc_k2_v1")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    source_hashes, reference_parts = {}, []
    for roi in ROIS:
        ref_path = reference_root / roi.removeprefix("OFC_K2_") / f"roi-{roi}_subject_spearman.csv"
        ref = pd.read_csv(ref_path)
        ref = ref.loc[~ref["sub"].isin(PAPER_EXCLUSIONS)]
        if not ref.center_within_run.all() or not ref.drop_nonfinite_voxels.all():
            raise ValueError("Reference settings differ from planned comparison")
        reference_parts.append(ref)
        source_hashes[str(ref_path.relative_to(project))] = sha256(ref_path)
    reference = pd.concat(reference_parts, ignore_index=True)
    if reference.duplicated(["sub", "roi", "cue_subset"]).any():
        raise ValueError("Duplicate first-image reference coefficients")
    subjects = sorted(reference["sub"].unique())
    expected = pd.MultiIndex.from_product([subjects, ROIS, SUBSETS],
                                         names=["sub", "roi", "cue_subset"])
    reference = reference.set_index(["sub", "roi", "cue_subset"])
    if len(subjects) != 31 or set(reference.index) != set(expected):
        raise ValueError("Expected all 31 paper participants and all four ROI/cue combinations")

    export = json.loads((raw_root / "export_config.json").read_text())
    if not set(subjects).issubset(export["exported_subjects"]):
        raise ValueError("Missing raw exports")
    source_hashes[str((raw_root / "export_config.json").relative_to(project))] = sha256(raw_root / "export_config.json")
    rows, validations = [], []
    for counter, sub in enumerate(subjects, 1):
        meta_path = raw_root / sub / "samples.csv"
        source_hashes[str(meta_path.relative_to(project))] = sha256(meta_path)
        meta = pd.read_csv(meta_path)
        np.testing.assert_array_equal(meta.sample_index, np.arange(len(meta)))
        if set(meta["sub"]) != {sub}:
            raise ValueError("Sample participant mismatch")
        for roi in ROIS:
            archive_path = raw_root / sub / f"roi-{roi}_patterns.npz"
            source_hashes[str(archive_path.relative_to(project))] = sha256(archive_path)
            with np.load(archive_path, allow_pickle=False) as archive:
                if str(archive["sub"]) != sub or str(archive["roi"]) != roi:
                    raise ValueError("Archive identity mismatch")
                for field, column in [("sample_index", "sample_index"), ("phase_labels", "session"),
                                      ("run_ids", "run"), ("sample_labels", "sample_label"),
                                      ("img_index", "img_index"), ("odor_index", "odor_index")]:
                    np.testing.assert_array_equal(archive[field], meta[column].to_numpy())
                raw = archive["raw_patterns"]
            for cue_subset, cue_type in SUBSETS.items():
                for scope in SCOPES:
                    result = analyze(raw, meta, cue_type, scope)
                    if scope == "first_only":
                        original = reference.loc[(sub, roi, cue_subset)]
                        error = abs(result[SPEARMAN_COL] - original[SPEARMAN_COL])
                        np.testing.assert_allclose(result[SPEARMAN_COL], original[SPEARMAN_COL],
                                                   rtol=0, atol=1e-12)
                        for key in ["n_samples", "n_runs", "n_pairs", "n_voxels_used"]:
                            if result[key] != original[key]:
                                raise ValueError(f"Reference QC mismatch: {sub}, {roi}, {key}")
                        validations.append(dict(sub=sub, roi=roi, cue_subset=cue_subset,
                                                first_only_reference_error=error))
                    rows.append(dict(sub=sub, roi=roi, cue_subset=cue_subset, scope=scope,
                                     session="premeal", center_within_run=True, **result))
        if counter % 10 == 0 or counter == len(subjects):
            print(f"Processed {counter}/{len(subjects)} participants", flush=True)

    coefficients = pd.DataFrame(rows)
    summaries, paired = [], []
    for (roi, cue_subset), group in coefficients.groupby(["roi", "cue_subset"], sort=False):
        for scope, g in group.groupby("scope", sort=False):
            row = summarize(g[SPEARMAN_COL], "greater_than_zero")
            summaries.append(dict(roi=roi, cue_subset=cue_subset, scope=scope, **row))
        wide = group.pivot(index="sub", columns="scope", values=SPEARMAN_COL).loc[subjects]
        delta = wide.both_images - wide.first_only
        row = summarize(delta, "both_images_minus_first_only")
        check = stats.ttest_rel(wide.both_images, wide.first_only)
        np.testing.assert_allclose([row["t"], row["p_two_sided"]],
                                   [check.statistic, check.pvalue], rtol=0, atol=1e-12)
        paired.append(dict(roi=roi, cue_subset=cue_subset, **row))
    summary = pd.DataFrame(summaries)
    for scope, group in summary.groupby("scope"):
        summary.loc[group.index, "p_greater_than_zero_holm_4_roi_cue_tests"] = holm(group.p_greater_than_zero)
    summary["p_greater_than_zero_holm_8_scope_roi_cue_tests"] = holm(summary.p_greater_than_zero)
    paired = pd.DataFrame(paired)
    paired["p_two_sided_holm_4_comparisons"] = holm(paired.p_two_sided)
    config = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), exploratory=True,
        objective="Sensitivity of paper odor-identity RSA to including both probe presentation positions",
        subjects=subjects, n_subjects=31, paper_exclusions=list(PAPER_EXCLUSIONS),
        phase="premeal", rois=list(ROIS), cue_subsets=SUBSETS, scopes=list(SCOPES),
        selection="Conditioned and preconditioned food-odor cues separately; clean-air cues excluded",
        averaging=False, ratings_required=False,
        preprocessing="For each participant/ROI/cue subset/scope, remove voxels nonfinite in any selected sample, then center each voxel within run over selected samples",
        neural_distance="1 - Pearson correlation", model="Binary odor identity: 0 same odor, 1 different odor",
        nuisance_rdms=[], model_fit="Spearman correlation on unique cross-run pairs",
        inference="Participant-level one-sided one-sample t tests against zero; paired two-sided tests of both minus first coefficients",
        confidence_intervals="Two-sided 95% t intervals of participant means and paired differences",
        correction="Holm across four ROI/cue tests per scope; additionally across eight tests combining scopes; paired comparisons corrected separately across four tests",
        interpretation="Both-image analysis is exploratory and is not an independent replication; second presentations include a later decision context",
        verification="All 124 first-image coefficients and sample/voxel/pair counts match saved paper results; correlation distances and Spearman coefficients independently checked; paired tests independently checked",
        source_hashes=source_hashes, script_sha256=sha256(__file__),
        versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__),
    )
    output.mkdir(parents=True)
    coefficients.to_csv(output / "subject_results.csv", index=False)
    summary.to_csv(output / "group_summary.csv", index=False)
    paired.to_csv(output / "paired_scope_comparisons.csv", index=False)
    pd.DataFrame(validations).to_csv(output / "verification.csv", index=False)
    pd.DataFrame([dict(sub=s, included=True,
                       reason="paper_participant; complete first/both comparison in both ROIs") for s in subjects]
                 + [dict(sub=s, included=False, reason="paper_exclusion") for s in PAPER_EXCLUSIONS]
                 ).to_csv(output / "subject_inclusion.csv", index=False)
    (output / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary[["roi", "cue_subset", "scope", "n", "mean", "t", "p_greater_than_zero",
                   "p_greater_than_zero_holm_4_roi_cue_tests"]].to_string(index=False))
    print("\nPaired differences (both minus first):")
    print(paired[["roi", "cue_subset", "mean", "t", "p_two_sided",
                  "p_two_sided_holm_4_comparisons"]].to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
