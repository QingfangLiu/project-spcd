#!/usr/bin/env python3
"""Aggregate subject-level odor-template alignment within one ROI scope."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
RNG_SEED = 20260731
CUE_METRIC_NULLS = {
    "matching_correlation_advantage": 0.0,
    "fisher_z_matching_advantage": 0.0,
    "mean_matching_correlation": 0.0,
    "mean_nonmatching_correlation": 0.0,
    "cue_nearest_prototype_balanced_accuracy": 0.5,
    "cue_nearest_prototype_accuracy": 0.5,
    "cue_nearest_prototype_auc": 0.5,
}
PASSIVE_METRIC_NULLS = {
    "passive_looro_matching_correlation_advantage": 0.0,
    "passive_looro_balanced_accuracy": 0.5,
    "passive_looro_accuracy": 0.5,
    "passive_looro_auc": 0.5,
    "passive_axis_run_similarity": 0.0,
}
METRIC_NULLS = {**CUE_METRIC_NULLS, **PASSIVE_METRIC_NULLS}


def _safe_wilcoxon(
    centered: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    centered = np.round(np.asarray(centered, dtype=float), decimals=12)
    if centered.size == 0 or np.all(centered == 0):
        return np.nan, np.nan
    result = stats.wilcoxon(
        centered,
        zero_method="wilcox",
        alternative=alternative,
    )
    return float(result.statistic), float(result.pvalue)


def _sign_test(centered: np.ndarray, alternative: str) -> float:
    positive = int((centered > 0).sum())
    negative = int((centered < 0).sum())
    if positive + negative == 0:
        return np.nan
    return float(
        stats.binomtest(
            positive,
            n=positive + negative,
            p=0.5,
            alternative=alternative,
        ).pvalue
    )


def _sign_flip_p(
    centered: np.ndarray,
    *,
    alternative: str,
    n_permutations: int = 100000,
) -> float:
    centered = np.asarray(centered, dtype=float)
    observed = float(centered.mean())
    rng = np.random.default_rng(RNG_SEED)
    exceed = 0
    remaining = n_permutations
    while remaining:
        batch = min(10000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(batch, len(centered)))
        permuted = (signs * centered).mean(axis=1)
        if alternative == "greater":
            exceed += int((permuted >= observed).sum())
        elif alternative == "two-sided":
            exceed += int((np.abs(permuted) >= abs(observed)).sum())
        else:
            raise ValueError(f"Unsupported sign-flip alternative: {alternative}")
        remaining -= batch
    return float((exceed + 1) / (n_permutations + 1))


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int = 10000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    means = np.empty(n_bootstrap, dtype=float)
    remaining = n_bootstrap
    offset = 0
    while remaining:
        batch = min(1000, remaining)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        means[offset : offset + batch] = values[indices].mean(axis=1)
        offset += batch
        remaining -= batch
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _one_sample_summary(
    values,
    *,
    metric: str,
    null_value: float,
    cue_subset: str,
    test: str,
    contrast: str,
    directional: bool,
) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(f"At least two values are required for {contrast}.")
    centered = values - null_value
    alternative = "greater" if directional else "two-sided"
    sd = float(np.std(centered, ddof=1))
    if sd <= 1e-15:
        t_stat = np.nan
        p_t_two_sided = np.nan
    else:
        t_result = stats.ttest_1samp(values, popmean=null_value)
        t_stat = float(t_result.statistic)
        p_t_two_sided = float(t_result.pvalue)
    wilcoxon_w, wilcoxon_p = _safe_wilcoxon(centered, alternative)
    ci_low, ci_high = _bootstrap_mean_ci(values)
    return {
        "test": test,
        "cue_subset": cue_subset,
        "metric": metric,
        "contrast": contrast,
        "alternative": alternative,
        "null_value": null_value,
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "mean_minus_null": float(centered.mean()),
        "sd": sd,
        "se": float(sd / np.sqrt(len(values))),
        "bootstrap_mean_ci_low": ci_low,
        "bootstrap_mean_ci_high": ci_high,
        "t": t_stat,
        "df": int(len(values) - 1),
        "p_t_two_sided": p_t_two_sided,
        "cohen_dz": float(centered.mean() / sd) if sd > 0 else np.nan,
        "wilcoxon_w": wilcoxon_w,
        "p_wilcoxon": wilcoxon_p,
        "p_sign": _sign_test(centered, alternative),
        "p_sign_flip_mean": _sign_flip_p(
            centered, alternative=alternative
        ),
        "n_above_null": int((centered > 0).sum()),
        "n_at_null": int((centered == 0).sum()),
        "n_below_null": int((centered < 0).sum()),
    }


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cue_subsets = sorted(df["cue_subset"].unique())
    for cue_subset in cue_subsets:
        subset = df.loc[df["cue_subset"] == cue_subset]
        for metric, null_value in CUE_METRIC_NULLS.items():
            directional = metric in {
                "matching_correlation_advantage",
                "fisher_z_matching_advantage",
                "cue_nearest_prototype_balanced_accuracy",
                "cue_nearest_prototype_accuracy",
                "cue_nearest_prototype_auc",
            }
            rows.append(
                _one_sample_summary(
                    subset[metric],
                    metric=metric,
                    null_value=null_value,
                    cue_subset=cue_subset,
                    test="one_sample",
                    contrast=f"{metric}_vs_{null_value:g}",
                    directional=directional,
                )
            )

    passive = df.sort_values("cue_subset").drop_duplicates("sub")
    for metric, null_value in PASSIVE_METRIC_NULLS.items():
        rows.append(
            _one_sample_summary(
                passive[metric],
                metric=metric,
                null_value=null_value,
                cue_subset="passive_odor_qc",
                test="one_sample",
                contrast=f"{metric}_vs_{null_value:g}",
                directional=True,
            )
        )

    expected = {"conditioned_odor_only", "preconditioned_odor_only"}
    if expected.issubset(cue_subsets):
        for metric in CUE_METRIC_NULLS:
            wide = df.pivot(index="sub", columns="cue_subset", values=metric)
            difference = (
                wide["conditioned_odor_only"]
                - wide["preconditioned_odor_only"]
            ).dropna()
            rows.append(
                _one_sample_summary(
                    difference,
                    metric=metric,
                    null_value=0.0,
                    cue_subset="conditioned_minus_preconditioned",
                    test="paired_difference",
                    contrast=f"conditioned_minus_preconditioned_{metric}",
                    directional=False,
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate one ROI/hemisphere template-alignment CSV."
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
    )
    parser.add_argument(
        "--voxel_loss_flag_threshold", type=float, default=0.20
    )
    args = parser.parse_args()

    input_csv = args.input_csv.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else input_csv.with_name("group_template_alignment_summary.csv")
    )
    df = pd.read_csv(input_csv)
    required = {
        "sub",
        "roi",
        "hemisphere",
        "cue_subset",
        "n_voxels_raw",
        "n_voxels_dropped_nonfinite",
        *METRIC_NULLS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")
    if df.duplicated(["sub", "roi", "hemisphere", "cue_subset"]).any():
        raise ValueError("Input contains duplicate subject analysis rows.")
    rois = df["roi"].dropna().unique()
    hemispheres = df["hemisphere"].dropna().unique()
    if len(rois) != 1 or len(hemispheres) != 1:
        raise ValueError("Aggregate exactly one ROI and hemisphere at a time.")

    subjects_before = int(df["sub"].nunique())
    df = df.loc[~df["sub"].isin(args.skip_subs)].copy()
    if df.empty:
        raise ValueError("No rows remain after subject exclusions.")
    subjects_after = int(df["sub"].nunique())
    loss = df["n_voxels_dropped_nonfinite"] / df["n_voxels_raw"]
    high_loss = sorted(
        df.loc[loss >= args.voxel_loss_flag_threshold, "sub"].unique()
    )
    passive_below = sorted(
        df.loc[df["passive_looro_balanced_accuracy"] < 0.5, "sub"].unique()
    )

    summary = _summarize(df)
    summary.insert(0, "roi", rois[0])
    summary.insert(1, "hemisphere", hemispheres[0])
    summary.insert(2, "input_csv", str(input_csv))
    summary.insert(3, "subjects_before_exclusion", subjects_before)
    summary.insert(4, "subjects_after_exclusion", subjects_after)
    summary.insert(5, "excluded_subjects", ",".join(args.skip_subs))
    summary.insert(6, "qc_high_voxel_loss_subjects", ",".join(high_loss))
    summary.insert(7, "qc_passive_below_chance_subjects", ",".join(passive_below))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"Saved group summary: {output_csv}")


if __name__ == "__main__":
    main()
