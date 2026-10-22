#!/usr/bin/env python3
"""Aggregate subject-level passive-to-cue ROI decoding results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
METRIC_NULLS = {
    "cue_balanced_accuracy": 0.5,
    "cue_accuracy": 0.5,
    "cue_auc": 0.5,
    "cue_mean_signed_distance": 0.0,
    "passive_cv_balanced_accuracy": 0.5,
    "passive_cv_accuracy": 0.5,
    "passive_cv_auc": 0.5,
    "passive_cv_mean_signed_distance": 0.0,
}


def _safe_wilcoxon(
    centered_values: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    if centered_values.size == 0 or np.all(centered_values == 0):
        return np.nan, np.nan
    result = stats.wilcoxon(
        centered_values,
        zero_method="wilcox",
        alternative=alternative,
    )
    return float(result.statistic), float(result.pvalue)


def _sign_test(
    centered_values: np.ndarray,
    alternative: str,
) -> float:
    positive = int((centered_values > 0).sum())
    negative = int((centered_values < 0).sum())
    n_nonzero = positive + negative
    if n_nonzero == 0:
        return np.nan
    return float(
        stats.binomtest(
            positive,
            n=n_nonzero,
            p=0.5,
            alternative=alternative,
        ).pvalue
    )


def _one_sample_summary(
    values,
    *,
    metric: str,
    null_value: float,
    cue_subset: str,
    test: str,
    contrast: str,
) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    centered = values - null_value
    n = int(values.size)
    mean = float(np.mean(values)) if n else np.nan
    median = float(np.median(values)) if n else np.nan
    mean_minus_null = (
        float(np.mean(centered)) if n else np.nan
    )
    sd = float(np.std(centered, ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 else np.nan
    df = n - 1

    if n > 1:
        t_result = stats.ttest_1samp(
            values,
            popmean=null_value,
            nan_policy="omit",
        )
        t_stat = float(t_result.statistic)
        p_two = float(t_result.pvalue)
        p_greater = float(stats.t.sf(t_stat, df))
        p_less = float(stats.t.cdf(t_stat, df))
        cohen_dz = (
            float(mean_minus_null / sd)
            if np.isfinite(sd) and sd > 0
            else np.nan
        )
        wilcoxon_w, p_wilcoxon_two = _safe_wilcoxon(
            centered,
            "two-sided",
        )
        _, p_wilcoxon_greater = _safe_wilcoxon(
            centered,
            "greater",
        )
        _, p_wilcoxon_less = _safe_wilcoxon(
            centered,
            "less",
        )
    else:
        t_stat = p_two = p_greater = p_less = cohen_dz = np.nan
        wilcoxon_w = p_wilcoxon_two = np.nan
        p_wilcoxon_greater = p_wilcoxon_less = np.nan

    return {
        "test": test,
        "cue_subset": cue_subset,
        "metric": metric,
        "contrast": contrast,
        "null_value": null_value,
        "n": n,
        "mean": mean,
        "median": median,
        "mean_minus_null": mean_minus_null,
        "sd": sd,
        "se": se,
        "t": t_stat,
        "df": df,
        "p_two_sided": p_two,
        "p_greater_than_null": p_greater,
        "p_less_than_null": p_less,
        "cohen_dz": cohen_dz,
        "wilcoxon_w": wilcoxon_w,
        "p_wilcoxon_two_sided": p_wilcoxon_two,
        "p_wilcoxon_greater_than_null": p_wilcoxon_greater,
        "p_wilcoxon_less_than_null": p_wilcoxon_less,
        "n_above_null": int((centered > 0).sum()),
        "n_below_null": int((centered < 0).sum()),
        "n_at_null": int((centered == 0).sum()),
        "p_sign_two_sided": _sign_test(centered, "two-sided"),
        "p_sign_greater_than_null": _sign_test(centered, "greater"),
        "p_sign_less_than_null": _sign_test(centered, "less"),
    }


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cue_subsets = sorted(df["cue_subset"].dropna().unique())

    for cue_subset in cue_subsets:
        subset_df = df.loc[df["cue_subset"] == cue_subset]
        for metric, null_value in METRIC_NULLS.items():
            rows.append(
                _one_sample_summary(
                    subset_df[metric],
                    metric=metric,
                    null_value=null_value,
                    cue_subset=cue_subset,
                    test="one_sample",
                    contrast=f"{metric}_greater_than_{null_value:g}",
                )
            )

    if {
        "conditioned_odor_only",
        "preconditioned_odor_only",
    }.issubset(cue_subsets):
        for metric in METRIC_NULLS:
            wide = df.pivot_table(
                index="sub",
                columns="cue_subset",
                values=metric,
                aggfunc="first",
            )
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
                    contrast=(
                        f"conditioned_minus_preconditioned_{metric}"
                    ),
                )
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test group-level passive-to-cue ROI decoding metrics against "
            "chance with t, Wilcoxon, and sign tests."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Subject results from the ROI cross-task decoder.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help=(
            "Group summary path. Default: replace _subject_results.csv with "
            "_group_summary.csv beside the input."
        ),
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
        help="Primary-analysis exclusions. Default: sub-19 sub-33.",
    )
    parser.add_argument(
        "--voxel_loss_flag_threshold",
        type=float,
        default=0.20,
        help=(
            "Report subjects losing at least this fraction of ROI voxels. "
            "Default: 0.20."
        ),
    )
    args = parser.parse_args()

    if not 0 <= args.voxel_loss_flag_threshold <= 1:
        raise ValueError(
            "--voxel_loss_flag_threshold must be between 0 and 1."
        )

    input_csv = args.input_csv.expanduser().resolve()
    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser().resolve()
    elif input_csv.name.endswith("_subject_results.csv"):
        output_csv = input_csv.with_name(
            input_csv.name.replace(
                "_subject_results.csv",
                "_group_summary.csv",
            )
        )
    else:
        output_csv = input_csv.with_name(
            input_csv.stem + "_group_summary.csv"
        )

    df = pd.read_csv(input_csv)
    if "phase" not in df.columns:
        df["phase"] = "premeal"
    phases = df["phase"].dropna().unique()
    if (
        df["phase"].isna().any()
        or len(phases) != 1
        or phases[0] not in {"premeal", "postmeal"}
    ):
        raise ValueError("Aggregate exactly one valid probe phase at a time.")
    phase = str(phases[0])
    if "hemisphere" not in df.columns:
        df["hemisphere"] = "bilateral"
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
        raise ValueError(
            f"Input CSV missing required columns: {sorted(missing)}"
        )
    if df.duplicated(["sub", "roi", "cue_subset"]).any():
        raise ValueError(
            "Input CSV has duplicate subject/ROI/cue-subset rows."
        )
    rois = sorted(df["roi"].dropna().astype(str).unique())
    if len(rois) != 1:
        raise ValueError(
            "Aggregate one ROI at a time; input contains: "
            + ", ".join(rois)
        )
    roi = rois[0]
    hemispheres = sorted(
        df["hemisphere"].dropna().astype(str).unique()
    )
    if len(hemispheres) != 1:
        raise ValueError(
            "Aggregate one hemisphere at a time; input contains: "
            + ", ".join(hemispheres)
        )
    hemisphere = hemispheres[0]

    subjects_before = int(df["sub"].nunique())
    if args.skip_subs:
        df = df.loc[~df["sub"].isin(args.skip_subs)].copy()
    subjects_after = int(df["sub"].nunique())
    if df.empty:
        raise ValueError("No subject rows remain after exclusions.")

    voxel_loss_fraction = (
        df["n_voxels_dropped_nonfinite"] / df["n_voxels_raw"]
    )
    high_voxel_loss = sorted(
        df.loc[
            voxel_loss_fraction >= args.voxel_loss_flag_threshold,
            "sub",
        ].unique()
    )
    passive_below_chance = sorted(
        df.loc[
            df["passive_cv_balanced_accuracy"] < 0.5,
            "sub",
        ].unique()
    )

    summary = _summarize(df)
    summary["phase"] = phase
    summary.insert(0, "roi", roi)
    summary.insert(1, "hemisphere", hemisphere)
    summary.insert(2, "input_csv", str(input_csv))
    summary.insert(3, "subjects_before_exclusion", subjects_before)
    summary.insert(4, "subjects_after_exclusion", subjects_after)
    summary.insert(5, "excluded_subjects", ",".join(args.skip_subs or []))
    summary.insert(
        6,
        "qc_voxel_loss_flag_threshold",
        args.voxel_loss_flag_threshold,
    )
    summary.insert(
        7,
        "qc_high_voxel_loss_subjects",
        ",".join(high_voxel_loss),
    )
    summary.insert(
        8,
        "qc_passive_cv_below_chance_subjects",
        ",".join(passive_below_chance),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print(f"Read subject results: {input_csv}")
    print(f"Probe phase: {phase}")
    print(f"Hemisphere: {hemisphere}")
    print(
        "Subjects before/after exclusion: "
        f"{subjects_before} / {subjects_after}"
    )
    if args.skip_subs:
        print(f"Excluded subjects: {', '.join(args.skip_subs)}")
    print(
        "QC high voxel loss: "
        + (", ".join(high_voxel_loss) if high_voxel_loss else "none")
    )
    print(
        "QC passive CV below chance: "
        + (
            ", ".join(passive_below_chance)
            if passive_below_chance
            else "none"
        )
    )
    print(f"Saved group summary: {output_csv}")

    primary = summary.loc[
        (summary["test"] == "one_sample")
        & (
            summary["metric"].isin(
                [
                    "cue_balanced_accuracy",
                    "cue_auc",
                    "cue_mean_signed_distance",
                    "passive_cv_balanced_accuracy",
                ]
            )
        )
    ]
    columns = [
        "cue_subset",
        "metric",
        "n",
        "mean",
        "null_value",
        "t",
        "p_two_sided",
        "p_greater_than_null",
    ]
    print("\nPrimary group summaries:")
    print(primary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
