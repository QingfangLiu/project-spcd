#!/usr/bin/env python3
"""Aggregate subject-level rank-based odor-identity ROI RSA results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")
METRICS = (
    "odor_identity_no_image_spearman_rho",
    "odor_identity_no_image_rank_biserial",
)


def _sanitize_label(label: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in label
    )


def _safe_wilcoxon(
    values: np.ndarray,
    alternative: str,
) -> tuple[float, float]:
    if values.size == 0 or np.all(values == 0):
        return np.nan, np.nan
    result = stats.wilcoxon(
        values,
        zero_method="wilcox",
        alternative=alternative,
    )
    statistic = float(getattr(result, "statistic", result[0]))
    p_value = float(getattr(result, "pvalue", result[1]))
    return statistic, p_value


def _sign_test(values: np.ndarray, alternative: str) -> float:
    positive = int((values > 0).sum())
    negative = int((values < 0).sum())
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
    analysis_set: str,
    test_label: str,
    metric: str,
    contrast: str,
) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = int(values.size)
    mean = float(np.mean(values)) if n else np.nan
    median = float(np.median(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 else np.nan
    df = n - 1

    if n > 1:
        t_result = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        t_stat = float(getattr(t_result, "statistic", t_result[0]))
        p_two = float(getattr(t_result, "pvalue", t_result[1]))
        p_greater = float(stats.t.sf(t_stat, df))
        p_less = float(stats.t.cdf(t_stat, df))
        cohen_dz = (
            float(mean / sd)
            if sd and np.isfinite(sd) and sd > 0
            else np.nan
        )
        wilcoxon_w, p_wilcoxon_two = _safe_wilcoxon(
            values,
            alternative="two-sided",
        )
        _, p_wilcoxon_greater = _safe_wilcoxon(
            values,
            alternative="greater",
        )
        _, p_wilcoxon_less = _safe_wilcoxon(
            values,
            alternative="less",
        )
    else:
        t_stat = p_two = p_greater = p_less = cohen_dz = np.nan
        wilcoxon_w = p_wilcoxon_two = np.nan
        p_wilcoxon_greater = p_wilcoxon_less = np.nan

    n_positive = int((values > 0).sum())
    n_negative = int((values < 0).sum())
    n_zero = int((values == 0).sum())

    return {
        "analysis_set": analysis_set,
        "test": test_label,
        "metric": metric,
        "contrast": contrast,
        "n": n,
        "mean": mean,
        "median": median,
        "sd": sd,
        "se": se,
        "t": t_stat,
        "df": df,
        "p_two_sided": p_two,
        "p_greater_than_zero": p_greater,
        "p_less_than_zero": p_less,
        "cohen_dz": cohen_dz,
        "wilcoxon_w": wilcoxon_w,
        "p_wilcoxon_two_sided": p_wilcoxon_two,
        "p_wilcoxon_greater_than_zero": p_wilcoxon_greater,
        "p_wilcoxon_less_than_zero": p_wilcoxon_less,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_zero": n_zero,
        "p_sign_two_sided": _sign_test(values, "two-sided"),
        "p_sign_greater_than_zero": _sign_test(values, "greater"),
        "p_sign_less_than_zero": _sign_test(values, "less"),
    }


def _summarize_subject_metrics(
    df: pd.DataFrame,
    analysis_set: str,
) -> pd.DataFrame:
    rows = []

    for subset in sorted(df["cue_subset"].dropna().unique()):
        subset_df = df.loc[df["cue_subset"] == subset]
        for metric in METRICS:
            rows.append(
                _one_sample_summary(
                    subset_df[metric].to_numpy(),
                    analysis_set=analysis_set,
                    test_label="one_sample",
                    metric=metric,
                    contrast=f"{subset}_gt_zero",
                )
            )

    conditioned = "conditioned_odor_only"
    preconditioned = "preconditioned_odor_only"
    for metric in METRICS:
        wide = df.pivot_table(
            index="sub",
            columns="cue_subset",
            values=metric,
            aggfunc="first",
        )
        if (
            conditioned not in wide.columns
            or preconditioned not in wide.columns
        ):
            continue
        difference = wide[conditioned] - wide[preconditioned]
        rows.append(
            _one_sample_summary(
                difference.to_numpy(),
                analysis_set=analysis_set,
                test_label="paired_difference",
                metric=metric,
                contrast="conditioned_minus_preconditioned",
            )
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize subject Spearman/rank-biserial ROI RSA with t, "
            "Wilcoxon, and sign tests."
        )
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Base data directory, used for default paths.",
    )
    parser.add_argument(
        "--roi_name",
        default="lateral_ofc",
        help="ROI label used in default filenames.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Subject results CSV from run_roi_rsa_spearman.py.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Group summary CSV path.",
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
        help="Primary-analysis subject exclusions.",
    )
    parser.add_argument(
        "--voxel_loss_flag_threshold",
        type=float,
        default=0.20,
        help=(
            "Report, but do not exclude, subjects at or above this fraction "
            "of dropped ROI voxels. Default: 0.20."
        ),
    )
    args = parser.parse_args()

    if not 0 <= args.voxel_loss_flag_threshold <= 1:
        raise ValueError("--voxel_loss_flag_threshold must be between 0 and 1")

    roi_name = _sanitize_label(args.roi_name)
    base_dir = (
        Path(args.base_dir).expanduser().resolve()
        if args.base_dir
        else None
    )

    if args.input_csv:
        input_csv = Path(args.input_csv).expanduser().resolve()
    elif base_dir is not None:
        input_csv = (
            base_dir
            / "rsa_roi"
            / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_spearman"
            / f"roi-{roi_name}_subject_spearman.csv"
        )
    else:
        raise ValueError("Provide --input_csv or --base_dir.")

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
    elif base_dir is not None:
        output_csv = (
            base_dir
            / "rsa_roi"
            / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_spearman"
            / f"roi-{roi_name}_spearman_group_summary.csv"
        )
    else:
        output_csv = input_csv.with_name(
            f"roi-{roi_name}_spearman_group_summary.csv"
        )

    df = pd.read_csv(input_csv)
    required = {
        "sub",
        "cue_subset",
        "n_runs",
        "n_voxels_raw",
        "n_voxels_dropped_nonfinite",
        *METRICS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV missing required columns: {sorted(missing)}"
        )
    if df.duplicated(["sub", "cue_subset"]).any():
        raise ValueError(
            "Input CSV has duplicate subject/cue-subset rows."
        )

    original_n = int(df["sub"].nunique())
    if args.skip_subs:
        df = df.loc[~df["sub"].isin(args.skip_subs)].copy()
    final_n = int(df["sub"].nunique())

    incomplete_subjects = sorted(
        df.loc[df["n_runs"] < 3, "sub"].unique()
    )
    voxel_loss_fraction = (
        df["n_voxels_dropped_nonfinite"] / df["n_voxels_raw"]
    )
    high_loss_subjects = sorted(
        df.loc[
            voxel_loss_fraction >= args.voxel_loss_flag_threshold,
            "sub",
        ].unique()
    )

    summary = _summarize_subject_metrics(
        df,
        analysis_set="primary",
    )
    summary.insert(0, "roi", roi_name)
    summary.insert(1, "input_csv", str(input_csv))
    summary.insert(2, "subjects_before_exclusion", original_n)
    summary.insert(3, "subjects_after_exclusion", final_n)
    summary.insert(4, "excluded_subjects", ",".join(args.skip_subs or []))
    summary.insert(
        5,
        "qc_incomplete_run_subjects",
        ",".join(incomplete_subjects),
    )
    summary.insert(
        6,
        "qc_voxel_loss_flag_threshold",
        args.voxel_loss_flag_threshold,
    )
    summary.insert(
        7,
        "qc_high_voxel_loss_subjects",
        ",".join(high_loss_subjects),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print(f"Read subject rank RSA results: {input_csv}")
    print(
        "Subjects before/after exclusion: "
        f"{original_n} / {final_n}"
    )
    if args.skip_subs:
        print(f"Excluded subjects: {', '.join(args.skip_subs)}")
    print("QC flags (reported only; no additional exclusion):")
    print(
        "  Fewer than three runs: "
        + (", ".join(incomplete_subjects) if incomplete_subjects else "none")
    )
    print(
        f"  Voxel loss >= {args.voxel_loss_flag_threshold:.0%}: "
        + (", ".join(high_loss_subjects) if high_loss_subjects else "none")
    )
    print(f"Saved group summary: {output_csv}")

    spearman_rows = summary.loc[
        summary["metric"] == "odor_identity_no_image_spearman_rho"
    ]
    if not spearman_rows.empty:
        print("\nSpearman rho group tests:")
        columns = [
            "test",
            "contrast",
            "n",
            "mean",
            "median",
            "t",
            "p_two_sided",
            "p_wilcoxon_two_sided",
            "p_sign_two_sided",
        ]
        print(spearman_rows[columns].to_string(index=False))


if __name__ == "__main__":
    main()
