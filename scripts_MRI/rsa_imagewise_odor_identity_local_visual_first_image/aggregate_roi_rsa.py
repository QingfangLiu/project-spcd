#!/usr/bin/env python3
"""Aggregate ROI RSA subject betas into group-level summary tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


MODEL_NAMES = (
    "odor_identity",
    "visual_image",
)

DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")


def _sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def _default_input_csv(base_dir: Path, roi_name: str) -> Path:
    return (
        base_dir
        / "rsa_roi"
        / "probe_imagewise_odor_identity_local_visual_first_image"
        / "premeal"
        / f"roi-{roi_name}_subject_betas.csv"
    )


def _default_output_csv(base_dir: Path, roi_name: str) -> Path:
    return (
        base_dir
        / "rsa_roi"
        / "probe_imagewise_odor_identity_local_visual_first_image"
        / "premeal"
        / f"roi-{roi_name}_group_summary.csv"
    )


def _one_sample_summary(values, test_label: str, model: str, contrast: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = int(values.size)
    mean = float(np.mean(values)) if n else np.nan
    sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
    se = float(sd / np.sqrt(n)) if n > 1 else np.nan
    df = n - 1

    if n > 1:
        t_stat, p_two = stats.ttest_1samp(values, 0.0, nan_policy="omit")
        t_stat = float(t_stat)
        p_two = float(p_two)
        p_greater = float(stats.t.sf(t_stat, df))
        p_less = float(stats.t.cdf(t_stat, df))
        cohen_dz = float(mean / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan
    else:
        t_stat = p_two = p_greater = p_less = cohen_dz = np.nan

    return {
        "test": test_label,
        "model": model,
        "contrast": contrast,
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": t_stat,
        "df": df,
        "p_two_sided": p_two,
        "p_greater_than_zero": p_greater,
        "p_less_than_zero": p_less,
        "cohen_dz": cohen_dz,
    }


def _summarize_subject_betas(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for subset in sorted(df["cue_subset"].dropna().unique()):
        subset_df = df.loc[df["cue_subset"] == subset]
        for model in MODEL_NAMES:
            col = f"{model}_beta"
            if col not in subset_df.columns:
                continue
            rows.append(
                _one_sample_summary(
                    subset_df[col].to_numpy(),
                    test_label="one_sample",
                    model=model,
                    contrast=f"{subset}_gt_zero",
                )
            )

    cond = "conditioned_odor_only"
    pre = "preconditioned_odor_only"
    for model in MODEL_NAMES:
        col = f"{model}_beta"
        if col not in df.columns:
            continue
        wide = df.pivot_table(index="sub", columns="cue_subset", values=col, aggfunc="first")
        if cond not in wide.columns or pre not in wide.columns:
            continue
        diff = wide[cond] - wide[pre]
        rows.append(
            _one_sample_summary(
                diff.to_numpy(),
                test_label="paired_difference",
                model=model,
                contrast="conditioned_minus_preconditioned",
            )
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ROI RSA subject betas with one-sample and paired tests."
    )
    parser.add_argument("--base_dir", default=None, help="Base data directory, used for default input/output paths")
    parser.add_argument("--roi_name", default="lateral_ofc", help="ROI label used in default filenames")
    parser.add_argument("--input_csv", default=None, help="Subject betas CSV from run_roi_rsa.py")
    parser.add_argument("--output_csv", default=None, help="Group summary CSV path")
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
        help="Subjects to exclude from group summaries",
    )
    args = parser.parse_args()

    roi_name = _sanitize_label(args.roi_name)
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else None

    if args.input_csv:
        input_csv = Path(args.input_csv).expanduser().resolve()
    elif base_dir is not None:
        input_csv = _default_input_csv(base_dir, roi_name)
    else:
        raise ValueError("Provide --input_csv or --base_dir.")

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
    elif base_dir is not None:
        output_csv = _default_output_csv(base_dir, roi_name)
    else:
        output_csv = input_csv.with_name(f"roi-{roi_name}_group_summary.csv")

    df = pd.read_csv(input_csv)
    required_cols = {"sub", "cue_subset"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    original_n = df["sub"].nunique()
    if args.skip_subs:
        df = df.loc[~df["sub"].isin(args.skip_subs)].copy()
    final_n = df["sub"].nunique()

    summary = _summarize_subject_betas(df)
    summary.insert(0, "roi", roi_name)
    summary.insert(1, "input_csv", str(input_csv))
    summary.insert(2, "subjects_before_exclusion", original_n)
    summary.insert(3, "subjects_after_exclusion", final_n)
    summary.insert(4, "excluded_subjects", ",".join(args.skip_subs or []))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)

    print(f"Read subject betas: {input_csv}")
    print(f"Subjects before/after exclusion: {original_n} / {final_n}")
    if args.skip_subs:
        print(f"Excluded subjects: {', '.join(args.skip_subs)}")
    print(f"Saved group summary: {output_csv}")

    odor_rows = summary.loc[summary["model"] == "odor_identity"]
    if not odor_rows.empty:
        print("\nOdor-identity tests:")
        cols = ["test", "contrast", "n", "mean", "t", "df", "p_two_sided", "cohen_dz"]
        print(odor_rows[cols].to_string(index=False))


if __name__ == "__main__":
    main()
