#!/usr/bin/env python3
"""Compile subject pleasantness summaries into one wide CSV for RSA.

This script reads:
  beh_processed_subs/sub-*/pleasantness_summary.csv

and writes one row per subject with columns:
  sub, pre_odor1, pre_odor2, pre_clean, post_odor1, post_odor2, post_clean

Processed behavior coding is assumed to be:
  odor_num 0 = clean
  odor_num 1 = non-devalued odor  -> odor1 in the MRI/RSA code
  odor_num 2 = devalued odor      -> odor2 in the MRI/RSA code
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    default_input = root / "beh_processed_subs"
    default_output = root / "beh_processed_subs" / "compiled_pleasantness_ratings.csv"

    parser = argparse.ArgumentParser(
        description="Compile subject pleasantness summaries into one wide CSV."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=default_input,
        help="Directory containing sub-*/pleasantness_summary.csv files",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=default_output,
        help="Output CSV path",
    )
    parser.add_argument(
        "--include_missing",
        action="store_true",
        help="Include subjects with missing summaries as rows with blank values",
    )
    return parser.parse_args()


def load_subject_summary(summary_file: Path) -> dict[str, float]:
    df = pd.read_csv(summary_file).copy()

    required_cols = {"odor_num", "pre_mean", "post_mean"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{summary_file} is missing required columns: {sorted(missing_cols)}"
        )

    value_map_pre = dict(zip(df["odor_num"].astype(int), df["pre_mean"].astype(float)))
    value_map_post = dict(zip(df["odor_num"].astype(int), df["post_mean"].astype(float)))

    missing_codes = {0, 1, 2} - set(value_map_pre.keys())
    if missing_codes:
        raise ValueError(
            f"{summary_file} is missing required odor_num values: {sorted(missing_codes)}"
        )

    return {
        "pre_odor1": float(value_map_pre[1]),
        "pre_odor2": float(value_map_pre[2]),
        "pre_clean": float(value_map_pre[0]),
        "post_odor1": float(value_map_post[1]),
        "post_odor2": float(value_map_post[2]),
        "post_clean": float(value_map_post[0]),
    }


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    rows: list[dict[str, object]] = []
    missing_subjects: list[str] = []

    for sub_dir in sorted(args.input_dir.glob("sub-*")):
        if not sub_dir.is_dir():
            continue

        subject = sub_dir.name
        summary_file = sub_dir / "pleasantness_summary.csv"

        if not summary_file.exists():
            missing_subjects.append(subject)
            if args.include_missing:
                rows.append(
                    {
                        "sub": subject,
                        "pre_odor1": pd.NA,
                        "pre_odor2": pd.NA,
                        "pre_clean": pd.NA,
                        "post_odor1": pd.NA,
                        "post_odor2": pd.NA,
                        "post_clean": pd.NA,
                    }
                )
            continue

        row = {"sub": subject}
        row.update(load_subject_summary(summary_file))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values("sub").reset_index(drop=True)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_file, index=False)

    print(f"Saved compiled pleasantness ratings to: {args.output_file}")
    print(f"Included subjects: {len(out_df)}")

    if missing_subjects:
        print("Subjects missing pleasantness_summary.csv:")
        for subject in missing_subjects:
            print(f" - {subject}")
    else:
        print("No subjects were missing pleasantness_summary.csv.")


if __name__ == "__main__":
    main()
