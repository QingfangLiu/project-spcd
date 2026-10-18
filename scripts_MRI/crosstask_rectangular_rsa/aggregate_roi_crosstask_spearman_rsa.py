#!/usr/bin/env python3
"""Aggregate subject-level rectangular cross-task Spearman ROI RSA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RSA_DIR = (
    SCRIPT_DIR.parent / "rsa_imagewise_odor_identity_no_image_first_image"
)
if str(RSA_DIR) not in sys.path:
    sys.path.insert(0, str(RSA_DIR))

import aggregate_roi_rsa_spearman as probe_aggregate


DEFAULT_SKIP_SUBS = probe_aggregate.DEFAULT_SKIP_SUBS
METRICS = (
    "cross_task_odor_identity_spearman_rho",
    "cross_task_odor_identity_rank_biserial",
)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cue_subset in sorted(df["cue_subset"].dropna().unique()):
        subset = df.loc[df["cue_subset"] == cue_subset]
        for metric in METRICS:
            rows.append(
                probe_aggregate._one_sample_summary(
                    subset[metric].to_numpy(),
                    analysis_set="primary",
                    test_label="one_sample",
                    metric=metric,
                    contrast=f"{cue_subset}_gt_zero",
                )
                | {"cue_subset": cue_subset}
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
        if not {conditioned, preconditioned}.issubset(wide.columns):
            continue
        difference = (wide[preconditioned] - wide[conditioned]).dropna()
        rows.append(
            probe_aggregate._one_sample_summary(
                difference.to_numpy(),
                analysis_set="primary",
                test_label="paired_difference",
                metric=metric,
                contrast="preconditioned_minus_conditioned",
            )
            | {"cue_subset": "preconditioned_minus_conditioned"}
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test rectangular cross-task Spearman ROI RSA across subjects."
        )
    )
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
    )
    parser.add_argument(
        "--voxel_loss_flag_threshold",
        type=float,
        default=0.20,
    )
    args = parser.parse_args()

    if not 0 <= args.voxel_loss_flag_threshold <= 1:
        raise ValueError(
            "--voxel_loss_flag_threshold must be between 0 and 1."
        )
    input_csv = args.input_csv.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else input_csv.with_name("group_crosstask_spearman_rsa_summary.csv")
    )
    df = pd.read_csv(input_csv)
    required = {
        "sub",
        "roi",
        "hemisphere",
        "cue_subset",
        "n_voxels_raw",
        "n_voxels_dropped_nonfinite",
        "n_passive_runs",
        "n_cue_runs",
        *METRICS,
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
        df.loc[
            loss >= args.voxel_loss_flag_threshold,
            "sub",
        ].unique()
    )
    incomplete_passive = sorted(
        df.loc[df["n_passive_runs"] < 2, "sub"].unique()
    )
    incomplete_cue = sorted(df.loc[df["n_cue_runs"] < 3, "sub"].unique())

    summary = _summarize(df)
    summary.insert(0, "roi", rois[0])
    summary.insert(1, "hemisphere", hemispheres[0])
    summary.insert(2, "input_csv", str(input_csv))
    summary.insert(3, "subjects_before_exclusion", subjects_before)
    summary.insert(4, "subjects_after_exclusion", subjects_after)
    summary.insert(5, "excluded_subjects", ",".join(args.skip_subs))
    summary.insert(6, "qc_high_voxel_loss_subjects", ",".join(high_loss))
    summary.insert(
        7,
        "qc_incomplete_passive_run_subjects",
        ",".join(incomplete_passive),
    )
    summary.insert(
        8,
        "qc_incomplete_cue_run_subjects",
        ",".join(incomplete_cue),
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"Saved group summary: {output_csv}")


if __name__ == "__main__":
    main()
