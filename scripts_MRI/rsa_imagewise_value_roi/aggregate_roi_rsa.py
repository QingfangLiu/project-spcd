#!/usr/bin/env python3
"""Summarize odor-value coefficients using the same participants in every ROI."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsa_imagewise_odor_identity.aggregate_roi_rsa import _one_sample_summary


ROI_NAMES = ("Bartra_vmPFC", "OFC_K2_cluster1_medial_posterior_lateral",
             "OFC_K2_cluster2_central_anterior_lateral")
DEFAULT_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")


def summarize_subject_betas(frame: pd.DataFrame) -> pd.DataFrame:
    """Require a balanced three-ROI table and test value betas against zero."""
    required = {"sub", "roi", "odor_value_beta"}
    if not required.issubset(frame):
        raise ValueError(f"Missing columns: {sorted(required - set(frame))}")
    if frame.empty or frame.duplicated(["sub", "roi"]).any():
        raise ValueError("Expected one nonempty row per participant and ROI.")
    if frame[["sub", "roi"]].isna().any().any():
        raise ValueError("Participant and ROI labels must be present.")
    if set(frame["roi"]) != set(ROI_NAMES):
        raise ValueError(f"Expected exactly these ROIs: {ROI_NAMES}")
    wide = frame.pivot(index="sub", columns="roi", values="odor_value_beta")
    if not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise ValueError("Every participant must have a finite value beta in all three ROIs.")
    if len(wide) < 2:
        raise ValueError("Group inference requires at least two participants.")

    rows = []
    for roi in ROI_NAMES:
        row = _one_sample_summary(wide[roi], "one_sample", "odor_value", "all_gt_zero")
        row.update(roi=roi, included_subjects=",".join(wide.index), session="both",
                   cue_subset="all")
        half_ci = stats.t.ppf(0.975, row["df"]) * row["se"]
        row.update(ci95_low=row["mean"] - half_ci, ci95_high=row["mean"] + half_ci)
        rows.append(row)
    result = pd.DataFrame(rows)
    # The family comprises the three primary value tests, one per ROI.
    for name in ("p_greater_than_zero", "p_two_sided"):
        p = result[name].to_numpy()
        if not np.isfinite(p).all():
            raise ValueError("Undefined group t test (check zero between-participant variance).")
        order = np.argsort(p)
        adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(3, 0, -1)))
        restored = np.empty(3)
        restored[order] = adjusted
        result[f"{name}_holm_3_rois"] = restored
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--skip_subs", nargs="*", default=list(DEFAULT_SKIP_SUBS))
    args = parser.parse_args()
    data = pd.read_csv(args.input_csv)
    data = data.loc[~data["sub"].isin(args.skip_subs)].copy()
    result = summarize_subject_betas(data)
    result["excluded_subjects"] = ",".join(args.skip_subs)
    destination = args.output_csv or args.input_csv.with_name("group_summary.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
