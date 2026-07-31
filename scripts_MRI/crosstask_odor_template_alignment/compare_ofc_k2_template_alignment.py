#!/usr/bin/env python3
"""Direct paired comparisons across OFC K=2 ROIs, cues, and hemispheres."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aggregate_roi_odor_template_alignment import (
    DEFAULT_SKIP_SUBS,
    _one_sample_summary,
)


ROIS = (
    "OFC_K2_cluster1_medial_posterior_lateral",
    "OFC_K2_cluster2_central_anterior_lateral",
)
HEMISPHERES = ("bilateral", "left", "right")
CUE_SUBSETS = (
    "preconditioned_odor_only",
    "conditioned_odor_only",
)
METRICS = (
    "matching_correlation_advantage",
    "fisher_z_matching_advantage",
)


def _load_results(output_root: Path, skip_subs: list[str]) -> pd.DataFrame:
    frames = []
    for roi in ROIS:
        for hemisphere in HEMISPHERES:
            path = (
                output_root
                / f"roi-{roi}_{hemisphere}"
                / "subject_template_alignment.csv"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Subject results not found: {path}")
            frame = pd.read_csv(path)
            frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.loc[~data["sub"].isin(skip_subs)].copy()
    required = {"sub", "roi", "hemisphere", "cue_subset", *METRICS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Subject results missing columns: {sorted(missing)}")
    if data.duplicated(["sub", "roi", "hemisphere", "cue_subset"]).any():
        raise ValueError("Subject results contain duplicate analysis rows.")
    return data


def _paired_row(
    data: pd.DataFrame,
    *,
    metric: str,
    contrast: str,
    cue_subset: str,
    left_query: str,
    right_query: str,
    left_label: str,
    right_label: str,
) -> dict:
    left = data.query(left_query).set_index("sub")[metric]
    right = data.query(right_query).set_index("sub")[metric]
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1)
    paired = paired.dropna()
    summary = _one_sample_summary(
        paired["left"] - paired["right"],
        metric=metric,
        null_value=0.0,
        cue_subset=cue_subset,
        test="paired_difference",
        contrast=contrast,
        directional=False,
    )
    summary["left_term"] = left_label
    summary["right_term"] = right_label
    return summary


def _comparisons(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for roi in ROIS:
            for hemisphere in HEMISPHERES:
                base = f"roi == '{roi}' and hemisphere == '{hemisphere}'"
                rows.append(
                    _paired_row(
                        data,
                        metric=metric,
                        contrast=(
                            f"conditioned_minus_preconditioned__{roi}__"
                            f"{hemisphere}"
                        ),
                        cue_subset="conditioned_minus_preconditioned",
                        left_query=(
                            base
                            + " and cue_subset == 'conditioned_odor_only'"
                        ),
                        right_query=(
                            base
                            + " and cue_subset == 'preconditioned_odor_only'"
                        ),
                        left_label="conditioned_odor_only",
                        right_label="preconditioned_odor_only",
                    )
                )

        for hemisphere in HEMISPHERES:
            for cue_subset in CUE_SUBSETS:
                rows.append(
                    _paired_row(
                        data,
                        metric=metric,
                        contrast=(
                            "cluster2_minus_cluster1__"
                            f"{hemisphere}__{cue_subset}"
                        ),
                        cue_subset=cue_subset,
                        left_query=(
                            f"roi == '{ROIS[1]}' and hemisphere == "
                            f"'{hemisphere}' and cue_subset == '{cue_subset}'"
                        ),
                        right_query=(
                            f"roi == '{ROIS[0]}' and hemisphere == "
                            f"'{hemisphere}' and cue_subset == '{cue_subset}'"
                        ),
                        left_label=ROIS[1],
                        right_label=ROIS[0],
                    )
                )

        for roi in ROIS:
            for cue_subset in CUE_SUBSETS:
                rows.append(
                    _paired_row(
                        data,
                        metric=metric,
                        contrast=f"left_minus_right__{roi}__{cue_subset}",
                        cue_subset=cue_subset,
                        left_query=(
                            f"roi == '{roi}' and hemisphere == 'left' and "
                            f"cue_subset == '{cue_subset}'"
                        ),
                        right_query=(
                            f"roi == '{roi}' and hemisphere == 'right' and "
                            f"cue_subset == '{cue_subset}'"
                        ),
                        left_label="left",
                        right_label="right",
                    )
                )

                cell = data.loc[
                    (data["roi"] == roi)
                    & (data["cue_subset"] == cue_subset),
                    ["sub", "hemisphere", metric],
                ].pivot(index="sub", columns="hemisphere", values=metric)
                difference = cell["bilateral"] - (
                    cell["left"] + cell["right"]
                ) / 2
                summary = _one_sample_summary(
                    difference.dropna(),
                    metric=metric,
                    null_value=0.0,
                    cue_subset=cue_subset,
                    test="paired_difference",
                    contrast=(
                        f"bilateral_minus_mean_left_right__{roi}__"
                        f"{cue_subset}"
                    ),
                    directional=False,
                )
                summary["left_term"] = "bilateral"
                summary["right_term"] = "mean(left,right)"
                rows.append(summary)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OFC K=2 template alignment within subjects."
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument(
        "--skip_subs", nargs="*", default=list(DEFAULT_SKIP_SUBS)
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv is not None
        else output_root / "ofc_k2_paired_comparisons.csv"
    )
    data = _load_results(output_root, args.skip_subs)
    comparisons = _comparisons(data)
    comparisons.insert(0, "excluded_subjects", ",".join(args.skip_subs))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(output_csv, index=False)
    print(f"Saved paired comparisons: {output_csv}")


if __name__ == "__main__":
    main()
