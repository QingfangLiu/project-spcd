#!/usr/bin/env python3
"""Create a paired ROI RSA plot for a prepared Kahnt OFC K=2 cluster.

This command uses shared ROI plotting helpers. Input, summary, output, and
title arguments are explicit so each K=2 cluster has its own plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import roi_rsa_plotting as established_plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired subject-level no-image-first-image ROI RSA betas for "
            "one Kahnt OFC K=2 cluster."
        )
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        required=True,
        help="Subject-level ROI RSA beta CSV.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        required=True,
        help="Group summary CSV from aggregate_roi_rsa.py.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        required=True,
        help="Output path without an extension.",
    )
    parser.add_argument(
        "--title",
        required=True,
        help="Plot title describing the selected K=2 cluster.",
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(established_plot.DEFAULT_SKIP_SUBS),
        help="Subjects to exclude. Default: sub-19 sub-33.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats. Default: pdf png.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster outputs. Default: 300.",
    )
    args = parser.parse_args()

    input_csv = args.input_csv.expanduser().resolve()
    summary_csv = args.summary_csv.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve()
    if output_prefix.suffix:
        raise ValueError(
            "--output_prefix must not include a filename extension: "
            f"{output_prefix}"
        )

    print(f"Input CSV:     {input_csv}")
    print(f"Summary CSV:   {summary_csv}")
    print(f"Output prefix: {output_prefix}")
    print(f"Title:         {args.title}")
    print(
        "Skip subjects: "
        + (", ".join(args.skip_subs) if args.skip_subs else "none")
    )

    data = established_plot._load_subject_data(
        input_csv,
        args.skip_subs,
    )
    group_summary = established_plot._load_group_summary(summary_csv)
    established_plot._plot_paired_bar(
        data=data,
        group_summary=group_summary,
        output_prefix=output_prefix,
        formats=args.formats,
        dpi=args.dpi,
        title=args.title,
    )


if __name__ == "__main__":
    main()
