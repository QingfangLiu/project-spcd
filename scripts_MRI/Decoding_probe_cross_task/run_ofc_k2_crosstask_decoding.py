#!/usr/bin/env python3
"""Run, summarize, and plot passive-to-cue decoding in both bilateral OFC ROIs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


CLUSTERS = (
    "cluster1_medial_posterior_lateral",
    "cluster2_central_anterior_lateral",
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("premeal", "postmeal"), default="premeal")
    parser.add_argument(
        "--diagnostics_root", type=Path,
        default=(
            project_root / "rsa_roi" / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_ofc_k2_spearman"
        ),
        help="Premeal RSA output root with one diagnostics/ directory per cluster.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--plot_dir", type=Path, default=project_root / "plots")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--skip_subs", nargs="*", default=["sub-19", "sub-33"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    diagnostics_root = args.diagnostics_root.expanduser().resolve()
    output_name = "probe_crosstask_odor2preconditioned"
    if args.phase == "postmeal":
        output_name += "_postmeal"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "roi_decoding" / output_name
    )
    if not args.dry_run:
        required = [base_dir / "GLM" / "odor_scan_trialwise_2mm"]
        if args.phase == "postmeal":
            required.append(base_dir / "GLM" / "postmeal_trialwise_2mm")
        required.extend(diagnostics_root / cluster / "diagnostics" for cluster in CLUSTERS)
        missing = [path for path in required if not path.is_dir()]
        if missing:
            raise FileNotFoundError("Missing input directories:\n" + "\n".join(map(str, missing)))

    def run(command: list[str]) -> None:
        print("+ " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)

    result_paths = []
    for cluster in CLUSTERS:
        roi_name = f"OFC_K2_{cluster}"
        result_path = output_dir / f"roi-{roi_name}_subject_results.csv"
        result_paths.append(result_path)
        command = [
            sys.executable, str(script_dir / "run_roi_crosstask_odor2preconditioned.py"),
            "--base_dir", str(base_dir), "--phase", args.phase,
            "--diagnostics_dir", str(diagnostics_root / cluster / "diagnostics"),
            "--roi_name", roi_name,
            "--cue_subsets", "conditioned_odor_only", "preconditioned_odor_only",
            "--output_csv", str(result_path), "--save_predictions",
        ]
        if args.subjects:
            command.extend(["--subjects", *args.subjects])
        if args.strict:
            command.append("--strict")
        run(command)
        run([
            sys.executable, str(script_dir / "aggregate_roi_crosstask_odor2preconditioned.py"),
            "--input_csv", str(result_path), "--skip_subs", *args.skip_subs,
        ])

    run([
        sys.executable, str(script_dir / "plot_roi_crosstask_decoding_accuracy.py"),
        "--input_csv", *map(str, result_paths),
        "--output_prefix", str(
            args.plot_dir.expanduser().resolve()
            / f"OFC_K2_two_roi_crosstask_decoding_accuracy_{args.phase}_1x4"
        ),
        "--skip_subs", *args.skip_subs,
        "--title", f"Cross-task odor identity decoding in OFC ({args.phase})",
    ])


if __name__ == "__main__":
    main()
