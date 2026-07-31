#!/usr/bin/env python3
"""Run, aggregate, compare, and plot OFC K=2 template alignment."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoiSpec:
    key: str
    name: str


ROIS = (
    RoiSpec(
        "cluster1_medial_posterior_lateral",
        "OFC_K2_cluster1_medial_posterior_lateral",
    ),
    RoiSpec(
        "cluster2_central_anterior_lateral",
        "OFC_K2_cluster2_central_anterior_lateral",
    ),
)
HEMISPHERES = ("bilateral", "left", "right")
CUE_SUBSETS = (
    "preconditioned_odor_only",
    "conditioned_odor_only",
)
DEFAULT_SKIP_SUBS = ("sub-19", "sub-33")


def _run(command: list[str], dry_run: bool) -> None:
    print("+ " + shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    project_root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Run the complete OFC K=2 odor-template alignment workflow."
    )
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument(
        "--project_root", type=Path, default=project_root_default
    )
    parser.add_argument(
        "--hemispheres",
        nargs="+",
        choices=HEMISPHERES,
        default=list(HEMISPHERES),
    )
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=CUE_SUBSETS,
        default=list(CUE_SUBSETS),
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--odor_runs", nargs="+", type=int, default=[1, 2])
    parser.add_argument(
        "--skip_subs", nargs="*", default=list(DEFAULT_SKIP_SUBS)
    )
    parser.add_argument("--sub14_original_events_dir", type=Path)
    parser.add_argument("--output_root", type=Path)
    parser.add_argument("--plot_dir", type=Path)
    parser.add_argument("--skip_trial_outputs", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--plot_formats", nargs="+", default=["pdf", "png"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else project_root
        / "roi_template_alignment"
        / "passive_odor_to_premeal_cues"
    )
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else project_root / "plots"
    )
    analysis_dir = Path(__file__).resolve().parent
    subject_script = analysis_dir / "run_roi_odor_template_alignment.py"
    aggregate_script = analysis_dir / "aggregate_roi_odor_template_alignment.py"
    compare_script = analysis_dir / "compare_ofc_k2_template_alignment.py"
    plot_script = analysis_dir / "plot_roi_odor_template_alignment.py"

    print("=" * 80)
    print("OFC K=2 passive odor -> premeal cue template alignment")
    print("=" * 80)
    print(f"Project root:   {project_root}")
    print(f"SPCD data:      {base_dir}")
    print(f"Output root:    {output_root}")
    print(f"Hemispheres:    {', '.join(args.hemispheres)}")
    print(f"Cue subsets:    {', '.join(args.cue_subsets)} (separate)")
    print(f"Dry run:        {args.dry_run}")
    print("=" * 80)

    for roi in ROIS:
        diagnostics = (
            project_root
            / "rsa_roi"
            / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_ofc_k2_spearman"
            / roi.key
            / "diagnostics"
        )
        command = [
            sys.executable,
            str(subject_script),
            "--base_dir",
            str(base_dir),
            "--diagnostics_dir",
            str(diagnostics),
            "--roi_name",
            roi.name,
            "--hemispheres",
            *args.hemispheres,
            "--cue_subsets",
            *args.cue_subsets,
            "--odor_runs",
            *map(str, args.odor_runs),
            "--output_root",
            str(output_root),
        ]
        if args.subjects:
            command.extend(["--subjects", *args.subjects])
        if args.sub14_original_events_dir:
            command.extend(
                [
                    "--sub14_original_events_dir",
                    str(args.sub14_original_events_dir),
                ]
            )
        if args.skip_trial_outputs:
            command.append("--skip_trial_outputs")
        if args.strict:
            command.append("--strict")
        _run(command, args.dry_run)

        for hemisphere in args.hemispheres:
            scope_dir = output_root / f"roi-{roi.name}_{hemisphere}"
            _run(
                [
                    sys.executable,
                    str(aggregate_script),
                    "--input_csv",
                    str(scope_dir / "subject_template_alignment.csv"),
                    "--skip_subs",
                    *args.skip_subs,
                ],
                args.dry_run,
            )

    all_scopes_requested = set(args.hemispheres) == set(HEMISPHERES)
    all_cues_requested = set(args.cue_subsets) == set(CUE_SUBSETS)
    if all_scopes_requested and all_cues_requested:
        _run(
            [
                sys.executable,
                str(compare_script),
                "--output_root",
                str(output_root),
                "--skip_subs",
                *args.skip_subs,
            ],
            args.dry_run,
        )
    else:
        print(
            "Skipping omnibus paired comparisons because all hemispheres "
            "and both cue subsets were not requested."
        )

    if not args.skip_plots and all_cues_requested:
        for hemisphere in args.hemispheres:
            input_csvs = [
                output_root
                / f"roi-{roi.name}_{hemisphere}"
                / "subject_template_alignment.csv"
                for roi in ROIS
            ]
            _run(
                [
                    sys.executable,
                    str(plot_script),
                    "--input_csv",
                    *map(str, input_csvs),
                    "--hemisphere",
                    hemisphere,
                    "--output_prefix",
                    str(
                        plot_dir
                        / f"OFC_K2_{hemisphere}_odor_template_alignment"
                    ),
                    "--skip_subs",
                    *args.skip_subs,
                    "--formats",
                    *args.plot_formats,
                ],
                args.dry_run,
            )
    elif not args.skip_plots:
        print("Skipping plots because both cue subsets were not requested.")

    print("Dry run complete." if args.dry_run else "Workflow complete.")


if __name__ == "__main__":
    main()
