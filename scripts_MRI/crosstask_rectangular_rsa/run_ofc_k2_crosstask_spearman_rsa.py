#!/usr/bin/env python3
"""Run rectangular cross-task Spearman RSA for the two Kahnt K=2 ROIs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROBE_RSA_DIR = (
    SCRIPT_DIR.parent / "rsa_imagewise_odor_identity_no_image_first_image"
)
if str(PROBE_RSA_DIR) not in sys.path:
    sys.path.insert(0, str(PROBE_RSA_DIR))

from run_ofc_cluster_map_k2 import (
    CLUSTERS,
    DEFAULT_CUE_SUBSETS,
    DEFAULT_SKIP_SUBS,
)


HEMISPHERES = ("bilateral", "left", "right")


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_command(command: list[str], dry_run: bool) -> None:
    print("+ " + shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _preflight_data(base_dir: Path) -> None:
    required = (
        base_dir / "GLM" / "odor_scan_trialwise_2mm",
        base_dir / "nii",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The SPCD data directory is missing required inputs:\n"
            f"{formatted}"
        )


def _scope_dir(
    output_root: Path,
    cluster_key: str,
    roi_name: str,
    hemisphere: str,
) -> Path:
    return (
        output_root
        / cluster_key
        / f"roi-{roi_name}_{hemisphere}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run, aggregate, and plot passive-odor to premeal-cue "
            "rectangular Spearman RSA in the Kahnt OFC K=2 ROIs."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help=(
            "SPCD data root containing odor_scan_trialwise_2mm betas "
            "and odor/premeal event files."
        ),
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=_default_project_root(),
        help="Project clone root. Default: infer from this script.",
    )
    parser.add_argument(
        "--repo_output_dir",
        type=Path,
        help=(
            "Output root. Default: <project_root>/rsa_roi/"
            "crosstask_passive_odor_to_premeal_cues_spearman."
        ),
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        help="Plot directory. Default: <repo_output_dir>/plots.",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        choices=sorted(CLUSTERS),
        default=sorted(CLUSTERS),
    )
    parser.add_argument(
        "--hemispheres",
        nargs="+",
        choices=HEMISPHERES,
        default=["bilateral"],
    )
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=DEFAULT_CUE_SUBSETS,
        default=list(DEFAULT_CUE_SUBSETS),
    )
    parser.add_argument(
        "--odor_runs",
        nargs="+",
        type=int,
        default=[1, 2],
    )
    parser.add_argument(
        "--sub14_original_events_dir",
        type=Path,
        help=(
            "Optional directory containing sub-14 original premeal event "
            "files. Default: derive from --base_dir."
        ),
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
    )
    parser.add_argument("--save_pair_diagnostics", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument(
        "--plot_formats",
        nargs="+",
        default=["pdf", "png"],
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without requiring external data or writing outputs.",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    output_root = (
        args.repo_output_dir.expanduser().resolve()
        if args.repo_output_dir is not None
        else (
            project_root
            / "rsa_roi"
            / "crosstask_passive_odor_to_premeal_cues_spearman"
        )
    )
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else output_root / "plots"
    )

    run_script = SCRIPT_DIR / "run_roi_crosstask_spearman_rsa.py"
    aggregate_script = (
        SCRIPT_DIR / "aggregate_roi_crosstask_spearman_rsa.py"
    )
    plot_script = SCRIPT_DIR / "plot_ofc_k2_crosstask_spearman_rsa.py"
    _require_file(run_script, "Subject-level cross-task RSA script")
    _require_file(aggregate_script, "Cross-task RSA aggregation script")
    if not args.skip_plots:
        _require_file(plot_script, "Cross-task RSA plotting script")
    if not args.dry_run:
        _preflight_data(base_dir)

    print("=" * 80)
    print("Kahnt OFC K=2 passive-odor -> premeal-cue Spearman RSA")
    print("=" * 80)
    print(f"Project root:       {project_root}")
    print(f"SPCD data root:     {base_dir}")
    print(f"Repository output:  {output_root}")
    print(f"Clusters:           {', '.join(args.clusters)}")
    print(f"Hemispheres:        {', '.join(args.hemispheres)}")
    print(f"Cue subsets:        {', '.join(args.cue_subsets)}")
    print("Normalization:      within-task, within-run voxel centering")
    print("Pair scope:         passive odor x premeal cue")
    print(f"Dry run:            {args.dry_run}")
    print("=" * 80)

    subject_csvs: dict[str, list[Path]] = {
        hemisphere: [] for hemisphere in args.hemispheres
    }
    diagnostics_root = (
        project_root
        / "rsa_roi"
        / "probe_imagewise_odor_identity_no_image_first_image"
        / "premeal_ofc_k2_spearman"
    )
    for cluster_label in args.clusters:
        spec = CLUSTERS[cluster_label]
        diagnostics_dir = diagnostics_root / spec.key / "diagnostics"
        if not args.dry_run and not diagnostics_dir.is_dir():
            raise FileNotFoundError(
                "First-image cue diagnostic directory not found: "
                f"{diagnostics_dir}"
            )
        cluster_output_root = output_root / spec.key
        run_command = [
            sys.executable,
            str(run_script),
            "--base_dir",
            str(base_dir),
            "--diagnostics_dir",
            str(diagnostics_dir),
            "--roi_name",
            spec.roi_name,
            "--hemispheres",
            *args.hemispheres,
            "--cue_subsets",
            *args.cue_subsets,
            "--odor_runs",
            *map(str, args.odor_runs),
            "--output_root",
            str(cluster_output_root),
        ]
        if args.subjects:
            run_command.extend(["--subjects", *args.subjects])
        if args.sub14_original_events_dir is not None:
            run_command.extend(
                [
                    "--sub14_original_events_dir",
                    str(args.sub14_original_events_dir.expanduser().resolve()),
                ]
            )
        if args.save_pair_diagnostics:
            run_command.append("--save_pair_diagnostics")
        if args.strict:
            run_command.append("--strict")
        _run_command(run_command, args.dry_run)

        for hemisphere in args.hemispheres:
            scope_dir = _scope_dir(
                output_root,
                spec.key,
                spec.roi_name,
                hemisphere,
            )
            subject_csv = scope_dir / "subject_crosstask_spearman_rsa.csv"
            group_csv = scope_dir / "group_crosstask_spearman_rsa_summary.csv"
            subject_csvs[hemisphere].append(subject_csv)
            aggregate_command = [
                sys.executable,
                str(aggregate_script),
                "--input_csv",
                str(subject_csv),
                "--output_csv",
                str(group_csv),
                "--skip_subs",
                *args.skip_subs,
            ]
            _run_command(aggregate_command, args.dry_run)

    if not args.skip_plots:
        for hemisphere, csvs in subject_csvs.items():
            plot_command = [
                sys.executable,
                str(plot_script),
                "--input_csvs",
                *map(str, csvs),
                "--output_prefix",
                str(
                    plot_dir
                    / f"OFC_K2_{hemisphere}_crosstask_spearman_rsa"
                ),
                "--skip_subs",
                *args.skip_subs,
                "--formats",
                *args.plot_formats,
            ]
            _run_command(plot_command, args.dry_run)

    if args.dry_run:
        print("\nDry run complete; no analyses, CSVs, or plots were created.")


if __name__ == "__main__":
    main()
