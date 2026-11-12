#!/usr/bin/env python3
"""Run rank-based odor-identity ROI RSA for Kahnt OFC K=2 ROIs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    from run_ofc_cluster_map_k2 import (
        CLUSTERS,
        DEFAULT_CUE_SUBSETS,
        DEFAULT_SKIP_SUBS,
    )
else:
    from .run_ofc_cluster_map_k2 import (
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


def _preflight_base_dir(base_dir: Path) -> None:
    required = (
        base_dir / "masks" / "gm_0.1.nii",
        base_dir / "GLM" / "premeal_trialwise_2mm",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The SPCD data directory is missing required inputs:\n"
            f"{formatted}"
        )


def _hemisphere_mask_filename(filename: str, hemisphere: str) -> str:
    if hemisphere == "bilateral":
        return filename
    suffix = "_gm0p1.nii.gz"
    if not filename.endswith(suffix):
        raise ValueError(
            f"Expected mask filename ending in {suffix!r}; got {filename!r}"
        )
    return f"{filename[:-len(suffix)]}_{hemisphere}{suffix}"


def _analysis_scope(spec, hemisphere: str) -> dict[str, str]:
    if hemisphere not in HEMISPHERES:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")
    if hemisphere == "bilateral":
        return {
            "mask_filename": spec.mask_filename,
            "roi_name": spec.roi_name,
            "output_key": spec.key,
            "plot_stem": (
                f"OFC_K2_{spec.key}_no_image_first_image_"
                "spearman_roi_rsa_paired"
            ),
            "plot_title": f"{spec.plot_title} — Spearman RSA",
        }
    return {
        "mask_filename": _hemisphere_mask_filename(
            spec.mask_filename, hemisphere
        ),
        "roi_name": f"{spec.roi_name}_{hemisphere}",
        "output_key": f"{spec.key}_{hemisphere}",
        "plot_stem": (
            f"OFC_K2_{spec.key}_{hemisphere}_no_image_first_image_"
            "spearman_roi_rsa_paired"
        ),
        "plot_title": (
            f"{spec.plot_title}, {hemisphere} hemisphere — Spearman RSA"
        ),
    }


def main() -> None:
    project_root_default = _default_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Run, aggregate, and plot Spearman odor-identity ROI "
            "RSA separately for the two Kahnt OFC K=2 clusters."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="SPCD data root containing GLM/ and masks/gm_0.1.nii.",
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=project_root_default,
        help="Project clone root. Default: infer from this script location.",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=None,
        help=(
            "Prepared K=2 mask directory. Default: "
            "<project_root>/masks/OFC_clusterMap_K2_gm0p1."
        ),
    )
    parser.add_argument(
        "--repo_output_dir",
        type=Path,
        default=None,
        help=(
            "Root for commit-ready rank-RSA CSVs. Default: <project_root>/"
            "rsa_roi/probe_imagewise_odor_identity_no_image_first_image/"
            "premeal_ofc_k2_spearman."
        ),
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=None,
        help="Plot output directory. Default: <project_root>/plots.",
    )
    parser.add_argument(
        "--clusters",
        nargs="+",
        choices=sorted(CLUSTERS),
        default=sorted(CLUSTERS),
        help="Clusters to run. Default: cluster1 cluster2.",
    )
    parser.add_argument(
        "--hemispheres",
        nargs="+",
        choices=HEMISPHERES,
        default=["bilateral"],
        help=(
            "ROI scopes to run. Default: bilateral. Use 'left right' for "
            "the unilateral analyses."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs. Default: discover all premeal subjects.",
    )
    parser.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=DEFAULT_CUE_SUBSETS,
        default=list(DEFAULT_CUE_SUBSETS),
        help="Cue subsets to run. Default: both.",
    )
    parser.add_argument(
        "--skip_subs",
        nargs="*",
        default=list(DEFAULT_SKIP_SUBS),
        help="Subjects excluded during aggregation and plotting.",
    )
    parser.add_argument(
        "--save_diagnostics",
        action="store_true",
        help=(
            "Save subject-level patterns and selected neural/model RDM pairs "
            "in each cluster directory."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop the ROI analysis at the first failed subject/subset.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Run subject and group analyses without generating plots.",
    )
    parser.add_argument(
        "--plot_formats",
        nargs="+",
        default=["pdf", "png"],
        help="Plot formats. Default: pdf png.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print all commands without requiring the external SPCD data.",
    )
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    mask_dir = (
        args.mask_dir.expanduser().resolve()
        if args.mask_dir is not None
        else project_root / "masks" / "OFC_clusterMap_K2_gm0p1"
    )
    repo_output_dir = (
        args.repo_output_dir.expanduser().resolve()
        if args.repo_output_dir is not None
        else (
            project_root
            / "rsa_roi"
            / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_ofc_k2_spearman"
        )
    )
    plot_dir = (
        args.plot_dir.expanduser().resolve()
        if args.plot_dir is not None
        else project_root / "plots"
    )

    analysis_dir = Path(__file__).resolve().parent
    run_script = analysis_dir / "run_roi_rsa_spearman.py"
    aggregate_script = analysis_dir / "aggregate_roi_rsa_spearman.py"
    plot_script = (
        project_root
        / "scripts_MRI"
        / "RSA_plot"
        / "plot_ofc_cluster_map_k2_roi_rsa_spearman.py"
    )
    _require_file(run_script, "Spearman ROI RSA script")
    _require_file(aggregate_script, "Spearman group aggregation script")
    if not args.skip_plots:
        _require_file(plot_script, "Spearman paired plotting script")
    if not args.dry_run:
        _preflight_base_dir(base_dir)

    print("=" * 80)
    print("Kahnt OFC K=2 Spearman ROI RSA")
    print("=" * 80)
    print(f"Project root:     {project_root}")
    print(f"SPCD data root:   {base_dir}")
    print(f"Prepared masks:   {mask_dir}")
    print(f"Repository CSVs:  {repo_output_dir}")
    print(f"Plots:            {plot_dir}")
    print(f"Clusters:         {', '.join(args.clusters)}")
    print(f"Hemispheres:      {', '.join(args.hemispheres)}")
    print("Existing OLS RSA: untouched")
    print(f"Dry run:          {args.dry_run}")
    print("=" * 80)

    for cluster_label in args.clusters:
        spec = CLUSTERS[cluster_label]
        for hemisphere in args.hemispheres:
            scope = _analysis_scope(spec, hemisphere)
            mask_path = mask_dir / scope["mask_filename"]
            try:
                _require_file(
                    mask_path,
                    f"Prepared {hemisphere} mask for {cluster_label}",
                )
            except FileNotFoundError as error:
                if hemisphere != "bilateral":
                    raise FileNotFoundError(
                        f"{error}\nGenerate hemisphere masks with: python "
                        "scripts_MRI/roi_mask_prep/"
                        "prepare_ofc_cluster_map_k2_hemisphere_masks.py"
                    ) from error
                raise

            cluster_output_dir = repo_output_dir / scope["output_key"]
            subject_csv = (
                cluster_output_dir
                / f"roi-{scope['roi_name']}_subject_spearman.csv"
            )
            group_csv = (
                cluster_output_dir
                / f"roi-{scope['roi_name']}_spearman_group_summary.csv"
            )
            plot_prefix = plot_dir / scope["plot_stem"]

            print("\n" + "-" * 80)
            print(
                f"{cluster_label}, {hemisphere}: {scope['roi_name']}"
            )
            print("-" * 80)

            run_command = [
                sys.executable,
                str(run_script),
                "--base_dir",
                str(base_dir),
                "--roi_mask",
                str(mask_path),
                "--roi_name",
                scope["roi_name"],
                "--roi_threshold",
                "0",
                "--center_within_run",
                "--drop_nonfinite_voxels",
                "--output_dir",
                str(cluster_output_dir),
                "--cue_subsets",
                *args.cue_subsets,
            ]
            if args.subjects:
                run_command.extend(["--subjects", *args.subjects])
            if args.save_diagnostics:
                run_command.append("--save_diagnostics")
            if args.strict:
                run_command.append("--strict")
            _run_command(run_command, dry_run=args.dry_run)

            aggregate_command = [
                sys.executable,
                str(aggregate_script),
                "--input_csv",
                str(subject_csv),
                "--output_csv",
                str(group_csv),
                "--roi_name",
                scope["roi_name"],
                "--skip_subs",
                *args.skip_subs,
            ]
            _run_command(aggregate_command, dry_run=args.dry_run)

            if not args.skip_plots:
                plot_command = [
                    sys.executable,
                    str(plot_script),
                    "--input_csv",
                    str(subject_csv),
                    "--summary_csv",
                    str(group_csv),
                    "--output_prefix",
                    str(plot_prefix),
                    "--title",
                    scope["plot_title"],
                    "--skip_subs",
                    *args.skip_subs,
                    "--formats",
                    *args.plot_formats,
                ]
                _run_command(plot_command, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete; no analyses, CSVs, or plots were created.")
    else:
        print("\nAll requested K=2 Spearman analyses completed.")


if __name__ == "__main__":
    main()
