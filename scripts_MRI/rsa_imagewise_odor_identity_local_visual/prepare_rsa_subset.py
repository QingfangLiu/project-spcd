#!/usr/bin/env python3
"""
prepare_rsa_subset.py

Prepare a subset for local-visual odor-identity image-wise RSA on Biowulf.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence


PACKAGE_FILES = [
    "__init__.py",
    "run_rsa.py",
    "searchlight.py",
    "io_imagewise.py",
    "estimator.py",
    "models.py",
    "neural_rdm.py",
    "rsa_stats.py",
    "save_maps.py",
    "visual_rdm.py",
    "make_swarm_imagewise.py",
    "aggregate_rsa_group.m",
    "README.md",
]

LOCALIZER_STIM_TYPES = ("character", "place")


def copy_file(src, dst, dry_run=False):
    if not src.exists():
        print(f"⚠️ Missing file, skipping: {src}")
        return False
    if dry_run:
        print(f"[DRY RUN] {src} -> {dst}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def discover_subjects(full_data_dir, session):
    glm_dir = full_data_dir / "GLM" / f"{session}_trialwise_2mm"
    if not glm_dir.is_dir():
        raise FileNotFoundError(f"GLM folder not found: {glm_dir}")
    subs = sorted(
        p.name for p in glm_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )
    if not subs:
        raise RuntimeError(f"No subjects found under: {glm_dir}")
    return subs


def copy_package_files(source_package_dir, subset_dir, package_name="rsa_imagewise_odor_identity_local_visual", dry_run=False):
    copied = 0
    for fname in PACKAGE_FILES:
        src = source_package_dir / fname
        dst = subset_dir / package_name / fname
        copied += int(copy_file(src, dst, dry_run=dry_run))
    return copied


def copy_first_existing(candidates, dst, dry_run=False):
    for src in candidates:
        if src.exists():
            return int(copy_file(src, dst, dry_run=dry_run))
    print(f"⚠️ Missing file, skipping all candidates for: {dst}")
    for src in candidates:
        print(f"    checked: {src}")
    return 0


def extract_beta_files_from_csv(csv_path):
    import csv

    beta_files = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "beta_file" not in (reader.fieldnames or []):
            return beta_files
        for row in reader:
            bf = row.get("beta_file", "")
            if bf:
                beta_files.add(os.path.basename(bf))
    return beta_files


def copy_glm_subject_folder(full_data_dir, subset_dir, sub, session, dry_run=False):
    copied = 0
    session_folder = f"{session}_trialwise_2mm"
    src_dir = full_data_dir / "GLM" / session_folder / sub
    dst_dir = subset_dir / "GLM" / session_folder / sub

    if not src_dir.is_dir():
        print(f"⚠️ Missing GLM subject folder, skipping: {src_dir}")
        return copied

    csv_files = sorted(src_dir.glob("*.csv"))
    beta_files_needed = set()

    for csv_path in csv_files:
        copied += int(copy_file(csv_path, dst_dir / csv_path.name, dry_run=dry_run))
        beta_files_needed.update(extract_beta_files_from_csv(csv_path))

    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        lower = name.lower()

        is_beta_named = lower.startswith("beta")
        is_beta_referenced = name in beta_files_needed
        is_supported_img = lower.endswith((".nii", ".nii.gz", ".img", ".hdr"))

        if is_supported_img and (is_beta_named or is_beta_referenced):
            copied += int(copy_file(p, dst_dir / name, dry_run=dry_run))

    return copied


def copy_events_for_subject(full_data_dir, subset_dir, sub, session, dry_run=False):
    copied = 0
    src_dir = full_data_dir / "nii" / sub / "func_PAID" / session
    dst_dir = subset_dir / "nii" / sub / "func_PAID" / session

    if not src_dir.is_dir():
        print(f"⚠️ Missing events folder, skipping: {src_dir}")
        return copied

    for p in sorted(src_dir.glob("*_events.tsv")):
        copied += int(copy_file(p, dst_dir / p.name, dry_run=dry_run))

    return copied


def copy_localizer_glm_subject_folder(full_data_dir, subset_dir, sub, stim_type, dry_run=False):
    copied = 0
    glm_folder = f"{stim_type}_trialwise_2mm"
    src_dir = full_data_dir / "GLM" / glm_folder / sub
    dst_dir = subset_dir / "GLM" / glm_folder / sub

    if not src_dir.is_dir():
        print(f"⚠️ Missing localizer GLM subject folder, skipping: {src_dir}")
        return copied

    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        lower = p.name.lower()
        is_localizer_beta = lower.startswith("beta_run") and lower.endswith((
            ".nii",
            ".nii.gz",
            ".img",
            ".hdr",
        ))
        if is_localizer_beta:
            copied += int(copy_file(p, dst_dir / p.name, dry_run=dry_run))

    return copied


def copy_localizer_events_for_subject(full_data_dir, subset_dir, sub, stim_type, dry_run=False):
    copied = 0
    src_dir = full_data_dir / "nii" / sub / "func_PAID" / stim_type
    dst_dir = subset_dir / "nii" / sub / "func_PAID" / stim_type

    if not src_dir.is_dir():
        print(f"⚠️ Missing localizer events folder, skipping: {src_dir}")
        return copied

    for p in sorted(src_dir.glob(f"*_task-{stim_type}_run-*_events.tsv")):
        copied += int(copy_file(p, dst_dir / p.name, dry_run=dry_run))

    return copied


def prepare_subset_imagewise(
    full_data_dir,
    subset_dir,
    source_package_dir,
    sessions=("premeal",),
    subjects=None,
    package_name="rsa_imagewise_odor_identity_local_visual",
    dry_run=False,
):
    full_data_dir = Path(full_data_dir)
    subset_dir = Path(subset_dir)
    source_package_dir = Path(source_package_dir)
    project_root = source_package_dir.parents[1]

    total_copied = 0

    print("=" * 70)
    print("Preparing premeal odor-identity local-visual RSA subset")
    print("=" * 70)

    print("\n[1/4] Copying package files...")
    n_pkg = copy_package_files(
        source_package_dir=source_package_dir,
        subset_dir=subset_dir,
        package_name=package_name,
        dry_run=dry_run,
    )
    total_copied += n_pkg
    print(f"Copied package files: {n_pkg}")

    print("\n[2/4] Copying GM mask and subject info...")
    mask_src_candidates = [
        full_data_dir / "masks" / "gm_0.1.nii",
        project_root / "masks" / "gm_0.1.nii",
    ]
    mask_dst = subset_dir / "masks" / "gm_0.1.nii"
    total_copied += copy_first_existing(mask_src_candidates, mask_dst, dry_run=dry_run)

    subject_info_src_candidates = [
        full_data_dir / "task_info" / "subject_info.csv",
        project_root / "subject_info.csv",
    ]
    subject_info_dst = subset_dir / "task_info" / "subject_info.csv"
    total_copied += copy_first_existing(
        subject_info_src_candidates,
        subject_info_dst,
        dry_run=dry_run,
    )

    print("\n[3/4] Copying premeal probe data...")
    if subjects is None:
        subjects = discover_subjects(full_data_dir, sessions[0])

    for sub in subjects:
        print(f"\n{sub}")
        for session in sessions:
            print(f"  Session: {session}")
            n_glm = copy_glm_subject_folder(
                full_data_dir=full_data_dir,
                subset_dir=subset_dir,
                sub=sub,
                session=session,
                dry_run=dry_run,
            )
            n_ev = copy_events_for_subject(
                full_data_dir=full_data_dir,
                subset_dir=subset_dir,
                sub=sub,
                session=session,
                dry_run=dry_run,
            )
            total_copied += n_glm + n_ev
            print(f"    GLM files copied:    {n_glm}")
            print(f"    events files copied: {n_ev}")

    print("\n[4/4] Copying independent image localizer data...")
    for sub in subjects:
        print(f"\n{sub}")
        for stim_type in LOCALIZER_STIM_TYPES:
            n_glm = copy_localizer_glm_subject_folder(
                full_data_dir=full_data_dir,
                subset_dir=subset_dir,
                sub=sub,
                stim_type=stim_type,
                dry_run=dry_run,
            )
            n_ev = copy_localizer_events_for_subject(
                full_data_dir=full_data_dir,
                subset_dir=subset_dir,
                sub=sub,
                stim_type=stim_type,
                dry_run=dry_run,
            )
            total_copied += n_glm + n_ev
            print(f"  {stim_type}: GLM files copied={n_glm}, events files copied={n_ev}")

    print("\n" + "=" * 70)
    print("Done")
    print(f"Total files copied: {total_copied}" + (" (dry run)" if dry_run else ""))
    print(f"Subset dir: {subset_dir}")
    print("\nNext:")
    print(f"  rsync -avP {subset_dir}/ biowulf:/data/liuq13/{subset_dir.name}/")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a Biowulf subset for local-visual odor-identity RSA."
    )
    parser.add_argument("--full_data_dir", default="/Users/liuq13/SPCD_data")
    parser.add_argument(
        "--subset_dir",
        default="/Users/liuq13/SPCD_RSA_imagewise_odor_identity_local_visual_subset",
    )
    parser.add_argument(
        "--source_package_dir",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned copies without writing files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_subset_imagewise(
        full_data_dir=args.full_data_dir,
        subset_dir=args.subset_dir,
        source_package_dir=args.source_package_dir,
        sessions=("premeal",),
        subjects=args.subjects,
        package_name="rsa_imagewise_odor_identity_local_visual",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
