#!/usr/bin/env python3
"""
prepare_rsa_subset.py

Prepare a minimal subset for premeal odor-identity local-visual first-image RSA on Biowulf.
"""

import os
import shutil
from pathlib import Path


PACKAGE_FILES = [
    "__init__.py",
    "run_rsa.py",
    "run_roi_rsa.py",
    "aggregate_roi_rsa.py",
    "searchlight.py",
    "io_imagewise.py",
    "estimator.py",
    "models.py",
    "visual_rdm.py",
    "neural_rdm.py",
    "rsa_stats.py",
    "save_maps.py",
    "make_swarm_imagewise.py",
    "README.md",
]


def copy_file(src, dst, dry_run=False):
    if not src.exists():
        print(f"Missing file, skipping: {src}")
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


def copy_package_files(source_package_dir, subset_dir, package_name="rsa_imagewise_odor_identity_local_visual_first_image", dry_run=False):
    copied = 0
    for fname in PACKAGE_FILES:
        src = source_package_dir / fname
        dst = subset_dir / package_name / fname
        copied += int(copy_file(src, dst, dry_run=dry_run))
    return copied


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
        print(f"Missing GLM subject folder, skipping: {src_dir}")
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

    if session == "premeal":
        src_dir = full_data_dir / "nii" / sub / "func_PAID" / session
        dst_dir = subset_dir / "nii" / sub / "func_PAID" / session
        if src_dir.is_dir():
            for p in sorted(src_dir.glob("*_events.tsv")):
                copied += int(copy_file(p, dst_dir / p.name, dry_run=dry_run))

    for stim_type in ("character", "place"):
        src_dir = full_data_dir / "nii" / sub / "func_PAID" / stim_type
        dst_dir = subset_dir / "nii" / sub / "func_PAID" / stim_type
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.glob("*_events.tsv")):
            copied += int(copy_file(p, dst_dir / p.name, dry_run=dry_run))

    return copied


def copy_localizer_glm_subject_folder(full_data_dir, subset_dir, sub, stim_type, dry_run=False):
    copied = 0
    session_folder = f"{stim_type}_trialwise_2mm"
    src_dir = full_data_dir / "GLM" / session_folder / sub
    dst_dir = subset_dir / "GLM" / session_folder / sub

    if not src_dir.is_dir():
        print(f"Missing {stim_type} localizer GLM folder, skipping: {src_dir}")
        return copied

    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if lower.endswith((".nii", ".nii.gz", ".img", ".hdr")) and lower.startswith("beta"):
            copied += int(copy_file(p, dst_dir / p.name, dry_run=dry_run))

    return copied


def copy_task_info(full_data_dir, subset_dir, dry_run=False):
    copied = 0
    for rel_path in ("task_info/subject_info.csv", "subject_info.csv"):
        src = full_data_dir / rel_path
        if src.exists():
            copied += int(copy_file(src, subset_dir / rel_path, dry_run=dry_run))
            break
    return copied


def prepare_subset_imagewise(
    full_data_dir,
    subset_dir,
    source_package_dir,
    sessions=("premeal",),
    subjects=None,
    package_name="rsa_imagewise_odor_identity_local_visual_first_image",
    dry_run=False,
):
    full_data_dir = Path(full_data_dir)
    subset_dir = Path(subset_dir)
    source_package_dir = Path(source_package_dir)

    total_copied = 0

    print("=" * 70)
    print("Preparing premeal odor-identity local-visual first-image RSA subset")
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

    print("\n[2/4] Copying mask and task info...")
    total_copied += int(copy_file(full_data_dir / "masks" / "gm_0.1.nii", subset_dir / "masks" / "gm_0.1.nii", dry_run=dry_run))
    total_copied += copy_task_info(full_data_dir, subset_dir, dry_run=dry_run)

    print("\n[3/4] Copying premeal probe subject/session data...")
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

    print("\n[4/4] Copying independent localizer GLMs...")
    for sub in subjects:
        n_char = copy_localizer_glm_subject_folder(full_data_dir, subset_dir, sub, "character", dry_run=dry_run)
        n_place = copy_localizer_glm_subject_folder(full_data_dir, subset_dir, sub, "place", dry_run=dry_run)
        total_copied += n_char + n_place
        print(f"{sub}: character={n_char}, place={n_place}")

    print("\n" + "=" * 70)
    print("Done")
    print(f"Total files copied: {total_copied}" + (" (dry run)" if dry_run else ""))
    print(f"Subset dir: {subset_dir}")
    print("\nNext:")
    print(f"  rsync -avP {subset_dir}/ biowulf:/data/liuq13/{subset_dir.name}/")
    print("=" * 70)


if __name__ == "__main__":
    full_data_dir = "/Users/liuq13/SPCD_data"
    subset_dir = "/Users/liuq13/SPCD_RSA_imagewise_odor_identity_local_visual_first_image_subset"
    source_package_dir = "/Users/liuq13/project-spcd/scripts_MRI/rsa_imagewise_odor_identity_local_visual_first_image"

    prepare_subset_imagewise(
        full_data_dir=full_data_dir,
        subset_dir=subset_dir,
        source_package_dir=source_package_dir,
        sessions=("premeal",),
        subjects=None,
        package_name="rsa_imagewise_odor_identity_local_visual_first_image",
        dry_run=False,
    )
