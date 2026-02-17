#!/usr/bin/env python3
"""
Prepare a subset of GLM files for cuewise RSA searchlight on Biowulf.

Copies only the necessary files:
  - GLM/<session_folder>/<sub>/<mode>/ (beta/tmap NIfTIs + index CSVs)
  - GLM/<session_folder>/<sub>/<mode>/run-XX_spm/ (SPM outputs needed for noise/whitening; default: ResMS.nii + SPM.mat)
  - masks/gm_0.1.nii

Usage:
    python prepare_rsa_subset_cuewise.py

Then transfer to Biowulf:
    rsync -avP /path/to/subset/ biowulf:/data/liuq13/SPCD_RSA_cuewise/
"""

import os
import shutil
from typing import Iterable, Optional, Tuple


def _copy_file(src_file: str, dst_file: str, dry_run: bool) -> bool:
    """Copy a single file. Returns True if copied (or would copy in dry_run)."""
    if dry_run:
        print(f"[DRY RUN] Would copy: {src_file} -> {dst_file}")
        return True
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    shutil.copy2(src_file, dst_file)
    return True


def prepare_subset_rsa(
    full_data_dir: str,
    subset_dir: str,
    session_folders: Tuple[str, ...] = ("premeal_cuewise_2mm", "postmeal_cuewise_2mm"),
    modes: Tuple[str, ...] = ("both",),  # which mode folders to copy: "both", "img1only", "img2only"
    mask_relpath: Tuple[str, ...] = ("masks", "gm_0.1.nii"),
    stat_types: Optional[Tuple[str, ...]] = ("beta", "tmap"),  # None = copy everything in mode folder
    dry_run: bool = False,

    # NEW: copy run-specific SPM folders (for ResMS whitening / provenance)
    copy_spm_folders: bool = True,
    spm_copy_mode: str = "minimal",  # "minimal" (default) or "all"
    spm_keep_files: Tuple[str, ...] = ("ResMS.nii", "SPM.mat"),
):
    """
    Prepare a subset of files for cuewise RSA searchlight on Biowulf.

    Copies:
      - GLM/<session_folder>/<sub>/<mode>/ (cuewise betas/tmaps + index CSVs)
      - (optional) GLM/<session_folder>/<sub>/<mode>/run-XX_spm/ (default minimal: ResMS.nii + SPM.mat)
      - masks/gm_0.1.nii

    Parameters
    ----------
    full_data_dir : str
        Full dataset root (e.g., /Users/liuq13/SPCD_data)
    subset_dir : str
        Output subset root (e.g., /Users/liuq13/SPCD_RSA_cuewise_subset)
    session_folders : tuple[str]
        Session folders to copy, e.g. ("premeal_cuewise_2mm", "postmeal_cuewise_2mm")
    modes : tuple[str]
        Which mode subfolders to copy: ("both",), ("img1only",), or multiple
    mask_relpath : tuple[str]
        Relative path parts to the GM mask inside full_data_dir
    stat_types : tuple[str] or None
        If provided, only copy files starting with these prefixes (e.g., "beta", "tmap")
        plus CSV files. If None, copy everything in the mode folder.
    dry_run : bool
        If True, print actions without copying
    copy_spm_folders : bool
        If True, also copy run-specific SPM folders inside each mode folder.
        This is required if you want to use run-specific ResMS.nii whitening on Biowulf.
    spm_copy_mode : {"minimal", "all"}
        - "minimal": copy only files listed in spm_keep_files (default: ResMS.nii + SPM.mat)
        - "all": copy every file in run-XX_spm (can be very large)
    spm_keep_files : tuple[str]
        File basenames to keep when spm_copy_mode="minimal".
    """

    if spm_copy_mode not in {"minimal", "all"}:
        raise ValueError("spm_copy_mode must be 'minimal' or 'all'")

    # Copy mask
    mask_src = os.path.join(full_data_dir, *mask_relpath)
    mask_dst = os.path.join(subset_dir, *mask_relpath)

    if not os.path.exists(mask_src):
        raise FileNotFoundError(f"Mask not found: {mask_src}")

    if dry_run:
        print(f"[DRY RUN] Would copy mask: {mask_src} -> {mask_dst}")
    else:
        os.makedirs(os.path.dirname(mask_dst), exist_ok=True)
        shutil.copy2(mask_src, mask_dst)
        print(f"✅ Copied mask to {mask_dst}")

    # Track statistics
    total_files = 0
    total_subjects = 0
    total_spm_dirs = 0

    # Copy GLM folders
    for session_folder in session_folders:
        glm_root = os.path.join(full_data_dir, "GLM", session_folder)
        if not os.path.isdir(glm_root):
            print(f"⚠️  GLM session folder not found (skipping): {glm_root}")
            continue

        sub_list = sorted([
            d for d in os.listdir(glm_root)
            if d.startswith("sub-") and os.path.isdir(os.path.join(glm_root, d))
        ])
        print(f"🧾 Found {len(sub_list)} subjects in {session_folder}")
        total_subjects += len(sub_list)

        for sub in sub_list:
            for mode in modes:
                src_mode_dir = os.path.join(glm_root, sub, mode)
                dst_mode_dir = os.path.join(subset_dir, "GLM", session_folder, sub, mode)

                if not os.path.isdir(src_mode_dir):
                    print(f"  ⚠️  Missing mode folder (skipping): {src_mode_dir}")
                    continue

                if dry_run:
                    print(f"[DRY RUN] Would copy mode files: {src_mode_dir} -> {dst_mode_dir}")
                else:
                    os.makedirs(dst_mode_dir, exist_ok=True)

                # 1) Copy files in mode folder (betas/tmaps + CSVs), skipping subdirectories
                files_copied = 0
                for fname in os.listdir(src_mode_dir):
                    src_file = os.path.join(src_mode_dir, fname)

                    # Skip subdirectories here; run-XX_spm handled separately
                    if os.path.isdir(src_file):
                        continue

                    # Filter by stat_type if specified
                    if stat_types is not None:
                        is_csv = fname.endswith(".csv")
                        is_stat = any(fname.startswith(st) for st in stat_types)
                        if not (is_csv or is_stat):
                            continue

                    dst_file = os.path.join(dst_mode_dir, fname)
                    _copy_file(src_file, dst_file, dry_run=dry_run)
                    files_copied += 1

                total_files += files_copied
                print(f"  📦 {session_folder}/{sub}/{mode}: {files_copied} mode files" + (" (dry_run)" if dry_run else ""))

                # 2) Copy run-XX_spm folders (optional)
                if not copy_spm_folders:
                    continue

                spm_dirs = sorted([
                    d for d in os.listdir(src_mode_dir)
                    if d.startswith("run-") and d.endswith("_spm") and os.path.isdir(os.path.join(src_mode_dir, d))
                ])

                if len(spm_dirs) == 0:
                    print(f"    ⚠️  No run-XX_spm folders found under: {src_mode_dir}")
                    continue

                spm_files_copied = 0
                for spm_dirname in spm_dirs:
                    src_spm_dir = os.path.join(src_mode_dir, spm_dirname)
                    dst_spm_dir = os.path.join(dst_mode_dir, spm_dirname)
                    total_spm_dirs += 1

                    if dry_run:
                        print(f"    [DRY RUN] Would copy SPM folder: {src_spm_dir} -> {dst_spm_dir}")
                        continue

                    os.makedirs(dst_spm_dir, exist_ok=True)

                    if spm_copy_mode == "all":
                        # Copy all files in the run-XX_spm directory (non-recursive)
                        for fname in os.listdir(src_spm_dir):
                            src_file = os.path.join(src_spm_dir, fname)
                            if os.path.isdir(src_file):
                                continue
                            dst_file = os.path.join(dst_spm_dir, fname)
                            shutil.copy2(src_file, dst_file)
                            spm_files_copied += 1
                    else:
                        # Minimal: copy only keep files if present
                        for keep in spm_keep_files:
                            src_file = os.path.join(src_spm_dir, keep)
                            if not os.path.exists(src_file):
                                # For your use case, ResMS.nii is the key; SPM.mat optional
                                continue
                            dst_file = os.path.join(dst_spm_dir, keep)
                            shutil.copy2(src_file, dst_file)
                            spm_files_copied += 1

                total_files += spm_files_copied
                if not dry_run:
                    print(f"    🧠 {session_folder}/{sub}/{mode}: copied {spm_files_copied} files from {len(spm_dirs)} run-XX_spm folders")

    print()
    print("=" * 60)
    print("✅ Subset preparation complete!")
    print(f"   Subjects scanned:      {total_subjects}")
    print(f"   SPM run folders seen:  {total_spm_dirs} (across all subs/sessions/modes)")
    print(f"   Files copied:          {total_files}" + (" (dry_run)" if dry_run else ""))
    print(f"   Output:               {subset_dir}")
    print()
    print("Next steps:")
    print(f"  rsync -avP {subset_dir}/ biowulf:/data/liuq13/SPCD_RSA_cuewise/")
    print("=" * 60)


if __name__ == "__main__":
    # ===== CONFIGURE THESE BEFORE RUNNING =====
    full_data_dir = "/Users/liuq13/SPCD_data"
    subset_dir = "/Users/liuq13/SPCD_RSA_cuewise_subset"

    prepare_subset_rsa(
        full_data_dir=full_data_dir,
        subset_dir=subset_dir,
        session_folders=("premeal_cuewise_2mm", "postmeal_cuewise_2mm"),
        modes=("both",),  # add "img1only", "img2only" if needed
        stat_types=("beta", "tmap"),  # set to None to copy everything in the mode folder
        dry_run=False,  # set to True to preview without copying

        copy_spm_folders=True,
        spm_copy_mode="minimal",  # "minimal" (ResMS.nii + SPM.mat) or "all"
        spm_keep_files=("ResMS.nii", "SPM.mat"),
    )
