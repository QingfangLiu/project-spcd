#!/usr/bin/env python3
"""
Prepare a subset of GLM files for cuewise RSA searchlight on Biowulf.

Copies only the necessary files:
  - GLM/<session>_cuewise_2mm/<sub>/<mode>/ (beta/tmap NIfTIs + index CSVs)
  - masks/gm_0.1.nii

Usage:
    python prepare_rsa_subset_cuewise.py

Then transfer to Biowulf:
    rsync -avP /path/to/subset/ biowulf:/data/liuq13/SPCD_RSA_cuewise/
"""

import os
import shutil


def prepare_subset_rsa(
    full_data_dir: str,
    subset_dir: str,
    session_folders=("premeal_cuewise_2mm", "postmeal_cuewise_2mm"),
    modes=("both",),  # which mode folders to copy: "both", "img1only", "img2only"
    mask_relpath=("masks", "gm_0.1.nii"),
    stat_types=("beta", "tmap"),  # which file types to copy; None = copy everything
    dry_run: bool = False,
):
    """
    Prepare a subset of files for cuewise RSA searchlight on Biowulf.

    Copies:
      - GLM/<session_folder>/<sub>/<mode>/ (cuewise betas/tmaps + index CSVs)
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
    """
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
                    print(f"[DRY RUN] Would copy: {src_mode_dir} -> {dst_mode_dir}")
                    continue

                os.makedirs(dst_mode_dir, exist_ok=True)

                # Copy files selectively or entirely
                files_copied = 0
                for fname in os.listdir(src_mode_dir):
                    src_file = os.path.join(src_mode_dir, fname)
                    
                    # Skip subdirectories
                    if os.path.isdir(src_file):
                        continue

                    # Filter by stat_type if specified
                    if stat_types is not None:
                        # Always copy CSV files (index files)
                        is_csv = fname.endswith(".csv")
                        # Copy if filename starts with any stat_type prefix
                        is_stat = any(fname.startswith(st) for st in stat_types)
                        
                        if not (is_csv or is_stat):
                            continue

                    dst_file = os.path.join(dst_mode_dir, fname)
                    shutil.copy2(src_file, dst_file)
                    files_copied += 1

                total_files += files_copied
                print(f"  📦 {session_folder}/{sub}/{mode}: {files_copied} files")

    print()
    print("=" * 60)
    print(f"✅ Subset preparation complete!")
    print(f"   Subjects: {total_subjects}")
    print(f"   Files copied: {total_files}")
    print(f"   Output: {subset_dir}")
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
        session_folders=("premeal_cuewiseLSS_2mm", "postmeal_cuewiseLSS_2mm"),
        modes=("both",),  # add "img1only", "img2only" if needed
        stat_types=("beta", "tmap"),  # set to None to copy everything
        dry_run=False,  # set to True to preview without copying
    )