import os
import shutil


def prepare_subset_rsa(
    full_data_dir: str,
    subset_dir: str,
    session_folders=("premeal_trialwise_2mm",),
    mask_relpath=("masks", "gm_0.1.nii"),
    dry_run: bool = False,
):
    """
    Prepare a subset of files for RSA searchlight on Biowulf.

    Copies:
      - GLM/<session_folder>/<sub-*> completely (contains betas + CSVs)
      - masks/gm_0.1.nii

    Parameters
    ----------
    full_data_dir : str
        Full dataset root (e.g., /Users/liuq13/SPCD_data)
    subset_dir : str
        Output subset root (e.g., /Users/liuq13/SPCD_RSA_probe_subset)
    session_folders : tuple[str]
        One or more session folders, e.g. ("premeal_trialwise_2mm", "postmeal_trialwise_2mm")
    mask_relpath : tuple[str]
        Relative path parts to the GM mask inside full_data_dir
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

    # Copy GLM folders
    for session_folder in session_folders:
        glm_root = os.path.join(full_data_dir, "GLM", session_folder)
        if not os.path.isdir(glm_root):
            raise FileNotFoundError(f"GLM session folder not found: {glm_root}")

        sub_list = sorted([
            d for d in os.listdir(glm_root)
            if d.startswith("sub-") and os.path.isdir(os.path.join(glm_root, d))
        ])
        print(f"🧾 Found {len(sub_list)} subjects in {session_folder}")

        glm_dst_root = os.path.join(subset_dir, "GLM", session_folder)

        for sub in sub_list:
            src_glm_sub = os.path.join(glm_root, sub)
            dst_glm_sub = os.path.join(glm_dst_root, sub)

            if dry_run:
                print(f"[DRY RUN] Would copytree: {src_glm_sub} -> {dst_glm_sub}")
                continue

            print(f"📦 Copying {session_folder}/{sub} ...")
            os.makedirs(glm_dst_root, exist_ok=True)
            shutil.copytree(src_glm_sub, dst_glm_sub, dirs_exist_ok=True)

    print("✅ Subset preparation complete.")


if __name__ == "__main__":
    # Change these before running locally
    full_data_dir = "/Users/liuq13/SPCD_data"
    subset_dir = "/Users/liuq13/SPCD_RSA_probe_subset"

    prepare_subset_rsa(
        full_data_dir=full_data_dir,
        subset_dir=subset_dir,
        session_folders=("premeal_trialwise_2mm", "postmeal_trialwise_2mm"),  # add "postmeal_trialwise_2mm" if needed
        dry_run=False
    )
