import os
import shutil

def prepare_subset(
    full_data_dir: str,
    subset_dir: str,
    session_folder: str = "premeal_trialwise_2mm",
    code_files: tuple = (
        "probe_decoding_utils_xcue.py",
        "run_searchlight_crosscue.py",
        "make_searchlight_swarm_withinset_crosscue.py",
    ),
):
    """
    Prepare a minimal subset of SPCD decoding inputs for WITHIN-SET cross-cue decoding on Biowulf.

    Copies:
      - GLM/<session_folder>/<sub-*>  (betas + *_beta_index.csv)
      - masks/gm_0.1.nii

    Optionally copies code files into:
      - <subset_dir>/code/
    so the subset folder is self-contained for running on Biowulf.
    """

    # --- Source roots ---
    glm_root = os.path.join(full_data_dir, "GLM", session_folder)
    mask_src = os.path.join(full_data_dir, "masks", "gm_0.1.nii")

    if not os.path.isdir(glm_root):
        raise FileNotFoundError(f"GLM folder not found: {glm_root}")
    if not os.path.exists(mask_src):
        raise FileNotFoundError(f"Mask not found: {mask_src}")

    # Subject list from GLM/<session_folder>
    sub_list = sorted([
        d for d in os.listdir(glm_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_root, d))
    ])
    print(f"🧾 Found {len(sub_list)} subjects in GLM/{session_folder}")

    # --- Destination roots ---
    glm_dst_root = os.path.join(subset_dir, "GLM", session_folder)
    mask_dst = os.path.join(subset_dir, "masks", "gm_0.1.nii")

    # Copy mask
    os.makedirs(os.path.dirname(mask_dst), exist_ok=True)
    shutil.copy2(mask_src, mask_dst)
    print(f"✅ Copied mask -> {mask_dst}")

    # Copy GLM folders per subject
    os.makedirs(glm_dst_root, exist_ok=True)
    for sub in sub_list:
        src_glm_sub = os.path.join(glm_root, sub)
        dst_glm_sub = os.path.join(glm_dst_root, sub)
        print(f"📦 Copying {sub}...")

        # If rerunning, keep existing and merge/overwrite
        shutil.copytree(src_glm_sub, dst_glm_sub, dirs_exist_ok=True)

    print("\n✅ Subset preparation complete.")
    print(f"📁 Subset dir: {subset_dir}")
    print(f"   - GLM/{session_folder}/sub-*")
    print("   - masks/gm_0.1.nii")
    if copy_code:
        print("   - code/ (optional runner/utils/swarm scripts)")

if __name__ == "__main__":
    full_data_dir = "/Users/liuq13/SPCD_data"
    subset_dir = "/Users/liuq13/SPCD_subset"
    session_folder = "premeal_trialwise_2mm"  # change if needed

    prepare_subset(
        full_data_dir=full_data_dir,
        subset_dir=subset_dir,
        session_folder=session_folder
    )
