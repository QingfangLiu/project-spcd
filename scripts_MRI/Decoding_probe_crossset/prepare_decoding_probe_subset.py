import os
import shutil

def prepare_subset(full_data_dir, subset_dir, session_folder="premeal_trialwise_2mm"):
    """
    Prepare a subset of files for cue decoding on Biowulf.
    - Copies GLM/<session_folder>/<sub-*> completely (contains betas + TSVs)
    - Copies masks/gm_0.1.nii
    """

    # Subject list from GLM/<session_folder>
    glm_root = os.path.join(full_data_dir, "GLM", session_folder)
    sub_list = sorted([
        d for d in os.listdir(glm_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_root, d))
    ])
    print(f"🧾 Found {len(sub_list)} subjects in {session_folder}")

    # Destination paths
    glm_dst_root = os.path.join(subset_dir, "GLM", session_folder)
    mask_src = os.path.join(full_data_dir, "masks", "gm_0.1.nii")
    mask_dst = os.path.join(subset_dir, "masks", "gm_0.1.nii")

    # Copy mask
    os.makedirs(os.path.dirname(mask_dst), exist_ok=True)
    shutil.copy2(mask_src, mask_dst)
    print(f"✅ Copied mask to {mask_dst}")

    for sub in sub_list:
        print(f"📦 Copying {sub}...")
        src_glm_sub = os.path.join(glm_root, sub)
        dst_glm_sub = os.path.join(glm_dst_root, sub)

        if os.path.exists(src_glm_sub):
            shutil.copytree(src_glm_sub, dst_glm_sub, dirs_exist_ok=True)
        else:
            print(f"⚠️ Missing GLM folder for {sub}")

if __name__ == "__main__":
    # Change these before running
    full_data_dir = "/Users/liuq13/SPCD_data"
    subset_dir = "/Users/liuq13/SPCD_decoding_probe_subset"
    session_folder = "premeal_trialwise_2mm"  # adjust if needed

    prepare_subset(full_data_dir, subset_dir, session_folder)
