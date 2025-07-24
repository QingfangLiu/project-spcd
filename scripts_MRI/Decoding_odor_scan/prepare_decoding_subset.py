import os
import shutil

def copy_if_exists(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    else:
        print(f"⚠️ Missing: {src}")

def prepare_subset(full_data_dir, subset_dir):
    # Dynamically get subject list
    glm_root = os.path.join(full_data_dir, 'GLM', 'odor_scan_trialwise_2mm')
    sub_list = sorted([
        d for d in os.listdir(glm_root)
        if d.startswith("sub-")
    ])
    print(f"🧾 Found {len(sub_list)} subjects")

    # Paths
    nii_src = os.path.join(full_data_dir, "nii")
    mask_src = os.path.join(full_data_dir, "masks", "gm_0.1.nii")

    glm_dst_root = os.path.join(subset_dir, "GLM", "odor_scan_trialwise_2mm")
    nii_dst_root = os.path.join(subset_dir, "nii")
    mask_dst = os.path.join(subset_dir, "masks", "gm_0.1.nii")

    # Copy mask
    os.makedirs(os.path.dirname(mask_dst), exist_ok=True)
    shutil.copy2(mask_src, mask_dst)

    for sub in sub_list:
        print(f"📦 Copying {sub}...")

        # Copy GLM betas
        src_glm_sub = os.path.join(glm_root, sub)
        dst_glm_sub = os.path.join(glm_dst_root, sub)
        if os.path.exists(src_glm_sub):
            shutil.copytree(src_glm_sub, dst_glm_sub, dirs_exist_ok=True)
        else:
            print(f"⚠️ Missing GLM for {sub}")

        # Copy event files
        src_evt_dir = os.path.join(nii_src, sub, "func_PAID", "odor_scan")
        dst_evt_dir = os.path.join(nii_dst_root, sub, "func_PAID", "odor_scan")
        if os.path.exists(src_evt_dir):
            os.makedirs(dst_evt_dir, exist_ok=True)
            for fname in os.listdir(src_evt_dir):
                if fname.endswith(".tsv") and "task-odor" in fname:
                    copy_if_exists(
                        os.path.join(src_evt_dir, fname),
                        os.path.join(dst_evt_dir, fname)
                    )
        else:
            print(f"⚠️ Missing event folder for {sub}")

if __name__ == "__main__":
    full_data_dir = "/Users/liuq13/SPCD_data"           # 🔁 Update to full data path
    subset_dir = "/Users/liuq13/SPCD_decoding_subset"               # 🔁 Output path (can be relative)
    prepare_subset(full_data_dir, subset_dir)
