# prepare_decoding_crosstask_subset.py
import os
import shutil

def _copytree(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

def prepare_subset(full_data_dir, subset_dir,
                   premeal_folder="premeal_trialwise_2mm",
                   odor_glm_folder="odor_scan_trialwise_2mm"):
    """
    Prepare a subset of files for cross-task decoding (odor -> pre-meal probe).
    - Copies GLM/<premeal_folder>/<sub-*> (probe betas + CSVs)
    - Copies GLM/<odor_glm_folder>/<sub-*> (odor betas)
    - Copies nii/<sub-*>/func_PAID/odor_scan/*.tsv (odor events)
    - Copies masks/gm_0.1.nii
    """

    # 1) Subject list = intersection of pre-meal GLM and odor GLM folders
    premeal_root = os.path.join(full_data_dir, "GLM", premeal_folder)
    odor_root    = os.path.join(full_data_dir, "GLM", odor_glm_folder)

    subs_premeal = {d for d in os.listdir(premeal_root)
                    if d.startswith("sub-") and os.path.isdir(os.path.join(premeal_root, d))}
    subs_odor    = {d for d in os.listdir(odor_root)
                    if d.startswith("sub-") and os.path.isdir(os.path.join(odor_root, d))}
    sub_list = sorted(subs_premeal & subs_odor)

    print(f"🧾 Subjects with BOTH pre-meal + odor GLM: {len(sub_list)}")

    # 2) Copy mask
    mask_src = os.path.join(full_data_dir, "masks", "gm_0.1.nii")
    mask_dst = os.path.join(subset_dir, "masks", "gm_0.1.nii")
    os.makedirs(os.path.dirname(mask_dst), exist_ok=True)
    shutil.copy2(mask_src, mask_dst)
    print(f"✅ Copied mask → {mask_dst}")

    # 3) Copy GLM folders (pre-meal + odor) per subject
    for sub in sub_list:
        print(f"📦 {sub}")

        # GLM pre-meal
        src_premeal = os.path.join(premeal_root, sub)
        dst_premeal = os.path.join(subset_dir, "GLM", premeal_folder, sub)
        _copytree(src_premeal, dst_premeal)

        # GLM odor
        src_odor = os.path.join(odor_root, sub)
        dst_odor = os.path.join(subset_dir, "GLM", odor_glm_folder, sub)
        _copytree(src_odor, dst_odor)

        # Odor events TSVs (nii/<sub>/func_PAID/odor_scan/*.tsv)
        src_events_dir = os.path.join(full_data_dir, "nii", sub, "func_PAID", "odor_scan")


        if os.path.isdir(src_events_dir):
            dst_events_dir = os.path.join(subset_dir, "nii", sub, "func_PAID", "odor_scan")
            os.makedirs(dst_events_dir, exist_ok=True)
            for fname in os.listdir(src_events_dir):
                if fname.endswith(".tsv"):
                    shutil.copy2(
                        os.path.join(src_events_dir, fname),
                        os.path.join(dst_events_dir, fname)
                    )
        else:
            print(f"⚠️ Odor events folder missing for {sub}: {src_events_dir}")

if __name__ == "__main__":
    # ✏️ Edit these before running
    full_data_dir = "/Users/liuq13/SPCD_data"
    subset_dir    = "/Users/liuq13/SPCD_decoding_crosstask_subset"
    prepare_subset(full_data_dir, subset_dir)
