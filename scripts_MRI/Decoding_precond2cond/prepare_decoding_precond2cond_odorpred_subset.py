# prepare_decoding_precond2cond_odorpred_subset.py
#
# Prepare data subset for Goal 4:
#   Day-1 pre-conditioned cues (A; localizer) →
#   Day-2 odor prediction conditioned cues (B)
#
# Copies, for each eligible subject:
#   - GLM/<odorpred_glm_folder>/<sub-XX>/           (odor prediction cue betas + CSVs)
#   - GLM/{character|place}_trialwise_2mm/<sub-XX>/ (Day-1 localizer betas)
#   - nii/<sub-XX>/func_PAID/{character|place}/*.tsv (localizer events)
# And also:
#   - masks/gm_0.1.nii
#   - task_info/subject_info.csv
#

import os
import shutil

def copytree(src, dst):
    """Robust directory copy (like cp -r)."""
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(src)

def prepare_subset_simple(
    full_data_dir,
    subset_dir,
    odorpred_glm_folder="odorpred_cueEARLY_trialwise_2mm",
):
    # ----------------------------
    # Create folder structure
    # ----------------------------
    os.makedirs(os.path.join(subset_dir, "GLM"), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, "nii"), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, "task_info"), exist_ok=True)

    # ----------------------------
    # 1. Copy masks
    # ----------------------------
    mask_src = os.path.join(full_data_dir, "masks", "gm_0.1.nii")
    mask_dst = os.path.join(subset_dir, "masks", "gm_0.1.nii")
    shutil.copy2(mask_src, mask_dst)
    print(f"Copied mask → {mask_dst}")

    # ----------------------------
    # 2. Copy subject_info.csv
    # ----------------------------
    si_src = os.path.join(full_data_dir, "task_info", "subject_info.csv")
    si_dst = os.path.join(subset_dir, "task_info", "subject_info.csv")
    shutil.copy2(si_src, si_dst)
    print(f"Copied subject_info.csv → {si_dst}")

    # ----------------------------
    # 3. Copy ALL Day-1 GLMs (char + place)
    # ----------------------------
    for glm_name in ["character_trialwise_2mm", "place_trialwise_2mm"]:
        src = os.path.join(full_data_dir, "GLM", glm_name)
        dst = os.path.join(subset_dir, "GLM", glm_name)
        if os.path.isdir(src):
            copytree(src, dst)
            print(f"Copied {glm_name}")

    # ----------------------------
    # 4. Copy ALL Day-2 odor prediction GLM
    # ----------------------------
    od_src = os.path.join(full_data_dir, "GLM", odorpred_glm_folder)
    od_dst = os.path.join(subset_dir, "GLM", odorpred_glm_folder)
    copytree(od_src, od_dst)
    print(f"Copied {odorpred_glm_folder}")

    # ----------------------------
    # 5. Copy localizer events (char and place) for ALL subjects
    # ----------------------------
    nii_root = os.path.join(full_data_dir, "nii")
    for sub in os.listdir(nii_root):
        if not sub.startswith("sub-"):
            continue
        for stim_type in ["character", "place"]:
            src_dir = os.path.join(full_data_dir, "nii", sub, "func_PAID", stim_type)
            if os.path.isdir(src_dir):
                dst_dir = os.path.join(subset_dir, "nii", sub, "func_PAID", stim_type)
                os.makedirs(dst_dir, exist_ok=True)
                for fname in os.listdir(src_dir):
                    if fname.endswith(".tsv"):
                        shutil.copy2(
                            os.path.join(src_dir, fname),
                            os.path.join(dst_dir, fname),
                        )
                print(f"Copied events for {sub}/{stim_type}")

    print("\n✅ DONE — simple subset is ready.\n")


if __name__ == "__main__":
    full_data_dir = "/Users/liuq13/SPCD_data"
    subset_dir = "/Users/liuq13/SPCD_odorpred_cueEXT5s_subset"

    prepare_subset_simple(
        full_data_dir,
        subset_dir,
        odorpred_glm_folder="odorpred_cueEXT5s_trialwise_2mm",
    )

