import os
import shutil

# === CONFIG ===
source_root = "/Users/liuq13/SPCD_data"
target_root = "/Users/liuq13/SPCD_glm_upload"
subject_prefix = "sub-"
task_folder_name = "character"

# === Clean up existing target ===
if os.path.exists(target_root):
    shutil.rmtree(target_root)
os.makedirs(target_root)


# === Copy only relevant character nii and tsv files ===
nii_source = os.path.join(source_root, "nii")
nii_target = os.path.join(target_root, "nii")
os.makedirs(nii_target)

for subj in os.listdir(nii_source):
    subj_root = os.path.join(nii_source, subj, "func_PAID", task_folder_name)
    if subj.startswith("sub-") and os.path.isdir(subj_root):
        dest_path = os.path.join(nii_target, subj, "func_PAID", task_folder_name)
        os.makedirs(dest_path, exist_ok=True)

        for fname in os.listdir(subj_root):
            if fname.startswith("s2w2") and fname.endswith(".nii"):
                shutil.copy2(os.path.join(subj_root, fname), dest_path)
                print(f"✅ Copied NIfTI: {subj}/{fname}")
            elif fname.endswith(".tsv"):
                shutil.copy2(os.path.join(subj_root, fname), dest_path)
                print(f"✅ Copied TSV: {subj}/{fname}")


# === Copy only character nuisance regressors ===
nreg_source = os.path.join(source_root, "NRegressor")
nreg_target = os.path.join(target_root, "NRegressor")

for subj in os.listdir(nreg_source):
    subj_path = os.path.join(nreg_source, subj)
    if not subj.startswith("sub-") or not os.path.isdir(subj_path):
        continue

    # Copy only relevant files
    out_subj_path = os.path.join(nreg_target, subj)
    os.makedirs(out_subj_path, exist_ok=True)

    for fname in os.listdir(subj_path):
        if "nuisance_regressors_character_run_" in fname and fname.endswith("_zscored.txt"):
            src = os.path.join(subj_path, fname)
            dst = os.path.join(out_subj_path, fname)
            shutil.copy2(src, dst)
            print(f"✅ Copied: NRegressor/{subj}/{fname}")



