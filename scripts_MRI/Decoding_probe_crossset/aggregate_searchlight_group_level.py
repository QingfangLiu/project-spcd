import os
import glob
import numpy as np
from nilearn.image import concat_imgs, smooth_img, new_img_like
from scipy.stats import ttest_1samp

# ---- Settings ----
base_dir = "/Users/liuq13/SPCD_data/searchlight_maps"
subfolder = "probe_crossset"
modes = ["strict", "set2set"]
cues  = ["A", "B"]
class_list = [2, 3]   # <-- new loop

fwhm = 6
radius_str = "8mm"

for n_cls in class_list:
    for mode in modes:
        for cue in cues:
            print(f"\n>>> Group analysis | classes={n_cls} | mode={mode} | cue={cue}")

            searchlight_dir = os.path.join(base_dir, subfolder)
            output_dir = os.path.join(
                searchlight_dir,
                f"mode-{mode}_cue-{cue}_classes-{n_cls}",
                "group_stats"
            )
            os.makedirs(output_dir, exist_ok=True)

            # Adjust pattern to match filenames with class number
            pattern = os.path.join(
                searchlight_dir,
                f"sub-*_session-premeal_mode-{mode}_cue-{cue}_classes-{n_cls}_radius-{radius_str}.nii.gz"
            )
            map_files = sorted(glob.glob(pattern))
            print(f"Found {len(map_files)} searchlight maps")
            if len(map_files) == 0:
                print("No maps found. Skipping.")
                continue

            # ---- Load and smooth 4D image ----
            img_4d = concat_imgs(map_files)
            img_4d = smooth_img(img_4d, fwhm=fwhm)
            data_4d = img_4d.get_fdata()

            # ---- Reshape and clean ----
            n_voxels = np.prod(data_4d.shape[:-1])
            n_subjects = data_4d.shape[-1]
            data_2d = data_4d.reshape(n_voxels, n_subjects)
            valid_mask = ~np.any(np.isnan(data_2d), axis=1)
            valid_data = data_2d[valid_mask, :]

            # ---- One-sample t-test vs chance ----
            chance_level = 0.5 if n_cls == 2 else 1/3
            t_vals, _ = ttest_1samp(valid_data - chance_level, popmean=0, axis=1, nan_policy="omit")

            # ---- Map back to brain ----
            t_map = np.full((n_voxels,), np.nan)
            t_map[valid_mask] = t_vals
            t_map_3d = t_map.reshape(data_4d.shape[:-1])
            t_img = new_img_like(img_4d.slicer[..., 0], t_map_3d)

            # Mean accuracy
            mean_vals = np.full((n_voxels,), np.nan)
            mean_vals[valid_mask] = valid_data.mean(axis=1)
            mean_img = new_img_like(img_4d.slicer[..., 0], mean_vals.reshape(data_4d.shape[:-1]))

            # ---- Save ----
            t_img_path = os.path.join(output_dir, f"group_tmap_{mode}_cue-{cue}_classes-{n_cls}.nii.gz")
            mean_img_path = os.path.join(output_dir, f"group_meanacc_{mode}_cue-{cue}_classes-{n_cls}.nii.gz")
            t_img.to_filename(t_img_path)
            mean_img.to_filename(mean_img_path)
            print(f"✅ Saved:\n  {t_img_path}\n  {mean_img_path}")
