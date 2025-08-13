# aggregate_searchlight_group_level_crosscue.py
import os
import glob
import numpy as np
from nilearn.image import concat_imgs, smooth_img, new_img_like
from scipy.stats import ttest_1samp

# ---------- Settings ----------
base_dir   = "/Users/liuq13/SPCD_data/searchlight_maps"
subfolder  = "probe_crosscue"
modes      = ["strict", "set2set"]      # cross-cue variants
cue_fixed  = "AB"                        # file naming uses cue-AB
class_list = [2, 3]
radius_str = "8mm"
fwhm       = 6

searchlight_dir = os.path.join(base_dir, subfolder)

for n_cls in class_list:
    for mode in modes:
        mode_tag = f"crosscue_{mode}"   # filenames have 'mode-crosscue_set2set' / 'mode-crosscue_strict'
        print(f"\n>>> Group analysis | classes={n_cls} | mode={mode_tag} | cue={cue_fixed}")

        # Output folder mirrors the per-combo leaf in your map tree
        output_dir = os.path.join(
            searchlight_dir,
            f"mode-{mode_tag}_cue-{cue_fixed}_classes-{n_cls}",
            "group_stats"
        )
        os.makedirs(output_dir, exist_ok=True)

        # --------- Find subject maps ----------
        pattern = os.path.join(
            searchlight_dir,
            f"sub-*_session-premeal_mode-{mode_tag}_cue-{cue_fixed}_classes-{n_cls}_radius-{radius_str}.nii.gz"
        )
        map_files = sorted(glob.glob(pattern))
        print(f"Found {len(map_files)} searchlight maps")
        if len(map_files) == 0:
            print("No maps found. Skipping.")
            continue

        # --------- Load & smooth ----------
        img_4d = concat_imgs(map_files)
        img_4d = smooth_img(img_4d, fwhm=fwhm)
        data_4d = img_4d.get_fdata()

        # --------- Reshape & clean ----------
        n_voxels   = np.prod(data_4d.shape[:-1])
        n_subjects = data_4d.shape[-1]
        data_2d    = data_4d.reshape(n_voxels, n_subjects)

        # Remove voxels with any NaN/Inf across subjects
        bad = np.any(~np.isfinite(data_2d), axis=1)
        valid_mask = ~bad
        valid_data = data_2d[valid_mask, :]

        # --------- One-sample t-test vs chance ----------
        chance = 0.5 if n_cls == 2 else 1/3
        t_vals, _ = ttest_1samp(valid_data - chance, popmean=0, axis=1, nan_policy="omit")

        # --------- Repack to NIfTI ----------
        t_map = np.full(n_voxels, np.nan, dtype=float)
        t_map[valid_mask] = t_vals
        t_img = new_img_like(img_4d.slicer[..., 0], t_map.reshape(data_4d.shape[:-1]))

        mean_vals = np.full(n_voxels, np.nan, dtype=float)
        mean_vals[valid_mask] = valid_data.mean(axis=1)
        mean_img = new_img_like(img_4d.slicer[..., 0], mean_vals.reshape(data_4d.shape[:-1]))

        # --------- Save ----------
        t_img.to_filename(os.path.join(output_dir, f"group_tmap_{mode_tag}_cue-{cue_fixed}_classes-{n_cls}.nii.gz"))
        mean_img.to_filename(os.path.join(output_dir, f"group_meanacc_{mode_tag}_cue-{cue_fixed}_classes-{n_cls}.nii.gz"))
        print("✅ Saved group t-map and mean-accuracy map.")
