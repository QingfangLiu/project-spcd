# aggregate_searchlight_group_level_crosstask.py
import os
import glob
import numpy as np
from nilearn.image import concat_imgs, smooth_img, new_img_like
from scipy.stats import ttest_1samp

# ---------- Settings (edit these as needed) ----------
# Root that contains: <base_dir>/searchlight_maps/probe_crosstask/...
base_dir    = "/Users/liuq13/SPCD_data/searchlight_maps"
subfolder   = "probe_crosstask"

# Aggregate both cue types and class schemes
cue_types   = ["A", "B"]        # cue type for PROBE test data
class_list  = [2, 3]

# Session tag must match what you used when saving maps (e.g., "premeal" or "postmeal")
session_tag = "premeal"

# Searchlight radius string must match filename convention (e.g., "8mm")
radius_str  = "8mm"

# Smoothing (in mm FWHM) applied before group stats
fwhm        = 6

# ----------------------------------------------------
searchlight_dir = os.path.join(base_dir, subfolder)

def _chance(n_cls: int) -> float:
    return 0.5 if n_cls == 2 else (1.0 / 3.0)

for n_cls in class_list:
    for cue in cue_types:
        mode_tag = "crosstask"   # filenames have 'mode-crosstask'
        print(f"\n>>> Group analysis | classes={n_cls} | mode={mode_tag} | cue={cue} | session={session_tag}")

        # Output folder mirrors the per-combo leaf in your map tree
        output_dir = os.path.join(
            searchlight_dir,
            f"mode-{mode_tag}_cue-{cue}_classes-{n_cls}",
            "group_stats"
        )
        os.makedirs(output_dir, exist_ok=True)

        # --------- Find subject maps ----------
        pattern = os.path.join(
            searchlight_dir,
            f"sub-*_session-{session_tag}_mode-{mode_tag}_cue-{cue}_classes-{n_cls}_radius-{radius_str}.nii.gz"
        )
        map_files = sorted(glob.glob(pattern))
        print(f"Found {len(map_files)} searchlight maps")
        if len(map_files) == 0:
            print("No maps found. Skipping.")
            continue

        # --------- Load & smooth ----------
        img_4d = concat_imgs(map_files)          # shape: X×Y×Z×Nsub
        img_4d = smooth_img(img_4d, fwhm=fwhm)   # optional smoothing
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
        chance = _chance(n_cls)
        # center accuracies at chance, then test mean>0 with two-sided t (you can convert to one-sided if needed)
        t_vals, _ = ttest_1samp(valid_data - chance, popmean=0, axis=1, nan_policy="omit")

        # --------- Repack to NIfTI ----------
        t_map = np.full(n_voxels, np.nan, dtype=float)
        t_map[valid_mask] = t_vals
        t_img = new_img_like(img_4d.slicer[..., 0], t_map.reshape(data_4d.shape[:-1]))

        mean_vals = np.full(n_voxels, np.nan, dtype=float)
        mean_vals[valid_mask] = valid_data.mean(axis=1)
        mean_img = new_img_like(img_4d.slicer[..., 0], mean_vals.reshape(data_4d.shape[:-1]))

        # --------- Save ----------
        t_out   = os.path.join(output_dir, f"group_tmap_{mode_tag}_cue-{cue}_classes-{n_cls}_session-{session_tag}.nii.gz")
        mean_out= os.path.join(output_dir, f"group_meanacc_{mode_tag}_cue-{cue}_classes-{n_cls}_session-{session_tag}.nii.gz")
        t_img.to_filename(t_out)
        mean_img.to_filename(mean_out)
        print(f"✅ Saved:\n  {t_out}\n  {mean_out}")
