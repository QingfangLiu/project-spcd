# aggregate_searchlight_group_level.py
#
# Group-level smoothing + stats for cue reactivation maps.
#
# Expects files like:
#   <base_dir>/<subfolder>/
#       sub-01_session-premeal_mode-cond2precond_classes-6_radius-8mm.nii.gz
#       ...
#
# Output:
#   group_tmap_cond2precond_classes-6_session-<tag>.nii.gz
#   group_meanacc_cond2precond_classes-6_session-<tag>.nii.gz

import os
import glob
import numpy as np
from nilearn.image import concat_imgs, smooth_img, new_img_like, load_img
from scipy.stats import ttest_1samp

# ---------- Settings (EDIT THESE) ----------
# Root that contains e.g.:
#   <base_dir>/cue_reactivation/sub-01_session-premeal_mode-cond2precond...
base_dir    = "/Users/liuq13/SPCD_data/searchlight_maps"
subfolder   = "cue_reactivation"

session_tag = "postmeal"   # must match your filename
radius_str  = "8mm"       # must match your filename
fwhm        = 6           # smoothing FWHM in mm (set 0 or None for no smoothing)
chance      = 1.0 / 6.0   # 6-way classification

# Path to the SAME mask you used for SearchLight
mask_img_path = "/Users/liuq13/SPCD_data/masks/gm_0.1.nii"

# Filters to avoid crazy t-values from boring/artefact voxels
eps_std  = 1e-4   # minimum across-subject SD to keep a voxel
eps_mean = 1e-4   # minimum |mean - chance| to keep a voxel

# ------------------------------------------
searchlight_dir = os.path.join(base_dir, subfolder)

print(f"\n>>> Group analysis | mode=cond2precond | classes=6 | session={session_tag}")

# Where to save results
output_dir = os.path.join(searchlight_dir, "group_stats")
os.makedirs(output_dir, exist_ok=True)

# --------- Find subject maps ----------
pattern = os.path.join(
    searchlight_dir,
    f"sub-*_session-{session_tag}_mode-cond2precond_classes-6_radius-{radius_str}.nii.gz"
)
map_files = sorted(glob.glob(pattern))
print(f"Found {len(map_files)} searchlight maps")
if len(map_files) == 0:
    raise RuntimeError(f"No maps found matching pattern:\n  {pattern}")

# --------- Load & (optionally) smooth ----------
img_4d = concat_imgs(map_files)       # shape: X×Y×Z×Nsub
if fwhm and fwhm > 0:
    img_4d = smooth_img(img_4d, fwhm=fwhm)

data_4d = img_4d.get_fdata()
n_voxels   = np.prod(data_4d.shape[:-1])
n_subjects = data_4d.shape[-1]
data_2d    = data_4d.reshape(n_voxels, n_subjects)

# --------- Restrict to true brain mask ----------
mask_img  = load_img(mask_img_path)
mask_data = mask_img.get_fdata() > 0

if mask_data.shape != data_4d.shape[:-1]:
    raise ValueError(
        f"Mask shape {mask_data.shape} does not match data shape {data_4d.shape[:-1]}"
    )

mask_flat = mask_data.reshape(-1)

# Remove voxels with any NaN/Inf across subjects
bad = np.any(~np.isfinite(data_2d), axis=1)

# Valid voxels = inside mask AND finite
valid_mask = (~bad) & mask_flat
valid_data = data_2d[valid_mask, :]

print(f"Voxels inside mask & finite: {valid_mask.sum()} / {n_voxels}")
print(f"Subjects in analysis: {n_subjects}")

if valid_data.size == 0:
    raise RuntimeError("No valid voxels after masking and NaN filtering.")

# --------- Additional filters: tiny variance or mean ~ chance ----------
mean_acc = valid_data.mean(axis=1)
std_acc  = valid_data.std(axis=1, ddof=1)

keep = (std_acc > eps_std) & (np.abs(mean_acc - chance) > eps_mean)

if not np.any(keep):
    raise RuntimeError("All voxels were filtered out by variance/mean thresholds.")

valid_data_filt = valid_data[keep, :]

# Build final voxel mask (in full-brain flat space)
valid_indices = np.where(valid_mask)[0]
final_mask = np.zeros_like(valid_mask, dtype=bool)
final_mask[valid_indices[keep]] = True

print(f"Voxels kept after variance/mean filters: {final_mask.sum()}")

# --------- One-sample t-test vs chance ----------
# Center accuracies at chance, then test mean > 0 (two-sided test on centered data).
t_vals, _ = ttest_1samp(valid_data_filt - chance, popmean=0, axis=1, nan_policy="omit")

# --------- Repack into NIfTI images ----------
t_map = np.full(n_voxels, np.nan, dtype=float)
t_map[final_mask] = t_vals
t_img = new_img_like(img_4d.slicer[..., 0], t_map.reshape(data_4d.shape[:-1]))

mean_vals = np.full(n_voxels, np.nan, dtype=float)
mean_vals[final_mask] = valid_data_filt.mean(axis=1)
mean_img = new_img_like(img_4d.slicer[..., 0], mean_vals.reshape(data_4d.shape[:-1]))

# --------- Save ----------
t_out    = os.path.join(output_dir, f"group_tmap_cond2precond_classes-6_session-{session_tag}.nii.gz")
mean_out = os.path.join(output_dir, f"group_meanacc_cond2precond_classes-6_session-{session_tag}.nii.gz")

t_img.to_filename(t_out)
mean_img.to_filename(mean_out)

print(f"✅ Saved:\n  {t_out}\n  {mean_out}")
