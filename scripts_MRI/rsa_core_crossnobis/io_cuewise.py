


import os
import numpy as np
import pandas as pd
from nilearn.image import load_img, new_img_like

def load_cuewise_maps_per_run(
	sub: str,
	base_dir: str,
	session_folder: str,
	session_tag: str,
	mode: str = "both", # "both", "img1only", "img2only"
	stat_type: str = "beta", # "beta" or "tmap"
	use_resms_whitening=False,
	):

	"""
load_cuewise_maps_per_run

Loads condition-wise GLM maps for each run.

Returns:
    dict[(run, condition_label)] -> NiftiImage

Supports:
- beta or tmap input
- optional diagonal ResMS whitening per run
- missing runs allowed
"""

	sub_mode_dir = os.path.join(base_dir, "GLM", session_folder, sub, mode)

	if not os.path.isdir(sub_mode_dir):
		raise FileNotFoundError(f"Subject/mode folder not found: {sub_mode_dir}")

	if stat_type not in ("beta", "tmap"):
		raise ValueError(f"stat_type must be 'beta' or 'tmap', got: {stat_type}")

	file_col = "beta_file" if stat_type == "beta" else "tmap_file"

	# mapping between cue_index to task conditions
	cue_to_condition = {
		1: "A11",
		2: "B11",
		3: "A21",
		4: "B21",
		5: "C1",
		6: "D1",
		7: "A12",
		8: "B12",
		9: "A22",
		10: "B22",
		11: "C2",
		12: "D2",
	}

	runs = (1, 2, 3) # task has 3 runs

	# indexed by a tuple 
	run_condition_imgs = {} # (run, condition) -> NiftiImage

	for run in runs:

		csv_path = os.path.join(
			sub_mode_dir,
			f"{sub}_task-{session_tag}_mode-{mode}_run-{run:02d}_index.csv"
			)

		if not os.path.exists(csv_path):
			# missing run is allowed
			print(f"[load_cuewise_maps_per_run] Missing CSV for run {run:02d}: {csv_path} (skipping run)")
			continue

		df = pd.read_csv(csv_path)

		# ensure existence of columns used next
		required = {"cue_index", file_col}
		missing = required - set(df.columns) # set minus. A - B: all elements in A that are NOT in B
		if missing:
			raise ValueError(f"Index CSV missing columns {sorted(missing)}: {csv_path}")

		# optional ResMS whitening
		# diagonal whitening only, not accounting for voxel-to-voxel covariance
		# use ResMS.nii in SPM folder to do whitening
		scale_data = None # initialize scale_data
		if use_resms_whitening:
			resms_path = os.path.join(sub_mode_dir, f"run-{run:02d}_spm", "ResMS.nii")
			whitening_eps = 1e-6

			if os.path.exists(resms_path):
				resms_img = load_img(resms_path)
				resms_data = resms_img.get_fdata()
				scale_data = 1.0 / np.sqrt(resms_data + whitening_eps)
			else:
				print(f"[whitening fallback] ResMS not found for run {run}, proceeding without whitening.")

		# load each cue_index (1...12) in this run
		for _, row in df.iterrows():

			cue_idx = int(row["cue_index"]) # 1...12

			if cue_idx not in cue_to_condition:
				continue
			
			cond = cue_to_condition[cue_idx] # condition label (e.g. "A11")
			fname = str(row[file_col]) # file name of beta or t map
			fpath = os.path.join(sub_mode_dir, fname)

			if not os.path.exists(fpath):
				raise FileNotFoundError(f"Map file not found: {fpath}")

			key = (run, cond) # key is a tuple, used as dictionary keys
			if key in run_condition_imgs:
				raise ValueError(f"Duplicate entry for run={run}, condition={cond} in {csv_path}")

			cond_img = load_img(fpath)
			# optional: do whitening after loading the image
			if use_resms_whitening and scale_data is not None:
				cond_data = cond_img.get_fdata()
				whitened_data = cond_data * scale_data # core whitening step
				cond_img = new_img_like(cond_img, whitened_data) # replace raw image with whitened image

			run_condition_imgs[key] = cond_img  # store the image (whitened or unwhitened)

	return run_condition_imgs # return a dictionary of images, indexed by run and condition



