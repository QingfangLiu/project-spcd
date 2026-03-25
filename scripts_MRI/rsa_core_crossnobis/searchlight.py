# searchlight.py

import os
import numpy as np

from nilearn.image import concat_imgs, new_img_like

from .io_cuewise import load_cuewise_maps_per_run
from .estimator import SingleModelRSA
from .odor_masking import build_odor_mask

canonical_condition_order = ["A11", "A21", "C1", "B11", "B21", "D1", "A12", "A22", "C2", "B12", "B22", "D2"]

cue_subset_to_conditions = {
	"all": canonical_condition_order,
	"conditioned": ["B11", "B21", "D1", "B12", "B22", "D2"],
	"preconditioned": ["A11", "A21", "C1", "A12", "A22", "C2"],
}

def _slice_square_by_conditions(
	square: np.ndarray, 
	from_order: list, 
	to_order: list,
	) -> np.ndarray: # a type hint (expected to return a NumPy array)
	if square.shape[0] != square.shape[1]:
		raise ValueError("Expected a square matrix to slice.")
	if square.shape[0] != len(from_order):
		raise ValueError(f"Square size {square.shape} does not match from_order length {len(from_order)}.")
	idx_map = {c: i for i, c in enumerate(from_order)}
	idx = [idx_map[c] for c in to_order]
	return square[np.ix_(idx, idx)]

def searchlight_rsa_single_subject(
	sub: str,
	base_dir: str,
	session_folder: str,
	session_tag: str,
	model_rdms: dict, 			# dictionary of model rdms
	radius: float,
	n_jobs: int,
	rsa_method: str,
    neural_metric: str,
	cue_subset: str, 	
	use_odor_mask: bool,
	mode: str,
	stat_type: str, 			# "beta" or "tmap"
	use_resms_whitening: bool,
	center_within_run: bool,
	):

	from nilearn.decoding import SearchLight
	from nilearn.image import load_img

	run_condition_imgs = load_cuewise_maps_per_run(
		sub=sub,
		base_dir=base_dir,
		session_folder=session_folder,
		session_tag=session_tag,
		mode=mode,
		stat_type=stat_type,
		use_resms_whitening=use_resms_whitening,
	)

	# subset condition order
	condition_order = cue_subset_to_conditions[cue_subset] # get condition order to include in RSA
	print(f"cue_subset={cue_subset}: using {len(condition_order)} conditions: {condition_order}")

	# slice model RDMs to subset
	sliced_model_rdms = {
		name: _slice_square_by_conditions(rdm, from_order=canonical_condition_order, to_order=condition_order)
		for name, rdm in model_rdms.items()
	}

	# get odor-odor mask if requested
	odor_mask = None
	if use_odor_mask:
		odor_mask = build_odor_mask(condition_order)
		print(f"Odor mask: keep {odor_mask.sum()} / {odor_mask.size} lower-triangle pairs (odor-odor only)")

	# build stacked samples: img, run, cond
	cond_to_idx = {c: i for i, c in enumerate(condition_order)} # create a dictionary

	img_list = []
	run_ids = []
	cond_ids = []

	runs = (1, 2, 3)
	used_runs = set() # record which runs can be included in RSA

	for run in runs:
		for cond in condition_order:
			key = (run, cond)
			if key not in run_condition_imgs:
				continue
			img_list.append(run_condition_imgs[key])
			run_ids.append(run)
			cond_ids.append(cond_to_idx[cond]) # condition as integers
			used_runs.add(run)

	if len(used_runs) < 2:
		raise ValueError(f"Cross-run RSA requires at least 2 runs, but only found: {used_runs}")

	n_samples = len(img_list)
	img_4d = concat_imgs(img_list) # 4D nifti with shape (x, y, z, n_samples)

	print("Loading mask file...")
	mask_file = os.path.join(base_dir, "masks", "gm_0.1.nii")

	if not os.path.exists(mask_file):
		raise FileNotFoundError("Mask file not found")

	mask_img = load_img(mask_file)

	print(f"Runing searchlight RSA (radius={radius}mm, n_jobs={n_jobs})...")
	rsa_maps = {}

	cv = [(np.arange(n_samples), np.arange(n_samples))]

	for model_name in sliced_model_rdms.keys():
		print(f"Computing {model_name} map...")

		estimator = SingleModelRSA(
			model_rdms=sliced_model_rdms,
			target_name=model_name,
			neural_metric=neural_metric,
			rsa_method=rsa_method,
			odor_mask=odor_mask,
			run_ids=run_ids,
			cond_ids=cond_ids,
			center_within_run=center_within_run,
		)

		searchlight = SearchLight(
			mask_img=mask_img,
			radius=radius,
			estimator=estimator,
			cv=cv,
			n_jobs=n_jobs,
			verbose=1
		)

		y_dummy = np.arange(n_samples)
		searchlight.fit(img_4d, y_dummy)

		scores_img = new_img_like(mask_img, searchlight.scores_)
		rsa_maps[model_name] = scores_img

	return rsa_maps













		























