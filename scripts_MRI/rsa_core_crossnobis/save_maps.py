
"""
save_rsa_maps

Saves RSA searchlight maps with informative filenames including:
- subject
- mode
- neural metric
- rsa method
- model name
- subset
- odor mask usage
- radius
- whitening flag
"""

import os
def save_rsa_maps(
	sub: str,
	rsa_maps: dict,
	output_dir: str,
	radius: float,
	cue_subset: str,
	mode: str,
	rsa_method: str,
    neural_metric: str,
	use_odor_mask: bool,
	use_resms_whitening: bool,
	):

	os.makedirs(output_dir, exist_ok=True)

	subset_label = f"subset-{cue_subset}"
	odor_mask_label = f"odor_mask-{int(bool(use_odor_mask))}"
	radius_label = f"radius-{radius}mm"
	whiten_label = f"whiten-{int(bool(use_resms_whitening))}"

	for model_name, rsa_map in rsa_maps.items():

		name_parts = [
		f"{sub}",
		f"mode-{mode}",
		f"{neural_metric}",
		f"{rsa_method}",
		f"model-{model_name}",
		]

		if subset_label is not None:
			name_parts.append(subset_label)

		name_parts.extend([odor_mask_label, radius_label, whiten_label])

		output_file = os.path.join(output_dir, "_".join(name_parts) + ".nii.gz")

		rsa_map.to_filename(output_file)
		print(f"Saved: {output_file}")









