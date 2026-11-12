# spm_value_n30_second_image_absolute_difference_preconditioned_only: second-image-only absolute value difference RSA group model

Second-level SPM one-sample t-test (df = 29) of the searchlight value RSA beta
maps from the **second-image-only** value analysis, value model
**`absolute_difference`**. Settings: premeal + postmeal combined (`--session both`),
**preconditioned cues only** (`subset-preconditioned_only`), run-wise centered patterns (`--center_within_run`).
Contrast: one-sample t contrast `[1]`, odor_value > 0.

This is the preconditioned-cue counterpart of `../spm_value_n30_second_image_absolute_difference/`. The
participants (n = 30), exclusions, value definition, smoothing, mask, and
pipeline are the same; see `../spm_value_n30_second_image_absolute_difference/README.md` for details. Because the
subset contains a single cue type, `cue_type` has no variance and is dropped
automatically, so the RSA regression is odor_value + image_identity + button_presence + button_side.

## Provenance

- Subject-level maps:
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/absolute_difference/both/sub-XX_model-odor_value_subset-preconditioned_only.nii.gz`,
  run on Biowulf on 2026-11-12 from `/data/liuq13/SPCD_mini` with
  `python -m rsa_imagewise.run_rsa --session both --image_scope second
  --value_model absolute_difference --cue_subset preconditioned_only --target odor_value --center_within_run
  --ratings_file compiled_pleasantness_ratings.csv` (code from commit
  `acfdd8c`). Per-subject design diagnostics are in
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/absolute_difference/both/diagnostics/`.
- The group model was built on 2026-11-12 with
  `scripts_MRI/rsa_imagewise/aggregate_rsa_group.m`
  (`analysis_mode = 'second_image_value'`,
  `skip_subs = {sub-19, sub-31, sub-33}`), 6 mm FWHM smoothing, and the
  gray-matter mask `masks/gm_0.1.nii`.
- Copied from
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/absolute_difference/both/group_spm/model-odor_value/subset-preconditioned_only/`.
  The persistent smoothed inputs (`spm_inputs/`) were not copied.

## Task for Codex

1. Rerun the same peak-level small-volume correction used for
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_v1/` (medial and
   lateral OFC K2 masks; see its `analysis_metadata.json`). Use this folder's
   `SPM.mat` as the source group model.
2. Write the results to
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_second_image_absolute_difference_preconditioned_only_v1/`.
3. Do not change the manuscript or supplement based on these results without
   asking first.

## Check values

- Whole-brain peak t in `spmT_0001.nii`: 3.942 at MNI [-20, -6, -26] (uncorrected).
- `mask.nii` is identical to `../spm_value_n30/mask.nii`
  (SHA-256 `4913f267...c970`).
- SHA-256 `SPM.mat`: `90dba28acacb6d8bf92277a1007dd9823265d7a8e1d7f7bc4561b3581e54dd29`
- SHA-256 `spmT_0001.nii`: `afd2b8b5bfbca1aa3b0b58332367f865cf6315a974f9b7294be75571b6ef8330`
- A numpy recomputation of the one-sample t from the smoothed inputs matched
  `spmT_0001.nii` to within 5e-7.

## Notes

- `SPM.mat` stores absolute paths of the smoothed input maps on the original
  machine, which are not in this repo. Analyses that only need `spmT`, `RPV`,
  `ResMS`, and `mask` are unaffected.
