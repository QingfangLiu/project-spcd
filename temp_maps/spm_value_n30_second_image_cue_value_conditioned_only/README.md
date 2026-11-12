# spm_value_n30_second_image_cue_value_conditioned_only: second-image-only cue value RSA group model

Second-level SPM one-sample t-test (df = 29) of the searchlight value RSA beta
maps from the **second-image-only** value analysis, value model
**`cue_value`**. Settings: premeal + postmeal combined (`--session both`),
**conditioned cues only** (`subset-conditioned_only`), run-wise centered patterns (`--center_within_run`).
Contrast: one-sample t contrast `[1]`, odor_value > 0.

This is the conditioned-cue counterpart of `../spm_value_n30_second_image_cue_value/`. The
participants (n = 30), exclusions, value definition, smoothing, mask, and
pipeline are the same; see `../spm_value_n30_second_image_cue_value/README.md` for details. Because the
subset contains a single cue type, `cue_type` has no variance and is dropped
automatically, so the RSA regression is odor_value + image_identity + button_presence + button_side.

## Provenance

- Subject-level maps:
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/cue_value/both/sub-XX_model-odor_value_subset-conditioned_only.nii.gz`,
  run on Biowulf on 2026-11-12 from `/data/liuq13/SPCD_mini` with
  `python -m rsa_imagewise.run_rsa --session both --image_scope second
  --value_model cue_value --cue_subset conditioned_only --target odor_value --center_within_run
  --ratings_file compiled_pleasantness_ratings.csv` (code from commit
  `acfdd8c`). Per-subject design diagnostics are in
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/cue_value/both/diagnostics/`.
- The group model was built on 2026-11-12 with
  `scripts_MRI/rsa_imagewise/aggregate_rsa_group.m`
  (`analysis_mode = 'second_image_value'`,
  `skip_subs = {sub-19, sub-31, sub-33}`), 6 mm FWHM smoothing, and the
  gray-matter mask `masks/gm_0.1.nii`.
- Copied from
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/cue_value/both/group_spm/model-odor_value/subset-conditioned_only/`.
  The persistent smoothed inputs (`spm_inputs/`) were not copied.

## Task for Codex

1. Rerun the same peak-level small-volume correction used for
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_v1/` (medial and
   lateral OFC K2 masks; see its `analysis_metadata.json`). Use this folder's
   `SPM.mat` as the source group model.
2. Write the results to
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_second_image_cue_value_conditioned_only_v1/`.
3. Do not change the manuscript or supplement based on these results without
   asking first.

## Check values

- Whole-brain peak t in `spmT_0001.nii`: 4.220 at MNI [-32, -6, 18] (uncorrected).
- `mask.nii` is identical to `../spm_value_n30/mask.nii`
  (SHA-256 `4913f267...c970`).
- SHA-256 `SPM.mat`: `84303ae78c1e7424da7b8d298b541a089d997dd4a6e46e6a1db678290e1eb0ad`
- SHA-256 `spmT_0001.nii`: `ed63524101a8881bb68429a4c38c6178aa5403fc0ea3034f84bb647417938692`
- A numpy recomputation of the one-sample t from the smoothed inputs matched
  `spmT_0001.nii` to within 5e-7.

## Notes

- `SPM.mat` stores absolute paths of the smoothed input maps on the original
  machine, which are not in this repo. Analyses that only need `spmT`, `RPV`,
  `ResMS`, and `mask` are unaffected.
