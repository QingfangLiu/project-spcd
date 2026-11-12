# spm_value_n30_second_image_signed_difference: second-image-only signed value difference RSA group model

Second-level SPM one-sample t-test (df = 29) of the searchlight value RSA beta
maps from the **second-image-only** value analysis, value model
**`signed_difference`**. Settings: premeal + postmeal combined (`--session both`),
all cues (`subset-all`), run-wise centered patterns (`--center_within_run`).
Contrast: one-sample t contrast `[1]`, odor_value > 0.

This is a control/alternative analysis for the manuscript's value RSA
(`../spm_value_n30/`). It is not reported in the manuscript. The three
second-image value models (`cue_value`, `signed_difference`,
`absolute_difference`) were fitted in **separate** regressions; a significant
effect in one and not another is not a test of their difference.

## Model

- Samples: one beta per probe trial, the second cue presentation
  (`img_index == 2`); 144 samples per participant (2 phases x 3 runs x 24).
- Trial variable: V2 - V1, the rating of the outcome predicted by the second cue minus that predicted by the first cue in the same trial.
  V1 and V2 use the participant's ratings from the trial's phase, z-scored
  jointly across the six phase-by-outcome ratings before expansion to trials.
- Value RDM: `abs(variable_i - variable_j)` between trials.
- RSA regression (OLS with intercept, between-run pairs including
  cross-phase pairs, 1 - Pearson neural distance): odor_value + cue_type + image_identity + button_presence + button_side.
  `image_identity` is second-cue identity. Button presence is dropped for a
  participant when every retained trial has a response. The phase RDM is not
  regressed out.

## How it differs from `spm_value_n30`

- Pipeline: same package (`scripts_MRI/rsa_imagewise/`) run with
  `--image_scope second --value_model signed_difference`.
- Samples: second cue presentations only, instead of both cue presentations.
- Value definition: a different trial variable: the signed change in predicted value from the first to the second cue.

## Participants (n = 30)

sub-01, 02, 03, 04, 07, 09, 10, 11, 13, 14, 15, 16, 17, 20, 22, 23, 25, 26, 27,
28, 29, 30, 32, 34, 35, 37, 38, 39, 40, 41.
Excluded: sub-19, sub-31, sub-33 (paper exclusions) and sub-36 (no ratings).
This is the same participant set as `../spm_value_n30/`.

## Provenance

- Subject-level maps:
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/signed_difference/both/sub-XX_model-odor_value_subset-all.nii.gz`,
  run on Biowulf on 2026-11-12 from `/data/liuq13/SPCD_mini` with
  `python -m rsa_imagewise.run_rsa --session both --image_scope second
  --value_model signed_difference --cue_subset all --target odor_value --center_within_run
  --ratings_file compiled_pleasantness_ratings.csv` (code from commit
  `acfdd8c`). Per-subject design diagnostics are in
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/signed_difference/both/diagnostics/`.
- The group model was built on 2026-11-12 with
  `scripts_MRI/rsa_imagewise/aggregate_rsa_group.m`
  (`analysis_mode = 'second_image_value'`,
  `skip_subs = {sub-19, sub-31, sub-33}`), 6 mm FWHM smoothing, and the
  gray-matter mask `masks/gm_0.1.nii`.
- Copied from
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_second_image/signed_difference/both/group_spm/model-odor_value/subset-all/`.
  The persistent smoothed inputs (`spm_inputs/`) were not copied.

## Task for Codex

1. Rerun the same peak-level small-volume correction used for
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_v1/` (medial and
   lateral OFC K2 masks; see its `analysis_metadata.json`). Use this folder's
   `SPM.mat` as the source group model.
2. Write the results to
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_second_image_signed_difference_v1/`.
3. Do not change the manuscript or supplement based on these results without
   asking first.

## Check values

- Whole-brain peak t in `spmT_0001.nii`: 3.837 at MNI [4, 52, -24] (uncorrected).
- `mask.nii` is identical to `../spm_value_n30/mask.nii`
  (SHA-256 `4913f267...c970`).
- SHA-256 `SPM.mat`: `cdc769e6f10f01aa5de504f95bbf78f2ba2aa57dc70f990e2a613449bf4d67b5`
- SHA-256 `spmT_0001.nii`: `457f3bbf136cef9566bf8a665d1dd815d302674332574a6223aa77289fec0418`
- A numpy recomputation of the one-sample t from the smoothed inputs matched
  `spmT_0001.nii` to within 5e-7.

## Notes

- `SPM.mat` stores absolute paths of the smoothed input maps on the original
  machine, which are not in this repo. Analyses that only need `spmT`, `RPV`,
  `ResMS`, and `mask` are unaffected.
- The conditioned-only and preconditioned-only versions of this model are in
  `../spm_value_n30_second_image_signed_difference_conditioned_only/` and
  `../spm_value_n30_second_image_signed_difference_preconditioned_only/`.
- The other second-image value models are in
  `../spm_value_n30_second_image_cue_value/`, `../spm_value_n30_second_image_absolute_difference/`.
