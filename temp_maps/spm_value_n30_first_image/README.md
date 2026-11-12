# spm_value_n30_first_image: first-image-only combined value RSA group model

Second-level SPM one-sample t-test (df = 29) of the searchlight odor-value RSA
beta maps from the **first-image-only** variant of the value analysis.
Settings: premeal + postmeal combined (`--session both`), all cues
(`subset-all`), run-wise centered patterns (`--center_within_run`).
Contrast: one-sample t contrast `[1]`, odor_value > 0.

This is a control analysis for the manuscript's value RSA
(`../spm_value_n30/`). The only difference is that it uses one beta per probe
trial, the first cue presentation (`img_index == 1`), instead of both cue
presentations. It is not reported in the manuscript.

## How it differs from `spm_value_n30`

- Pipeline: `scripts_MRI/rsa_imagewise_first_image/` instead of
  `scripts_MRI/rsa_imagewise/`. Apart from the two points below, the code is
  the same: ratings, run-wise centering, between-run pairs including
  cross-phase pairs, 1 − Pearson neural distance, and OLS with intercept.
- Samples: first cue presentations only.
- RSA regression: odor_value + cue_type + image_identity. The button-presence
  and button-side RDMs are omitted. Under the manuscript's
  `second_image_only` button scope, every first-cue sample is labeled "no
  response", so those RDMs carry no information.

## Participants (n = 30)

sub-01, 02, 03, 04, 07, 09, 10, 11, 13, 14, 15, 16, 17, 20, 22, 23, 25, 26, 27,
28, 29, 30, 32, 34, 35, 37, 38, 39, 40, 41.
Excluded: sub-19, sub-31, sub-33 (paper exclusions) and sub-36 (no ratings).
This is the same participant set as `../spm_value_n30/`.

## Provenance

- Subject-level maps were run on Biowulf on 2026-11-12 with
  `python -m rsa_imagewise_first_image.run_rsa --session both --cue_subset all
  --target odor_value --center_within_run --ratings_file
  compiled_pleasantness_ratings.csv`. The maps are stored at
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_first_image/both/sub-XX_model-odor_value_subset-all.nii.gz`.
- The group model was built on 2026-11-12 with
  `scripts_MRI/rsa_imagewise_first_image/aggregate_rsa_group.m`
  (`skip_subs = {sub-19, sub-31, sub-33}`), 6 mm FWHM smoothing, and the
  gray-matter mask `masks/gm_0.1.nii`.
- Copied from
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_first_image/both/group_spm/model-odor_value/subset-all/`.
  The persistent smoothed inputs (`spm_inputs/`) were not copied.

## Task for Codex

1. Rerun the same peak-level small-volume correction used for
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_v1/` (medial and
   lateral OFC K2 masks; see its `analysis_metadata.json`). Use
   `temp_maps/spm_value_n30_first_image/SPM.mat` as the source group model.
2. Write the results to
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_first_image_v1/`.
3. Do not change the manuscript or supplement based on these results without
   asking first.

## Check values

- Whole-brain peak t in `spmT_0001.nii`: 3.781 at MNI [24, -72, -50].
- `mask.nii` is identical to `../spm_value_n30/mask.nii`
  (SHA-256 `4913f267...c970`).
- SHA-256 `SPM.mat`: `cbf92e98380b424b1ab29cd285e386ab0046514c1d748a5656f051fe0859caa5`
- SHA-256 `spmT_0001.nii`: `23b1b8ae629c50cbee8d2c8e586210b0b75013f5150439f89a0f4d6f0f33d6e5`

### Independent cross-check (Python preview, not the SVC result of record)

A numpy re-implementation of the SPM12 one-sample model and the peak-level
SVC produced the values below. Run on `../spm_value_n30/`, the same
re-implementation exactly reproduces `searchlight_svc_n30_ofc_v1`
(medial OFC t = 4.497, p_FWE = 0.0269, critical t = 4.217). Its t map matches
this folder's `spmT_0001.nii` to within 3e-7.

Estimated smoothness: FWHM = [8.218, 7.906, 7.847] voxels.

| ROI | Peak T | MNI | p_FWE_SVC | p_unc | Critical T | Voxels > T(0.001) |
|---|---|---|---|---|---|---|
| Medial OFC | 2.540 | [8, 46, -14] | 0.748 | 0.0084 | 4.196 | 0 |
| Lateral OFC | 3.335 | [44, 42, -2] | 0.239 | 0.0012 | 4.119 | 0 |

## Notes

- `SPM.mat` stores absolute paths of the smoothed input maps on the original
  machine, which are not in this repo. Analyses that only need `spmT`, `RPV`,
  `ResMS`, and `mask` are unaffected.
- The conditioned-only and preconditioned-only first-image models are in
  `../spm_value_n30_first_image_conditioned_only/` and
  `../spm_value_n30_first_image_preconditioned_only/`.
