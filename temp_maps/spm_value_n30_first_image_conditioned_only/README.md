# spm_value_n30_first_image_conditioned_only

Second-level SPM one-sample t-test (df = 29) of the first-image-only searchlight
odor-value RSA beta maps. Settings: premeal + postmeal combined
(`--session both`), **conditioned cues only** (`subset-conditioned_only`),
run-wise centered patterns. Contrast: one-sample t contrast `[1]`,
odor_value > 0.

This is the first-image-only counterpart of
`../spm_value_n30_conditioned_only/`. The participants (n = 30), exclusions,
smoothing, mask, and pipeline are the same as in
`../spm_value_n30_first_image/README.md`; see that file for details. Because
the subset contains a single cue type, `cue_type` has no variance and is
dropped automatically, so the RSA regression is odor_value + image_identity.

## Provenance

- Subject-level maps:
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_first_image/both/sub-XX_model-odor_value_subset-conditioned_only.nii.gz`,
  run with `--cue_subset conditioned_only` and otherwise the same settings as
  `../spm_value_n30_first_image/`.
- Copied from
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_first_image/both/group_spm/model-odor_value/subset-conditioned_only/`.
  The persistent smoothed inputs (`spm_inputs/`) were not copied.

## Task for Codex

1. Rerun the same peak-level small-volume correction used for
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_v1/` (medial and
   lateral OFC K2 masks). Use this folder's `SPM.mat` as the source group
   model.
2. Write the results to
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_first_image_conditioned_only_v1/`.
3. Do not change the manuscript or supplement based on these results without
   asking first.

## Check values

- Whole-brain peak t in `spmT_0001.nii`: 3.914 at MNI [56, -60, -36].
- `mask.nii` is identical to `../spm_value_n30/mask.nii`
  (SHA-256 `4913f267...c970`).
- SHA-256 `SPM.mat`: `f0aeaae357a255329d63c442a8a2c56e29b1b9247417bf9943f1a22bc8db7dd2`
- SHA-256 `spmT_0001.nii`: `8abe9d3749c3be02eb1b3325f4c2fbabc2df22e6f0b17b30eda284bfd5dbf4cb`

### Independent cross-check (Python preview, not the SVC result of record)

These values come from the same numpy re-implementation described in
`../spm_value_n30_first_image/README.md`. Its t map matches this folder's
`spmT_0001.nii` to within 5e-7.

Estimated smoothness: FWHM = [8.000, 7.701, 7.665] voxels.

| ROI | Peak T | MNI | p_FWE_SVC | p_unc | Critical T | Voxels > T(0.001) |
|---|---|---|---|---|---|---|
| Medial OFC | 2.521 | [-8, 36, -10] | 0.775 | 0.0087 | 4.220 | 0 |
| Lateral OFC | 2.779 | [-32, 64, -10] | 0.562 | 0.0047 | 4.143 | 0 |

## Notes

- `SPM.mat` stores absolute paths of the smoothed input maps on the original
  machine, which are not in this repo. Analyses that only need `spmT`, `RPV`,
  `ResMS`, and `mask` are unaffected.
