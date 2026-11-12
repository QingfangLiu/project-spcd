# spm_value_n30_first_image_preconditioned_only

Second-level SPM one-sample t-test (df = 29) of the first-image-only searchlight
odor-value RSA beta maps. Settings: premeal + postmeal combined
(`--session both`), **preconditioned cues only**
(`subset-preconditioned_only`), run-wise centered patterns. Contrast:
one-sample t contrast `[1]`, odor_value > 0.

This is the first-image-only counterpart of
`../spm_value_n30_preconditioned_only/`. The participants (n = 30),
exclusions, smoothing, mask, and pipeline are the same as in
`../spm_value_n30_first_image/README.md`; see that file for details. Because
the subset contains a single cue type, `cue_type` has no variance and is
dropped automatically, so the RSA regression is odor_value + image_identity.

## Provenance

- Subject-level maps:
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_first_image/both/sub-XX_model-odor_value_subset-preconditioned_only.nii.gz`,
  run with `--cue_subset preconditioned_only` and otherwise the same settings
  as `../spm_value_n30_first_image/`.
- Copied from
  `/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_first_image/both/group_spm/model-odor_value/subset-preconditioned_only/`.
  The persistent smoothed inputs (`spm_inputs/`) were not copied.

## Task for Codex

1. Rerun the same peak-level small-volume correction used for
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_v1/` (medial and
   lateral OFC K2 masks). Use this folder's `SPM.mat` as the source group
   model.
2. Write the results to
   `rsa_roi/probe_imagewise_value/searchlight_svc_n30_ofc_first_image_preconditioned_only_v1/`.
3. Do not change the manuscript or supplement based on these results without
   asking first.

## Check values

- Whole-brain peak t in `spmT_0001.nii`: 3.517 at MNI [42, -74, -28].
- `mask.nii` is identical to `../spm_value_n30/mask.nii`
  (SHA-256 `4913f267...c970`).
- SHA-256 `SPM.mat`: `fef65eaf099b00ab064318ba2a5ca5511289f3592c0f9a7e1f85b9d454d54c2a`
- SHA-256 `spmT_0001.nii`: `1b31266a5fa332ac247fdab2c67080cdbdb23dc744b72c17a76a62afd346e2fd`

### Independent cross-check (Python preview, not the SVC result of record)

These values come from the same numpy re-implementation described in
`../spm_value_n30_first_image/README.md`. Its t map matches this folder's
`spmT_0001.nii` to within 5e-7.

Estimated smoothness: FWHM = [8.304, 7.989, 7.876] voxels.

| ROI | Peak T | MNI | p_FWE_SVC | p_unc | Critical T | Voxels > T(0.001) |
|---|---|---|---|---|---|---|
| Medial OFC | 3.147 | [4, 38, -12] | 0.367 | 0.0019 | 4.187 | 0 |
| Lateral OFC | 2.055 | [50, 40, -2] | 0.912 | 0.0245 | 4.111 | 0 |

## Notes

- `SPM.mat` stores absolute paths of the smoothed input maps on the original
  machine, which are not in this repo. Analyses that only need `spmT`, `RPV`,
  `ResMS`, and `mask` are unaffected.
