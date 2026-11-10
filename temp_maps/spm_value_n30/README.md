# spm_value_n30: corrected combined value RSA group model

Second-level SPM one-sample t-test (df = 29) of the searchlight odor-value RSA
beta maps, premeal + postmeal combined (`--session both`), all cues
(`subset-all`), run-wise centered patterns (`--center_within_run`).
Contrast: `RSA both | subset-all | model-odor_value > 0`.

## Why this folder exists

`temp_maps/spm_value/` also has n = 30, but the wrong participants: it includes
sub-31 (a paper exclusion) and lacks sub-10 (his map was missing). The existing
SVC results in `rsa_roi/probe_imagewise_value/searchlight_svc_existing_group_v1/`
were computed from that old model, so the manuscript values built on them
(medial OFC peak t = 4.415, p_FWE_SVC = 0.032, critical t = 4.218) are stale.

## Participants (n = 30)

sub-01, 02, 03, 04, 07, 09, 10, 11, 13, 14, 15, 16, 17, 20, 22, 23, 25, 26, 27,
28, 29, 30, 32, 34, 35, 37, 38, 39, 40, 41.
Excluded: sub-19, sub-31, sub-33 (paper exclusions) and sub-36 (no ratings).

## Task for Codex

Rerun the same peak-level small-volume correction used for
`searchlight_svc_existing_group_v1` (vmPFC, medial OFC, lateral OFC), but with
`temp_maps/spm_value_n30/SPM.mat` as the source group model. Write the results to
a new results folder under `rsa_roi/probe_imagewise_value/`. The earlier SVC
script is not in the repo; please commit it.

After that, the manuscript and supplement values need updating, and
`scripts_MRI/RSA_plot/make_ofc_value_supplement_figure.py` needs to point at this
folder and the new SVC CSV (it asserts that peak t matches the CSV).

## Check values

- Peak t in `spmT_0001.nii`: 4.497 at MNI [14, 46, -6] (old model: 4.415 at the
  same location).
- `mask.nii` is identical to the old model's (SHA-256 `4913f267...c70`).
- SHA-256 `SPM.mat`: `b84678b6f3792b6900702eddadc1d8f9a6b7202d411a1e79ba32567f9f6159cb`
- SHA-256 `spmT_0001.nii`: `bb530c4c0e6f9f87fd15e948d3c37e7d1d4f60cd0898d1b6fce9dbb76362d9bc`

## Notes

- Built 2026-11-10 with `scripts_MRI/rsa_imagewise/aggregate_rsa_group.m`
  (`skip_subs = {sub-19, sub-31, sub-33}`), 6 mm FWHM smoothing, gray-matter mask
  `masks/gm_0.1.nii`.
- `SPM.mat` stores absolute paths of the smoothed input maps (original machine),
  which are not in this repo. Analyses that only need `spmT`, `RPV`, `ResMS`, and
  `mask` are unaffected.
- The conditioned-only and preconditioned-only group models are not included here.
