# spm_value_n30_preconditioned_only

Second-level SPM one-sample t-test (df = 29) of the searchlight odor-value RSA
beta maps, premeal + postmeal combined (`--session both`), **preconditioned
cues only** (`subset-preconditioned_only`), run-wise centered patterns.
Contrast: `RSA both | subset-preconditioned_only | model-odor_value > 0`.

Same participants (n = 30), exclusions, smoothing, and mask as
`../spm_value_n30/README.md`; see that file for details.

## Task for Codex

Rerun the same peak-level SVC used for `searchlight_svc_existing_group_v1`
(vmPFC, medial OFC, lateral OFC) with this folder's `SPM.mat` as the source group
model, and write the results to a new folder under
`rsa_roi/probe_imagewise_value/`.

The supplement currently says separate conditioned and preconditioned analyses
had no significant value-related peaks (all FWE p >= 0.201). That statement was
based on an older group model that included sub-31 (31 participants), so recheck
it after the SVC.

## Check values

- Peak t in `spmT_0001.nii`: 3.658 at MNI [40, -12, -32].
- SHA-256 `SPM.mat`: `72e614c418231c1157b56f4271bf793eb3dccbb4aac269ee02fdd38c318a8ebc`
- SHA-256 `spmT_0001.nii`: `2ef031713e27664c9211edf254b6da6d2a39331f3aa4f82751d0183752dddcc2`
