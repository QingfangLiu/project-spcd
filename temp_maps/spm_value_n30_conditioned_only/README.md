# spm_value_n30_conditioned_only

Second-level SPM one-sample t-test (df = 29) of the searchlight odor-value RSA
beta maps, premeal + postmeal combined (`--session both`), **conditioned cues
only** (`subset-conditioned_only`), run-wise centered patterns. Contrast:
`RSA both | subset-conditioned_only | model-odor_value > 0`.

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

- Peak t in `spmT_0001.nii`: 4.225 at MNI [-36, 58, -6].
- SHA-256 `SPM.mat`: `0c5a73ccf49bf32b10b3c351fb35b166af0d8b2fdaae500d3d2b1cf7dbaaf162`
- SHA-256 `spmT_0001.nii`: `24221028fca2328bd1aeb04e8c07f7ad680c7df843edd91f14ad2ec59eb15646`
