# Second-cue image value RSA: OFC small-volume correction

None of the nine analyses survives peak-level SVC at FWE p < .05 in either OFC ROI. The smallest corrected p is .124252 for the signed difference model with all cues in medial OFC: T(29) = 3.813151, MNI [4, 52, -22].

Native SPM12 positive one-sample T contrasts, n = 30, df = 29; premeal and postmeal combined. The nine models use only the second cue image.

Peak-level FWE correction is applied separately within the established medial and lateral OFC K2 gray-matter masks. This matches `searchlight_svc_n30_ofc_v1`: full native ROI resels with each group model's own residual smoothness, and the finite ROI/group-mask overlap as the voxel count. No correction is applied across the two ROIs or nine analyses.

| Value model | Cue subset | Medial OFC p FWE | Lateral OFC p FWE |
|---|---|---:|---:|
| cue_value | all | 0.299745 | 0.803073 |
| cue_value | conditioned_only | 0.243325 | 0.696012 |
| cue_value | preconditioned_only | 0.998960 | 0.994778 |
| signed_difference | all | 0.124252 | 0.978241 |
| signed_difference | conditioned_only | 0.813366 | 0.823546 |
| signed_difference | preconditioned_only | 0.978172 | 0.999531 |
| absolute_difference | all | 0.862852 | 0.769812 |
| absolute_difference | conditioned_only | 0.696963 | 0.693386 |
| absolute_difference | preconditioned_only | 0.453480 | 0.478005 |

The combined CSV includes all 18 peaks, MNI coordinates, T values, uncorrected and corrected p values, critical T thresholds, ROI voxel counts, resel volumes, and counts above voxelwise p < .001 and SVC FWE < .05.

Per-analysis results and metadata are saved in the sibling directories `searchlight_svc_n30_ofc_second_image_<model>[_<subset>]_v1/`. Metadata includes source checksums and full resel vectors.

Validation: reproduced both existing n30 OFC SVC rows; independently checked every numeric summary field, native ROI resels, image grids, participants, synced source hashes, and identical group masks with Python. No subject-level maps were needed; source SPM models were read only.

Reproduction: run `scripts_MRI/rsa_imagewise/run_second_image_value_svc.m` in MATLAB with native SPM12, then `python scripts_MRI/rsa_imagewise/verify_second_image_value_svc.py`.

These separately fitted models do not test differences between value representations. The supplement reports the medial OFC all-cue results for first-cue outcome value and second-cue signed value difference. The main manuscript is unchanged.
