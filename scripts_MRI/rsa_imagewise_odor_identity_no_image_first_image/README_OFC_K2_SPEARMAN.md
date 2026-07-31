# Exploratory Spearman RSA for the Kahnt OFC K=2 clusters

This additions-only workflow reruns the established first-image, premeal,
no-image odor-identity ROI analysis with rank-based RDM statistics. It does not
change or overwrite the existing OLS regression RSA scripts or results.

The analysis should be described as an exploratory robustness analysis because
it was selected after inspecting the OLS results.

## What stays identical to the OLS analysis

- Separate medial/posterior-lateral and central/anterior-lateral OFC masks.
- First image only (`img_index == 1`).
- Odor-predictive cues only; clean-air cues are excluded.
- Conditioned and preconditioned cue subsets are analyzed separately.
- Patterns are centered within run.
- Neural dissimilarity is `1 - Pearson correlation` across ROI patterns.
- Only cross-run pairs are used.
- Voxels that are nonfinite in any selected sample are dropped.
- `sub-19` and `sub-33` are excluded only at group aggregation/plotting.

## Rank statistics

For every subject, cue subset, and ROI, the lower triangle of the neural RDM
and binary odor-identity RDM are restricted to cross-run pairs.

- `odor_identity_no_image_spearman_rho` is Spearman's correlation between the
  two selected vectors.
- `odor_identity_no_image_rank_biserial` is the rank-biserial separation of
  different-odor from same-odor neural distances.
- `odor_identity_no_image_auc` is the equivalent probability-scale statistic:
  `rank_biserial = 2 * AUC - 1`.
- `odor_identity_no_image_distance_contrast` is the unranked mean
  different-minus-same neural distance. It is saved as a reproducibility check
  against the single-regressor OLS beta.

Positive values mean that different-odor pairs tend to have greater neural
dissimilarity than same-odor pairs.

Pair-level p-values are intentionally not calculated because RDM pairs are
dependent. Group inference treats subjects as the independent units.
Spearman rho, rank-biserial correlation, and AUC are related summaries of the
same rank separation and should not be treated as independent confirmations.
Group p-values are uncorrected across clusters, metrics, and contrasts.

## Run on the computer containing SPCD_data

From the project clone:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_ofc_cluster_map_k2_spearman.py \
  --base_dir /Users/liuq13/SPCD_data \
  --save_diagnostics
```

The default runs both clusters and both cue subsets. `--save_diagnostics` is
recommended so the selected neural and model RDM pair vectors are retained.

To run only the new left- and right-hemisphere analyses without rerunning the
existing bilateral analyses:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_ofc_cluster_map_k2_spearman.py \
  --base_dir /Users/liuq13/SPCD_data \
  --hemispheres left right \
  --save_diagnostics \
  --strict
```

The prepared hemisphere masks are tracked in the repository. If they need to
be regenerated, run
`scripts_MRI/roi_mask_prep/prepare_ofc_cluster_map_k2_hemisphere_masks.py`
before the analysis. Use `--hemispheres bilateral left right` to run all three
ROI scopes together. The default remains `bilateral` for backward
compatibility.

To inspect commands without reading the external data:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_ofc_cluster_map_k2_spearman.py \
  --base_dir /Users/liuq13/SPCD_data \
  --dry_run
```

## Outputs

Subject and group CSVs are kept separate from the OLS results:

```text
rsa_roi/probe_imagewise_odor_identity_no_image_first_image/
  premeal_ofc_k2_spearman/
    cluster1_medial_posterior_lateral/
    cluster1_medial_posterior_lateral_left/
    cluster1_medial_posterior_lateral_right/
    cluster2_central_anterior_lateral/
    cluster2_central_anterior_lateral_left/
    cluster2_central_anterior_lateral_right/
```

Plots are written as:

```text
plots/OFC_K2_cluster1_medial_posterior_lateral_no_image_first_image_spearman_roi_rsa_paired.{pdf,png}
plots/OFC_K2_cluster2_central_anterior_lateral_no_image_first_image_spearman_roi_rsa_paired.{pdf,png}
```

Unilateral plot names add `_left` or `_right` immediately after the cluster
name. Each unilateral output uses its own ROI name, subject CSV, group summary,
diagnostics directory, and plot prefix, so no bilateral output is overwritten.

Group summaries include one-sample and paired t-tests, Wilcoxon signed-rank
tests, and sign tests for Spearman rho and rank-biserial correlation.

Separate significance tests in the left and right ROIs do not establish a
hemisphere difference. A laterality claim requires a paired within-subject
left-minus-right test. Because these atlas subdivisions contain different
numbers of voxels in the two hemispheres, interpret that comparison alongside
a voxel-count-matched sensitivity analysis.

## QC flags

The subject CSV explicitly records incomplete three-run data and voxel-loss
fractions. The group command reports subjects with fewer than three runs and
subjects losing at least 20% of ROI voxels, but it does not automatically
exclude them.

Based on the preceding OLS run, inspect:

- `sub-36`, which had runs 1 and 2 but no run 3.
- `sub-38`, which had substantial nonfinite voxel loss, particularly in the
  central/anterior-lateral cluster.

Any sensitivity analysis excluding a QC-flagged subject should be labeled and
reported alongside the primary analysis rather than replacing it.
