# Cross-task passive-odor template alignment

This ROI workflow tests whether first-image premeal cue patterns resemble the
independently measured sensory pattern of the odor predicted by each cue. It is
separate from classifier decoding and does not modify the existing RSA or
decoding results.

## Analysis

For each subject and ROI, passive odor trials are z-scored voxelwise within
run. Odor-1 and odor-2 patterns are averaged within each passive run, and the
run-specific means are averaged with equal run weight to form two prototypes.

Each cue pattern is independently z-scored within its premeal run and
correlated with both passive prototypes. Conditioned and preconditioned cue
subsets are normalized and analyzed separately. The primary trial measure is:

```text
matching_correlation_advantage =
  correlation(cue, predicted_odor_prototype)
  - correlation(cue, alternative_odor_prototype)
```

The subject statistic gives every cue run and odor identity equal weight.
Positive values indicate that cue patterns are more similar to the matching
perceived-odor representation.

After centering and unit-normalizing patterns across voxels, the same measure
is the label-oriented projection onto the correlation-space identity axis
`odor2_prototype - odor1_prototype`. The implementation verifies this equality
for every trial. Fisher-z correlation contrasts and nearest-prototype accuracy
are secondary measures, not independent confirmations.

## Shared conventions

The workflow imports the tested loading and labeling functions from
`scripts_MRI/Decoding_probe_cross_task/` and therefore preserves:

- passive beta numbering before clean-air trials are removed;
- raw passive `odor_1` and `odor_2` labels;
- original premeal event labels for `sub-14`;
- first-image-only cue samples from the Spearman RSA diagnostics;
- joint passive/cue nonfinite-voxel removal;
- within-task, within-run voxelwise z-scoring;
- MNI `x < 0` and `x > 0` hemisphere definitions.

When both cue subsets are requested, one common finite-voxel set is used for
them, but each subset is normalized and scored separately.

## Run on the data computer

From the project clone:

```bash
python scripts_MRI/crosstask_odor_template_alignment/run_ofc_k2_odor_template_alignment.py \
  --base_dir /Users/liuq13/SPCD_data \
  --hemispheres bilateral left right \
  --cue_subsets preconditioned_odor_only conditioned_odor_only \
  --strict
```

The hemisphere and cue arguments shown above are also the defaults. The runner
uses the bilateral Spearman RSA diagnostics for each K=2 ROI and performs the
hemisphere selection internally. Inspect all generated commands without
reading the external data with `--dry_run`.

## Outputs

Subject results are written under the project clone, not under `SPCD_data`:

```text
roi_template_alignment/passive_odor_to_premeal_cues/
  roi-OFC_K2_cluster1_medial_posterior_lateral_bilateral/
  roi-OFC_K2_cluster1_medial_posterior_lateral_left/
  roi-OFC_K2_cluster1_medial_posterior_lateral_right/
  roi-OFC_K2_cluster2_central_anterior_lateral_bilateral/
  roi-OFC_K2_cluster2_central_anterior_lateral_left/
  roi-OFC_K2_cluster2_central_anterior_lateral_right/
  ofc_k2_paired_comparisons.csv
```

Each ROI/scope directory contains:

```text
subject_template_alignment.csv
group_template_alignment_summary.csv
failures.csv
trial_alignment/<sub>/<cue_subset>.csv
```

The group summary reports one-sided tests against zero for the primary
matching advantage, subject-level bootstrap confidence intervals, Wilcoxon and
sign tests, and a deterministic subject-level sign-flip test. Conditioned
minus preconditioned comparisons are two-sided.

The omnibus comparison CSV contains direct paired tests for:

- conditioned minus preconditioned cues;
- cluster 2 minus cluster 1;
- left minus right;
- bilateral minus the mean of left and right.

Plots are written to `plots/` separately for bilateral, left, and right ROI
scopes. A significant effect in one hemisphere but not the other does not show
lateralization; use the direct left-minus-right comparison. Because the two
hemisphere masks differ in voxel count, interpret any laterality result with a
voxel-count-matched sensitivity analysis.

## Inference hierarchy

The preconditioned cue effect in the ROI motivating this validation should be
treated as primary. Conditioned cues are a positive comparison, the second K=2
ROI is a comparison region, and hemisphere analyses are exploratory. Group
inference uses subjects, not cue trials, as independent observations. This is
an independent sensory-template test but uses the same premeal cue data as the
original RSA, so it is convergent validation rather than an independent
replication.
