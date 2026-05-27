# Image-wise RSA pipeline

This pipeline runs searchlight RSA on **image-wise betas** and includes **button press** as a nuisance regressor.

## What it uses
- one beta per image presentation
- `trial` and `img_index` from the beta index CSV
- `response_hand` from the matching `*_events.tsv`
- button labels: `left`, `right`, or `none`

## Button scope options
- `second_image_only` is the default and intended setting: keep both image
  betas, but apply the trial button label only to `img_index == 2`; `img_index
  == 1` is labeled `none` for the button nuisance RDMs.
- `both_images` remains available for diagnostics: apply the trial button label
  to both image presentations.

RSA map filenames do not include `button_scope`. The standard output names
therefore assume the default `second_image_only` setting. If you explicitly run
`--button_scope both_images`, use a separate `--output_dir` to avoid overwriting
the default maps.

## Cue subset options
- `all`: include all cues
- `conditioned_only`: include only conditioned cues (`B*`, `D*`)
- `preconditioned_only`: include only preconditioned cues (`A*`, `C*`)

When using a single-class subset (`conditioned_only` or `preconditioned_only`),
`cue_type` is automatically removed from the multiple-regression RSA model because it has no variance.

## Session options
- `premeal`: run RSA on premeal image-wise betas only
- `postmeal`: run RSA on postmeal image-wise betas only
- `both`: concatenate premeal and postmeal image-wise betas into one RSA

In `both` mode, premeal and postmeal runs get distinct run IDs before the
cross-run RSA mask is built. The `odor_value` model uses premeal ratings for
premeal samples and postmeal ratings for postmeal samples. For `both`, the six
subject-specific phase-by-odor ratings are z-scored together before distances
are computed, preserving pre/post value changes while removing subject-level
rating scale differences. A binary `phase` RDM (`premeal` vs `postmeal`) is
still computed in the combined-session code path, but it is intentionally
excluded from the RSA regression so pre/post value change remains part of the
`odor_value` effect instead of being partialled out by a generic phase nuisance
regressor.

## Run
```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session both \
  --base_dir /path/to/SPCD_data
```

Odor-value target example:

```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --ratings_file /path/to/compiled_pleasantness_ratings.csv \
  --target odor_value
```

Conditioned-only example:

```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --cue_subset conditioned_only
```

To run a single output map while still fitting the full multiple-regression RSA model:

```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --target image_identity
```

## Expected files
The code expects these fixed paths under `base_dir`:
- `GLM/premeal_trialwise_2mm/sub-01/sub-01_premeal_run-1_beta_index.csv`
- `nii/sub-01/func_PAID/premeal/sub-01_task-premeal_run-1_events.tsv`
- `masks/gm_0.1.nii`

For the `odor_value` model, the code can also take a compiled pleasantness
ratings CSV via `--ratings_file`. Expected columns:

- `sub`
- `pre_odor1`, `pre_odor2`, `pre_clean`
- `post_odor1`, `post_odor2`, `post_clean`

If `--ratings_file` is not provided, the loader falls back to:

- `beh_processed_subs/<sub>/pleasantness_summary.csv`

## Models
- `odor_value`: subject-specific odor-value-distance model derived from pleasantness ratings. For single-session runs, the three group ratings are z-scored within that session; for `--session both`, the six premeal/postmeal group ratings are z-scored together within subject before distances are computed
- `cue_type`: preconditioned (`A`, `C`) vs conditioned (`B`, `D`)
- `image_identity`: cue/image identity ignoring first-vs-second presentation position (`A11_img1` and `A11_img2` are treated as the same identity)
- `button_presence`: pressed vs none
- `button_side`: left vs right, with `none` treated as different

The `phase` RDM can still be computed in `--session both` for diagnostics, but
it is not included in the multiple-regression RSA model and is not written as
an output target.

`create_odor_group_rdm` remains available in `models.py` for diagnostics or
future comparisons, but `odor_group` is intentionally excluded from the current
multiple-regression RSA model because it overlaps strongly with the
odor_value RDM.

## Output
One file per model regressor:
- `sub-01_model-odor_value_subset-<...>.nii.gz`
- `sub-01_model-cue_type_subset-<...>.nii.gz`
- `sub-01_model-image_identity_subset-<...>.nii.gz`
- `sub-01_model-button_presence_subset-<...>.nii.gz`
- `sub-01_model-button_side_subset-<...>.nii.gz`

Each subject-level map contains the searchlight multiple-regression RSA beta
for that model regressor. Positive values mean local neural pattern
dissimilarity increases as the target model RDM predicts greater sample
dissimilarity, controlling for the other model RDMs included in the same RSA.

## Group aggregation outputs

`aggregate_rsa_group.m` runs SPM group-level aggregation on the subject-level
RSA maps. It loops over:

- sessions: `premeal`, `postmeal`, `both`
- cue subsets: `subset-all`, `subset-conditioned_only`, `subset-preconditioned_only`
- models: `odor_value`, `cue_type`, `image_identity`, `button_presence`, `button_side`

For each available combination, the script masks voxels outside the gray-matter
mask to `NaN`, smooths the maps with a 6 mm FWHM kernel, and runs a one-sample
SPM t-test against zero. It saves only the group t-statistic image:

```text
rsa_maps/probe_imagewise/<session>/group_tmaps/group_model-<model>_<subset>.nii.gz
```

These `group_model-*.nii.gz` files are SPM `spmT_0001.nii` maps saved in a
shallow output layout. They are not thresholded statistical reports; they are
voxelwise t-statistic maps showing where the subject-level RSA beta is reliably
greater than zero at the group level.

The script also creates a `collapsed` group analysis. For this output, it finds
subjects with both premeal and postmeal maps for the same model/subset, averages
each subject's premeal and postmeal RSA maps, smooths those averaged maps, and
runs the same one-sample group t-test:

```text
rsa_maps/probe_imagewise/collapsed/group_tmaps/group_model-<model>_<subset>.nii.gz
```

Some requested combinations may be skipped because the corresponding
subject-level maps do not exist. For example, `cue_type` is dropped for single
cue-type subsets because it has no variance. The MATLAB script currently
excludes `sub-19` and `sub-33` from group aggregation.

## Odor-value cue-subset difference

`aggregate_rsa_odor_value_subset_difference.m` tests whether the odor-value RSA
effect differs between conditioned and preconditioned cue subsets. It is a
paired subject-level contrast, not a separate test of each subset against zero.

For each subject and session, it subtracts:

```text
odor_value subset-conditioned_only - odor_value subset-preconditioned_only
```

The script writes subject-level difference maps:

```text
rsa_maps/probe_imagewise/<session>/subject_diffs/sub-XX_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
```

It then masks, smooths, and runs the same one-sample SPM group test against zero
used by `aggregate_rsa_group.m`, saving:

```text
rsa_maps/probe_imagewise/<session>/group_tmaps/group_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
```

It runs this for `premeal`, `postmeal`, and `both`. It also creates a collapsed
pre/post analysis by averaging each subject's premeal and postmeal difference
maps before the group test:

```text
rsa_maps/probe_imagewise/collapsed/group_tmaps/group_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
```

Positive t-values indicate stronger odor-value representation for conditioned
than preconditioned cues. Negative t-values indicate the reverse.

## Biowulf

`make_swarm_imagewise.py` accepts `--ratings_file` and passes it through to
each RSA job. Example:

```bash
python scripts_MRI/rsa_imagewise/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_subset \
  --ratings_file /data/liuq13/SPCD_RSA_imagewise_subset/compiled_pleasantness_ratings.csv \
  --sessions premeal postmeal both \
  --split_targets \
  --cue_subset all \
  --out imagewise_all_sessions_split_targets.swarm
```

With `--split_targets`, the swarm generator writes one task per valid target
model instead of one task that runs every model for a subject/session. For
`--sessions premeal postmeal both`, each subject gets 15 tasks for `cue_subset
all`: five models each for premeal, postmeal, and both.

If `odor_value` is requested as a target, subjects missing compiled ratings
are skipped automatically when writing the swarm file.
