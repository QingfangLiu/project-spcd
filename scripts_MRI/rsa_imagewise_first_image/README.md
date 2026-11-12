# First-image-only image-wise RSA pipeline

This pipeline is a first-image-only clone of `rsa_imagewise`. It runs
searchlight RSA on image-wise betas, but keeps only samples with
`img_index == 1`.

Button events are not loaded and button nuisance regressors are not included.

## What it uses
- one beta per trial: the first image presentation only
- `img_index`, `cue_index`, and `odor_index` from the beta index CSV
- `masks/gm_0.1.nii`

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
python -m rsa_imagewise_first_image.run_rsa \
  --sub sub-01 \
  --session both \
  --base_dir /path/to/SPCD_data
```

Odor-value target example:

```bash
python -m rsa_imagewise_first_image.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --ratings_file /path/to/compiled_pleasantness_ratings.csv \
  --target odor_value
```

Conditioned-only example:

```bash
python -m rsa_imagewise_first_image.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --cue_subset conditioned_only
```

To run a single output map while still fitting the full multiple-regression RSA model:

```bash
python -m rsa_imagewise_first_image.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --target image_identity
```

## Expected files
The code expects these fixed paths under `base_dir`:
- `GLM/premeal_trialwise_2mm/sub-01/sub-01_premeal_run-1_beta_index.csv`
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
- `image_identity`: cue/image identity among first-image-only samples

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

Each subject-level map contains the searchlight multiple-regression RSA beta
for that model regressor. Positive values mean local neural pattern
dissimilarity increases as the target model RDM predicts greater sample
dissimilarity, controlling for the other model RDMs included in the same RSA.

## Group aggregation outputs

`aggregate_rsa_group.m` runs SPM group-level aggregation on the subject-level
RSA maps. It loops over:

- sessions: `premeal`, `postmeal`, `both`
- cue subsets: `subset-all`, `subset-conditioned_only`, `subset-preconditioned_only`
- models: `odor_value`, `cue_type`, `image_identity`

For each available combination, the script masks voxels outside the gray-matter
mask to `NaN`, smooths the maps with a 6 mm FWHM kernel, and runs a persistent
SPM one-sample t-test against zero. It saves the full second-level SPM model
folder:

```text
rsa_maps/probe_imagewise_first_image/<session>/group_spm/model-<model>/<subset>/SPM.mat
```

Each model folder retains the standard SPM outputs, including `SPM.mat`,
`spmT_0001.nii`, `con_0001.nii`, `beta_0001.nii`, and `ResMS.nii`, plus the
persistent smoothed input maps under `spm_inputs/`. Open the `SPM.mat` in SPM
Results to inspect the second-level model. For the vmPFC value analysis, use
`model-odor_value/subset-all`.

The script also creates a `collapsed` group analysis. For this output, it finds
subjects with both premeal and postmeal maps for the same model/subset, averages
each subject's premeal and postmeal RSA maps, smooths those averaged maps, and
runs the same persistent one-sample SPM model:

```text
rsa_maps/probe_imagewise_first_image/collapsed/group_spm/model-<model>/<subset>/SPM.mat
```

Some requested combinations may be skipped because the corresponding
subject-level maps do not exist. For example, `cue_type` is dropped for single
cue-type subsets because it has no variance. The MATLAB script currently
excludes `sub-19`, `sub-31`, and `sub-33` from group aggregation, matching the
paper participant exclusions. Because the script skips any model folder that
already has an `SPM.mat`, delete the existing model folder before rerunning a
group model after changing this list.

## Cue-type pre/post difference

`aggregate_rsa_cue_type_phase_difference.m` tests whether cue-type RSA
representation changes from premeal to postmeal. This does not require a new RSA
run if the subject-level `subset-all` cue-type maps already exist for both
sessions:

```text
rsa_maps/probe_imagewise_first_image/premeal/sub-XX_model-cue_type_subset-all.nii.gz
rsa_maps/probe_imagewise_first_image/postmeal/sub-XX_model-cue_type_subset-all.nii.gz
```

The calculation starts from the subject-level `cue_type` RSA beta maps produced
by `run_rsa.py`. The `cue_type` RSA beta is the local searchlight coefficient
for the conditioned-vs-preconditioned cue-type RDM, controlling for the other
RDMs included in that RSA model. The script subtracts those RSA beta maps within
the same subject:

```text
cue_type subset-all postmeal - cue_type subset-all premeal
cue_type subset-all premeal - cue_type subset-all postmeal
```

This asks whether the strength of cue-type representation is modulated by meal
phase. It does not test whether raw BOLD magnitude differs between premeal and
postmeal cues. It should use `subset-all`; `cue_type` is not available in
`conditioned_only` or `preconditioned_only` maps because cue type has no
variance within a single cue-type subset.

The script writes subject-level difference maps:

```text
rsa_maps/probe_imagewise_first_image/phase_diffs/subject_diffs/sub-XX_model-cue_type_subset-all_contrast-postmeal_minus_premeal.nii.gz
rsa_maps/probe_imagewise_first_image/phase_diffs/subject_diffs/sub-XX_model-cue_type_subset-all_contrast-premeal_minus_postmeal.nii.gz
```

It then masks, smooths, and runs persistent one-sample SPM models against zero.
The group test asks whether the subject-level RSA beta difference is reliably
greater than zero across subjects. It saves:

```text
rsa_maps/probe_imagewise_first_image/phase_diffs/group_spm/model-cue_type/subset-all/postmeal_minus_premeal/SPM.mat
rsa_maps/probe_imagewise_first_image/phase_diffs/group_spm/model-cue_type/subset-all/premeal_minus_postmeal/SPM.mat
```

Each model folder retains standard SPM outputs and persistent smoothed inputs
under `spm_inputs/`. For the claim that cue-type representation decreases after
the meal, inspect `premeal_minus_postmeal`.

For `postmeal_minus_premeal`, positive t-values indicate stronger cue-type
representation postmeal than premeal. For `premeal_minus_postmeal`, positive
t-values indicate the reverse. These are the same paired contrast with opposite
signs, so they do not represent independent statistical tests.

## Odor-value cue-subset difference

`aggregate_rsa_odor_value_subset_difference.m` tests whether the odor-value RSA
effect differs between conditioned and preconditioned cue subsets. It is a
paired subject-level contrast, not a separate test of each subset against zero.

For each subject and session, the calculation starts from the subject-level RSA
beta maps produced by `run_rsa.py`. The `odor_value` RSA beta is the local
searchlight coefficient for the odor-value RDM, controlling for the other RDMs
included in that RSA model. The script subtracts those RSA beta maps within the
same subject:

```text
odor_value subset-conditioned_only - odor_value subset-preconditioned_only
odor_value subset-preconditioned_only - odor_value subset-conditioned_only
```

This asks whether the strength of value representation changes by cue type. It
does not test whether raw BOLD magnitude differs between conditioned and
preconditioned cues.

The script writes subject-level difference maps:

```text
rsa_maps/probe_imagewise_first_image/<session>/subject_diffs/sub-XX_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
rsa_maps/probe_imagewise_first_image/<session>/subject_diffs/sub-XX_model-odor_value_contrast-preconditioned_minus_conditioned.nii.gz
```

It then masks, smooths, and runs the same one-sample SPM group test against zero
used by `aggregate_rsa_group.m`. The group test asks whether the subject-level
RSA beta difference is reliably greater than zero across subjects. It saves:

```text
rsa_maps/probe_imagewise_first_image/<session>/group_tmaps/group_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
rsa_maps/probe_imagewise_first_image/<session>/group_tmaps/group_model-odor_value_contrast-preconditioned_minus_conditioned.nii.gz
```

It runs this for `premeal`, `postmeal`, and `both`. It also creates a collapsed
pre/post analysis by averaging each subject's premeal and postmeal difference
maps before the group test:

```text
rsa_maps/probe_imagewise_first_image/collapsed/group_tmaps/group_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
rsa_maps/probe_imagewise_first_image/collapsed/group_tmaps/group_model-odor_value_contrast-preconditioned_minus_conditioned.nii.gz
```

For `conditioned_minus_preconditioned`, positive t-values indicate stronger
odor-value representation for conditioned than preconditioned cues. For
`preconditioned_minus_conditioned`, positive t-values indicate the reverse.
These are the same paired contrast with opposite signs, so they do not represent
independent statistical tests. A significant positive cluster in either output
means the local pattern-value relationship is stronger for the cue type named
first in the contrast filename.

## Biowulf

`make_swarm_imagewise.py` accepts `--ratings_file` and passes it through to
each RSA job. Example:

```bash
python scripts_MRI/rsa_imagewise_first_image/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_first_image_subset \
  --ratings_file /data/liuq13/SPCD_RSA_imagewise_first_image_subset/compiled_pleasantness_ratings.csv \
  --sessions premeal postmeal both \
  --split_targets \
  --cue_subset all \
  --out imagewise_all_sessions_split_targets.swarm
```

With `--split_targets`, the swarm generator writes one task per valid target
model instead of one task that runs every model for a subject/session. For
`--sessions premeal postmeal both`, each subject gets 9 tasks for `cue_subset
all`: three models each for premeal, postmeal, and both.

If `odor_value` is requested as a target, subjects missing compiled ratings
are skipped automatically when writing the swarm file.
