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

## Second-image-only value analyses

The same package supports three alternative value models evaluated on **second
cue presentations only**. Use `--image_scope second` and choose one
`--value_model`:

| Value model | Variable assigned to each trial |
| --- | --- |
| `cue_value` | Rating of the outcome associated with cue 2, V2 |
| `signed_difference` | V2 - V1 |
| `absolute_difference` | abs(V2 - V1) |

For every model, the model RDM is `abs(variable_i - variable_j)` between trials.
V1 and V2 use the participant's ratings from the trial's probe phase. Ratings
are z-scored across the three outcomes for a single-phase analysis, or jointly
across all six phase-by-outcome ratings for `--session both`, **before** ratings
are expanded to trials. Differences use this same rating scale, without a
second normalization. The three value models are fitted in **separate
regressions**, not simultaneously. The selected value RDM is called `odor_value`
in the regression and output map; `--target` selects the output coefficient,
whereas `--value_model` selects which value RDM enters the fit.

The loader matches cues using trial IDs within each subject, phase, and run;
it does not assume adjacent CSV rows. Each trial must contain exactly one
first and one second cue of the same cue class. Missing/duplicated cue rows,
unknown cue mappings, or malformed button data stop the second-image analysis.
An absent run CSV is reported and skipped, allowing participants with a missing
run; a requested phase with no usable samples fails. The beta index CSV must
include `trial`, `img_index`, `cue_index`, `odor_index`, and `beta_file`. The
existing GLM and events files can be reused; no new first-level GLM is required.

The neural RDM, OLS estimator with intercept, searchlight, run-wise centering,
and cross-run pair selection are shared with the two-image pipeline. In `both`
mode, cross-phase pairs are included and a phase RDM is not regressed out.
Nuisance RDMs are second-image identity, button presence, button side, and cue
type when both cue classes are included. Controls that are constant on the
retained cross-run pairs are dropped and recorded. In particular, button
presence is dropped if every trial has a response. The analysis stops if the
value RDM is constant or the remaining regression is rank deficient. The older
two-image analysis remains the default and retains its existing model fitting.

From the folder containing the `rsa_imagewise` package, run one subject:

```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 --session both \
  --base_dir /Users/liuq13/SPCD_data \
  --ratings_file /Users/liuq13/project-spcd/beh_processed_subs/compiled_pleasantness_ratings.csv \
  --image_scope second --value_model signed_difference \
  --cue_subset all --target odor_value --center_within_run
```

Repeat with `--value_model cue_value` and `--value_model absolute_difference`.
Change `--cue_subset` to `conditioned_only` or `preconditioned_only` to estimate
either cue class separately. The default output folders prevent the three
versions from overwriting one another:

```text
<base_dir>/rsa_maps/probe_imagewise_second_image/cue_value/both/
<base_dir>/rsa_maps/probe_imagewise_second_image/signed_difference/both/
<base_dir>/rsa_maps/probe_imagewise_second_image/absolute_difference/both/
```

Each contains `sub-XX_model-odor_value_subset-all.nii.gz` (or the selected
subset). Use a distinct explicit `--output_dir` if changing centering or any
other setting. Under `diagnostics/sub-XX/subset-all/`, each run saves:

- `samples.csv`: actual paired cue labels, trial IDs, phase, run, button labels,
  and all normalized trial variables.
- `model_rdm_correlations.csv`: correlations on the retained cross-run pairs
  among all three value RDMs and the controls. First-cue identity and unordered
  outcome-pair identity are included **for diagnosis only**, not as additional
  regression controls.
- `model_rdms.npz`: exact diagnostic RDMs and the cross-run pair mask.
- `analysis_metadata.json`: fitted/dropped predictors, sample and pair counts,
  design rank/condition number, target variance remaining after controls, source
  code hashes, requested outputs, and completion status. A `prepared` status
  indicates the searchlight did not reach successful completion.

Inspect the correlations and remaining value variance before interpreting
results, especially absolute value differences with only three outcome-pair
types per phase. A significant coefficient in one version and a nonsignificant
coefficient in another is not a direct test of the difference between versions.

### Prepare and submit Biowulf jobs

Refresh the existing subset on the data computer:

```bash
python scripts_MRI/rsa_imagewise/prepare_rsa_subset.py
```

This copies the updated package (including `ratings.py` and `diagnostics.py`),
the compiled ratings CSV, mask, beta index CSVs, betas, and events. Its default
source is `/Users/liuq13/SPCD_data` and destination is
`/Users/liuq13/SPCD_RSA_imagewise_subset`; the script prints the rsync command.
The existing staged data can be reused; refresh the package and ratings before
submitting. After syncing, generate one job per subject and value model:

```bash
python scripts_MRI/rsa_imagewise/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_subset \
  --ratings_file /data/liuq13/SPCD_RSA_imagewise_subset/compiled_pleasantness_ratings.csv \
  --image_scope second --sessions both --cue_subset all \
  --targets odor_value --center_within_run \
  --out second_image_value_all.swarm
```

All three value models are generated by default for second-image jobs. To
restrict them, pass e.g. `--value_models cue_value signed_difference`. The new
second-image job generation excludes sub-19, sub-31, and sub-33 and skips
participants with missing/nonfinite ratings. Full cohort inclusion still
depends on the data present. Repeat with another `--cue_subset` and output
swarm filename for cue-specific analyses. Run jobs from the staged subset root,
so `python -m rsa_imagewise.run_rsa` can find the package. Use the same Biowulf
resources and swarm submission setup as the existing image-wise RSA jobs.

### Group models

After copying the subject maps back into the same relative `rsa_maps/` layout,
use the existing MATLAB aggregation script:

```matlab
base_dir = '/Users/liuq13/SPCD_data/rsa_maps';
analysis_mode = 'second_image_value';
run('/Users/liuq13/project-spcd/scripts_MRI/rsa_imagewise/aggregate_rsa_group.m');
```

It loops over all three value-model folders and the three cue subsets, using
6 mm smoothing, the existing gray-matter mask, and exclusions sub-19, sub-31,
sub-33. It saves each available model under
`<variant>/<session>/group_spm/model-odor_value/subset-<subset>/`.
Combined-session RSA lives in `both`; `collapsed` is instead an average of
separately estimated premeal and postmeal maps and is not equivalent to `both`.
Existing group model folders are skipped to avoid overwriting a fitted model.
For the original analysis, explicitly set `analysis_mode = 'both_images'`.

### Checks

```bash
python -B -m unittest discover -s scripts_MRI/rsa_imagewise -p 'test_*.py' -v
```

The tests use synthetic NIfTI data to check actual searchlight outputs against
direct OLS, the signed versus absolute definitions, shared normalization,
trial alignment across runs/phases, cue subsets, button controls, job
generation, and staged package dependencies. Full subject-data searchlights
must be run on the data computer or Biowulf.

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
mask to `NaN`, smooths the maps with a 6 mm FWHM kernel, and runs a persistent
SPM one-sample t-test against zero. It saves the full second-level SPM model
folder:

```text
rsa_maps/probe_imagewise/<session>/group_spm/model-<model>/<subset>/SPM.mat
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
rsa_maps/probe_imagewise/collapsed/group_spm/model-<model>/<subset>/SPM.mat
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
rsa_maps/probe_imagewise/premeal/sub-XX_model-cue_type_subset-all.nii.gz
rsa_maps/probe_imagewise/postmeal/sub-XX_model-cue_type_subset-all.nii.gz
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
rsa_maps/probe_imagewise/phase_diffs/subject_diffs/sub-XX_model-cue_type_subset-all_contrast-postmeal_minus_premeal.nii.gz
rsa_maps/probe_imagewise/phase_diffs/subject_diffs/sub-XX_model-cue_type_subset-all_contrast-premeal_minus_postmeal.nii.gz
```

It then masks, smooths, and runs persistent one-sample SPM models against zero.
The group test asks whether the subject-level RSA beta difference is reliably
greater than zero across subjects. It saves:

```text
rsa_maps/probe_imagewise/phase_diffs/group_spm/model-cue_type/subset-all/postmeal_minus_premeal/SPM.mat
rsa_maps/probe_imagewise/phase_diffs/group_spm/model-cue_type/subset-all/premeal_minus_postmeal/SPM.mat
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
rsa_maps/probe_imagewise/<session>/subject_diffs/sub-XX_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
rsa_maps/probe_imagewise/<session>/subject_diffs/sub-XX_model-odor_value_contrast-preconditioned_minus_conditioned.nii.gz
```

It then masks, smooths, and runs the same one-sample SPM group test against zero
used by `aggregate_rsa_group.m`. The group test asks whether the subject-level
RSA beta difference is reliably greater than zero across subjects. It saves:

```text
rsa_maps/probe_imagewise/<session>/group_tmaps/group_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
rsa_maps/probe_imagewise/<session>/group_tmaps/group_model-odor_value_contrast-preconditioned_minus_conditioned.nii.gz
```

It runs this for `premeal`, `postmeal`, and `both`. It also creates a collapsed
pre/post analysis by averaging each subject's premeal and postmeal difference
maps before the group test:

```text
rsa_maps/probe_imagewise/collapsed/group_tmaps/group_model-odor_value_contrast-conditioned_minus_preconditioned.nii.gz
rsa_maps/probe_imagewise/collapsed/group_tmaps/group_model-odor_value_contrast-preconditioned_minus_conditioned.nii.gz
```

The script excludes `sub-19`, `sub-31`, and `sub-33` (the paper exclusions).
For the `both` session only (`persist_sessions` in the script), it also keeps the
full second-level SPM model for each contrast, for example for small-volume
correction:

```text
rsa_maps/probe_imagewise/both/group_spm/model-odor_value/contrast-conditioned_minus_preconditioned/SPM.mat
rsa_maps/probe_imagewise/both/group_spm/model-odor_value/contrast-preconditioned_minus_conditioned/SPM.mat
```

Each folder retains the standard SPM outputs plus the smoothed inputs under
`spm_inputs/`. The script skips a contrast whose `group_tmaps` file or model
folder already exists, so delete the existing file or folder before rerunning
after changing the skip list.

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
