# First-image-only image-wise run-pair RSA pipeline

This pipeline is a run-pair variant of `rsa_imagewise_first_image`. It runs
searchlight RSA on first-image-only image-wise betas, but estimates each RSA map
using only one acquisition-run pair at a time.

Button events are not loaded and button nuisance regressors are not included.

## Run-Pair Options

- `run1_vs_run2`
- `run1_vs_run3`
- `run2_vs_run3`

`--session both` is intentionally not supported in this variant. Run `premeal`
and `postmeal` separately to keep temporal trends within meal phase.

## Models

- `odor_value`: subject-specific odor-value-distance model derived from pleasantness ratings
- `cue_type`: preconditioned (`A`, `C`) vs conditioned (`B`, `D`)
- `image_identity`: cue/image identity among first-image-only samples

When using `conditioned_only` or `preconditioned_only`, `cue_type` is dropped
because it has no variance.

## Run One Searchlight Job

```bash
python -m rsa_imagewise_first_image_run_pairs.run_rsa \
  --sub sub-01 \
  --session premeal \
  --run_pair run1_vs_run2 \
  --base_dir /Users/liuq13/SPCD_data \
  --ratings_file /Users/liuq13/project-spcd/beh_processed_subs/compiled_pleasantness_ratings.csv \
  --target odor_value
```

## Output

Default output:

```text
<base_dir>/rsa_maps/probe_imagewise_first_image_run_pairs/<session>/run_pair-<run_pair>/
```

Example:

```text
/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_first_image_run_pairs/premeal/run_pair-run1_vs_run2/
```

Each folder contains maps like:

```text
sub-01_model-odor_value_subset-all.nii.gz
sub-01_model-cue_type_subset-all.nii.gz
sub-01_model-image_identity_subset-all.nii.gz
```

## Swarm

Generate jobs for both meal phases and all three run pairs:

```bash
python scripts_MRI/rsa_imagewise_first_image_run_pairs/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_first_image_run_pairs_subset \
  --out imagewise_first_image_run_pairs_rsa.swarm \
  --split_targets
```

Default sessions are `premeal postmeal`; default run pairs are all three.

## Subset Prep

```bash
python scripts_MRI/rsa_imagewise_first_image_run_pairs/prepare_rsa_subset.py
```

This writes:

```text
/Users/liuq13/SPCD_RSA_imagewise_first_image_run_pairs_subset
```
