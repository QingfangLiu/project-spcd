# Image-wise RSA pipeline

This pipeline runs searchlight RSA on **image-wise betas** and includes **button press** as a nuisance regressor.

## What it uses
- one beta per image presentation
- `trial` and `img_index` from the beta index CSV
- `response_hand` from the matching `*_events.tsv`
- button labels: `left`, `right`, or `none`

## Button scope options
- `both_images`: apply the trial button to both image presentations
- `second_image_only`: apply the button only to `img_index == 2`

## Cue subset options
- `all`: include all cues
- `conditioned_only`: include only conditioned cues (`B*`, `D*`)
- `preconditioned_only`: include only preconditioned cues (`A*`, `C*`)

When using a single-class subset (`conditioned_only` or `preconditioned_only`),
`conditioning` is automatically removed from the multiple-regression RSA model because it has no variance.

## Run
```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --button_scope both_images
```

Pleasantness-target example:

```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --button_scope both_images \
  --ratings_file /path/to/compiled_pleasantness_ratings.csv \
  --target pleasantness
```

Conditioned-only example:

```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --cue_subset conditioned_only \
  --button_scope both_images
```

To run a single output map while still fitting the full multiple-regression RSA model:

```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --button_scope both_images \
  --target image_identity
```

## Expected files
The code expects these fixed paths under `base_dir`:
- `GLM/premeal_trialwise_2mm/sub-01/sub-01_premeal_run-1_beta_index.csv`
- `nii/sub-01/func_PAID/premeal/sub-01_task-premeal_run-1_events.tsv`
- `masks/gm_0.1.nii`

For the pleasantness model, the code can also take a compiled ratings CSV via
`--ratings_file`. Expected columns:

- `sub`
- `pre_odor1`, `pre_odor2`, `pre_clean`
- `post_odor1`, `post_odor2`, `post_clean`

If `--ratings_file` is not provided, the loader falls back to:

- `beh_processed_subs/<sub>/pleasantness_summary.csv`

## Models
- `odor_group`: odor-group structure (`odor1`, `odor2`, `clean`)
- `pleasantness`: subject- and session-specific pleasantness-distance model derived from odor ratings
- `conditioning`: preconditioned (`A`, `C`) vs conditioned (`B`, `D`)
- `image_identity`: exact cue-image identity at the `A11_img1` / `A11_img2` level
- `button_presence`: pressed vs none
- `button_side`: left vs right, with `none` treated as different

## Output
One file per model regressor:
- `sub-01_model-odor_group_buttonscope-<...>_subset-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-pleasantness_buttonscope-<...>_subset-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-conditioning_buttonscope-<...>_subset-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-image_identity_buttonscope-<...>_subset-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-button_presence_buttonscope-<...>_subset-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-button_side_buttonscope-<...>_subset-<...>_radius-<...>mm.nii.gz`

## Biowulf

`make_swarm_imagewise.py` accepts `--ratings_file` and passes it through to
each RSA job. Example:

```bash
python scripts_MRI/rsa_imagewise/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_subset \
  --ratings_file /data/liuq13/SPCD_RSA_imagewise_subset/compiled_pleasantness_ratings.csv \
  --targets pleasantness \
  --cue_subset all \
  --out pleasantness_imagewise.swarm
```

If `pleasantness` is requested as a target, subjects missing compiled ratings
are skipped automatically when writing the swarm file.
