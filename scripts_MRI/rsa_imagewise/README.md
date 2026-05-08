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
premeal samples and postmeal ratings for postmeal samples. A `phase` RDM
(`premeal` vs `postmeal`) is added as a nuisance model and is also available as
an output target.

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
- `odor_value`: subject- and session-specific odor-value-distance model derived from pleasantness ratings, with the three group ratings z-scored by default before distances are computed
- `cue_type`: preconditioned (`A`, `C`) vs conditioned (`B`, `D`)
- `image_identity`: cue/image identity ignoring first-vs-second presentation position (`A11_img1` and `A11_img2` are treated as the same identity)
- `phase`: premeal vs postmeal; only included when running `--session both`
- `button_presence`: pressed vs none
- `button_side`: left vs right, with `none` treated as different

`create_odor_group_rdm` remains available in `models.py` for diagnostics or
future comparisons, but `odor_group` is intentionally excluded from the current
multiple-regression RSA model because it overlaps strongly with the
odor_value RDM.

## Output
One file per model regressor:
- `sub-01_model-odor_value_subset-<...>.nii.gz`
- `sub-01_model-cue_type_subset-<...>.nii.gz`
- `sub-01_model-image_identity_subset-<...>.nii.gz`
- `sub-01_model-phase_subset-<...>.nii.gz` for `--session both`
- `sub-01_model-button_presence_subset-<...>.nii.gz`
- `sub-01_model-button_side_subset-<...>.nii.gz`

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
`--sessions premeal postmeal both`, each subject gets 16 tasks for `cue_subset
all`: five models for premeal, five for postmeal, and six for `both` because
the combined session also includes `phase`.

If `odor_value` is requested as a target, subjects missing compiled ratings
are skipped automatically when writing the swarm file.
