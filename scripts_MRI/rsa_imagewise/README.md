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

## Run
```bash
python -m rsa_imagewise.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --button_scope both_images
```

## Expected files
The code expects these fixed paths under `base_dir`:
- `GLM/premeal_trialwise_2mm/sub-01/sub-01_premeal_run-1_beta_index.csv`
- `nii/sub-01/func_PAID/premeal/sub-01_task-premeal_run-1_events.tsv`
- `masks/gm_0.1.nii`

## Models
- `odor_group`: odor-group structure (`odor1`, `odor2`, `clean`)
- `conditioning`: preconditioned (`A`, `C`) vs conditioned (`B`, `D`)
- `image_identity`: exact cue-image identity at the `A11_img1` / `A11_img2` level
- `button_presence`: pressed vs none
- `button_side`: left vs right, with `none` treated as different

## Output
One file per model regressor:
- `sub-01_model-odor_group_buttonscope-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-conditioning_buttonscope-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-image_identity_buttonscope-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-button_presence_buttonscope-<...>_radius-<...>mm.nii.gz`
- `sub-01_model-button_side_buttonscope-<...>_radius-<...>mm.nii.gz`
