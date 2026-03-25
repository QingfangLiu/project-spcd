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
The code searches recursively for:
- `GLM/**/sub-01_premeal_run-1_beta_index.csv`
- `**/sub-01_task-premeal_run-1_events.tsv`
- `masks/gm_0.1.nii`

## Output
One file by default:
- `sub-01_model-identity_buttonscope-<...>_radius-<...>mm.nii.gz`
