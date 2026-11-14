# First-image-only image-wise odor-identity RSA with local visual control

This folder is a first-image-only variant of `rsa_imagewise_odor_identity_local_visual`.
It uses the same premeal image-wise beta index files, but keeps only samples
with `img_index == 1`.

The model is:

```text
neural RDM ~ odor_identity + visual_image
```

`visual_image` is estimated from independent localizer betas within the same
searchlight sphere or ROI voxel set. Button responses are not loaded and no
button regressors are included. Clean-air cues are excluded. The supported cue
subsets are:

- `conditioned_odor_only`: `B11`, `B21`, `B12`, `B22`
- `preconditioned_odor_only`: `A11`, `A21`, `A12`, `A22`

## Searchlight RSA

Run one subject/subset/target:

```bash
cd /Users/liuq13/SPCD_RSA_imagewise_odor_identity_local_visual_first_image_subset

python -m rsa_imagewise_odor_identity_local_visual_first_image.run_rsa \
  --sub sub-01 \
  --base_dir /Users/liuq13/SPCD_data \
  --cue_subset preconditioned_odor_only \
  --target odor_identity
```

Available targets:

- `odor_identity`
- `visual_image`

Default output:

```text
<base_dir>/rsa_maps/probe_imagewise_odor_identity_local_visual_first_image/premeal/
```

Make a Biowulf swarm file:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_local_visual_first_image/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_odor_identity_local_visual_first_image_subset \
  --out imagewise_odor_identity_local_visual_first_image_premeal.swarm
```

## ROI RSA

Run one binary Kahnt OFC K=2 ROI mask:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_local_visual_first_image/run_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --roi_mask /Users/liuq13/project-spcd/masks/OFC_clusterMap_K2_gm0p1/mask-OFC_K2_cluster2_central_anterior_lateral_gm0p1.nii.gz \
  --roi_name OFC_K2_cluster2_central_anterior_lateral \
  --roi_threshold 0 \
  --drop_nonfinite_voxels
```

Default output:

```text
<base_dir>/rsa_roi/probe_imagewise_odor_identity_local_visual_first_image/premeal/
```

Aggregate subject ROI betas:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_local_visual_first_image/aggregate_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --roi_name OFC_K2_cluster2_central_anterior_lateral
```
