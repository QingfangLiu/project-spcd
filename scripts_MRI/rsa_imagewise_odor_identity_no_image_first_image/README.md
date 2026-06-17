# First-image-only no-image odor-identity RSA

This folder is a first-image-only variant of
`rsa_imagewise_odor_identity_no_image`. It uses the same premeal image-wise
beta index files, but keeps only samples with `img_index == 1`.

The model is:

```text
neural RDM ~ odor_identity_no_image
```

Image identity and button responses are not modeled. Button events are not
loaded. Clean-air cues are excluded. The supported cue subsets are:

- `conditioned_odor_only`: `B11`, `B21`, `B12`, `B22`
- `preconditioned_odor_only`: `A11`, `A21`, `A12`, `A22`

## Searchlight RSA

Run one subject/subset:

```bash
cd /Users/liuq13/SPCD_RSA_imagewise_odor_identity_no_image_first_image_subset

python -m rsa_imagewise_odor_identity_no_image_first_image.run_rsa \
  --sub sub-01 \
  --base_dir /Users/liuq13/SPCD_data \
  --cue_subset preconditioned_odor_only \
  --target odor_identity_no_image
```

Default output:

```text
<base_dir>/rsa_maps/probe_imagewise_odor_identity_no_image_first_image/premeal/
```

Make a Biowulf swarm file:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_odor_identity_no_image_first_image_subset \
  --out imagewise_odor_identity_no_image_first_image_premeal.swarm
```

## ROI RSA

Run one thresholded ROI mask:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --roi_mask rsa_roi/neubert2015_ofc_thresholded_masks/Left_47m_thr50.nii.gz \
  --roi_name Left_47m_thr50 \
  --roi_threshold 0 \
  --drop_nonfinite_voxels
```

Default output:

```text
<base_dir>/rsa_roi/probe_imagewise_odor_identity_no_image_first_image/premeal/
```

Aggregate subject ROI betas:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/aggregate_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --roi_name Left_47m_thr50
```
