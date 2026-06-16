# Neubert 2015 OFC thresholded masks

These masks are binary copies of the Neubert 2015 OFC masks used in the
thresholded ROI RSA outputs under `rsa_roi/probe_imagewise_odor_identity/` and
`rsa_roi/probe_imagewise_odor_identity_no_image/`.

They were generated with:

```bash
python3 scripts_MRI/roi_mask_prep/save_thresholded_neubert_roi_masks.py
```

For each ROI, the script reads the saved ROI RSA subject-level CSV metadata,
uses the recorded `roi_threshold`, and applies the same mask logic as
`run_roi_rsa.py`:

```text
ROI mask >= roi_threshold
AND
masks/gm_0.1.nii > 0
```

The `thr50/` masks correspond to `roi_threshold = 50.0`; the `thr60/` masks
correspond to `roi_threshold = 60.0`. The `manifest.csv` file records source
mask paths, output paths, thresholds, and voxel counts.

## Rerun ROI RSA from saved masks

To rerun the thresholded Neubert ROI RSA analyses directly from these saved
binary masks, use:

```bash
python3 scripts_MRI/roi_mask_prep/run_neubert_thresholded_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data
```

The runner processes both RSA variants by default:

```text
rsa_imagewise_odor_identity
rsa_imagewise_odor_identity_no_image
```

and writes outputs back to:

```text
<base_dir>/rsa_roi/probe_imagewise_odor_identity/premeal_thr50/
<base_dir>/rsa_roi/probe_imagewise_odor_identity/premeal_thr60/
<base_dir>/rsa_roi/probe_imagewise_odor_identity_no_image/premeal_thr50/
<base_dir>/rsa_roi/probe_imagewise_odor_identity_no_image/premeal_thr60/
```

The runner passes the saved `mask-*_thr*_gm0p1.nii.gz` files as `--roi_mask`
and explicitly uses `--roi_threshold 0`. Therefore, rerun subject CSVs will show
`roi_threshold = 0.0`; the 50/60 threshold is encoded in the binary mask file
name and path rather than being applied again inside `run_roi_rsa.py`.

If the saved masks are not under `<base_dir>/rsa_roi/`, provide the synced mask
folder explicitly:

```bash
python3 scripts_MRI/roi_mask_prep/run_neubert_thresholded_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --mask_root /Users/liuq13/project-spcd/rsa_roi/neubert2015_ofc_thresholded_masks
```

Use `--dry_run` first to inspect the exact commands without running them.
