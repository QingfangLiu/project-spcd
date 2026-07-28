# Kahnt OFC K=2 masks and ROI RSA

This workflow prepares and analyzes the two clusters in
`masks/OFC_clusterMap_K2.nii.gz`. The source map is from the NeuroVault
collection **Connectivity-Based Parcellation of the Human Orbitofrontal
Cortex**:

- NeuroVault collection:
  <https://identifiers.org/neurovault.collection:17528>
- Collection page: <https://neurovault.org/collections/17528/>
- Related paper: Kahnt T, Chang LJ, Park SQ, Heinzle J, Haynes J-D.
  *Connectivity-based parcellation of the human orbitofrontal cortex.*
  Journal of Neuroscience (2012).
  DOI: <https://doi.org/10.1523/JNEUROSCI.0257-12.2012>

## Published K=2 cluster identities

The paper describes the two bilateral connectivity-based clusters as:

1. **Medial/posterior-lateral OFC**: medial OFC together with posterior parts
   of lateral OFC.
2. **Central/anterior-lateral OFC**: central and anterior-lateral parts of the
   orbital surface.

The workflow preserves the original cluster numbers in every mask, ROI label,
result directory, and plot name.

## Prepared masks

The downloaded map is a 3 mm label image, whereas the SPCD trialwise beta maps
and `masks/gm_0.1.nii` use a 2 mm grid. The preparation script:

1. identifies the two positive source labels;
2. maps the lower label to published cluster 1 and the higher label to cluster
   2;
3. resamples the integer label image to `masks/gm_0.1.nii` using
   nearest-neighbor interpolation;
4. intersects each cluster with the positive gray-matter mask;
5. writes separate binary masks and a provenance manifest.

Run from any directory:

```bash
python3 scripts_MRI/roi_mask_prep/prepare_ofc_cluster_map_k2_masks.py
```

Default outputs:

```text
masks/OFC_clusterMap_K2_gm0p1/
  mask-OFC_K2_cluster1_medial_posterior_lateral_gm0p1.nii.gz
  mask-OFC_K2_cluster2_central_anterior_lateral_gm0p1.nii.gz
  manifest.csv
```

Use `--dry_run` to inspect the operation without writing files. Existing
outputs are protected unless `--force` is supplied.

If the `gm_0.1.nii` file on the analysis computer differs from the repository
copy, regenerate the masks against that file and direct the runner to the new
directory:

```bash
python3 scripts_MRI/roi_mask_prep/prepare_ofc_cluster_map_k2_masks.py \
  --reference_mask /Users/liuq13/SPCD_data/masks/gm_0.1.nii \
  --output_dir /Users/liuq13/SPCD_data/rsa_roi/OFC_clusterMap_K2_gm0p1
```

## Run both ROI analyses

The orchestration script uses the established
`rsa_imagewise_odor_identity_no_image_first_image` implementation without
modifying it. For each cluster it runs:

- premeal data only;
- first image only;
- conditioned-odor and preconditioned-odor cue subsets separately;
- `neural RDM ~ odor_identity_no_image`;
- cross-run RDM pairs;
- within-run pattern centering;
- removal of voxels with nonfinite values;
- subject-level ROI RSA, group aggregation, and paired plotting.

On the computer containing `/Users/liuq13/SPCD_data`, run:

```bash
python3 \
  scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_ofc_cluster_map_k2.py \
  --base_dir /Users/liuq13/SPCD_data
```

Inspect all commands without accessing the external data:

```bash
python3 \
  scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_ofc_cluster_map_k2.py \
  --base_dir /Users/liuq13/SPCD_data \
  --dry_run
```

By default, subject CSVs, group summaries, and optional diagnostics remain
separate by cluster:

```text
rsa_roi/probe_imagewise_odor_identity_no_image_first_image/premeal_ofc_k2/
  cluster1_medial_posterior_lateral/
  cluster2_central_anterior_lateral/
```

The group summaries and plots exclude `sub-19` and `sub-33`, matching the
previous Left 47m analysis. The subject-level CSVs retain all successfully
analyzed subjects.

Default plots:

```text
plots/OFC_K2_cluster1_medial_posterior_lateral_no_image_first_image_roi_rsa_paired.pdf
plots/OFC_K2_cluster1_medial_posterior_lateral_no_image_first_image_roi_rsa_paired.png
plots/OFC_K2_cluster2_central_anterior_lateral_no_image_first_image_roi_rsa_paired.pdf
plots/OFC_K2_cluster2_central_anterior_lateral_no_image_first_image_roi_rsa_paired.png
```

Useful options include:

```text
--clusters cluster1
--subjects sub-01 sub-02
--save_diagnostics
--strict
--skip_plots
--repo_output_dir /custom/result/path
--plot_dir /custom/plot/path
--mask_dir /custom/prepared/mask/path
```

No existing analysis or plotting script is changed by this workflow.
