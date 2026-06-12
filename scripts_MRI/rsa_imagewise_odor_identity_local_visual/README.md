# Premeal image-wise odor-identity RSA with local visual control

This folder implements a fourth premeal image-wise odor-identity RSA variant.
It is intended to sit alongside:

- `scripts_MRI/rsa_imagewise_odor_identity/`
- `scripts_MRI/rsa_imagewise_odor_identity_no_image/`
- `scripts_MRI/rsa_imagewise_odor_identity_visual_image/`

The model is:

```text
probe searchlight neural RDM
  ~ odor_identity
  + visual_image
  + button_presence
  + button_side
```

The difference from `rsa_imagewise_odor_identity_visual_image` is how the
`visual_image` RDM is computed:

- `rsa_imagewise_odor_identity_visual_image`: one fixed subject/subset visual
  RDM from the independent-scan visual-cortex union mask, reused across all
  probe searchlights.
- `rsa_imagewise_odor_identity_local_visual`: one subject/subset/searchlight
  visual RDM from independent-scan beta patterns in the same voxel sphere as
  the current probe searchlight.

This variant asks whether odor identity explains probe similarity beyond
whatever independent image similarity is represented in the same local brain
region.

## Cue subsets

This remains an odor-predictive cue analysis. Clean-air cues are excluded.

```text
conditioned_odor_only:    B11, B21, B12, B22
preconditioned_odor_only: A11, A21, A12, A22
```

Clean-air cue conditions are not included:

```text
C1, C2, D1, D2
```

## Local visual RDM construction

For each subject, subset, and searchlight sphere:

1. The probe neural RDM is computed from the premeal probe beta patterns inside
   the current searchlight sphere.
2. Independent image-scan beta patterns are extracted from the same sphere.
3. Independent localizer betas are z-scored voxelwise within localizer run
   across all localizer trials by default.
4. The subset-relevant localizer image rows are averaged by base cue.
5. `visual_image` is computed as `1 - correlation` between those averaged
   base-cue patterns.
6. The four-cue base RDM is expanded to the probe image-wise sample order.

Localizer category selection follows `task_info/subject_info.csv`:

```text
conditioned_odor_only:
  pairing = c -> character localizer
  pairing = p -> place localizer

preconditioned_odor_only:
  pairing = c -> place localizer
  pairing = p -> character localizer
```

Base cue to independent localizer image mapping:

```text
A11 -> img_1    B11 -> img_2
A21 -> img_3    B21 -> img_4
A12 -> img_7    B12 -> img_8
A22 -> img_9    B22 -> img_10
```

The localizer image list includes all independent localizer trials from the
selected character/place task so that the default within-run z-scoring is done
across the full independent-scan run. Rows outside the current subset's four
cue images are used only for z-scoring.

Degenerate searchlights return `NaN`, including cases with too few usable
voxels, nonfinite values, constant probe patterns, or a non-informative local
visual RDM.

## Diagnostics

This variant does not save every local visual RDM because that would create one
RDM per searchlight. If `--save_visual_diagnostics` is used, it writes compact
diagnostics:

```text
<output_dir>/diagnostics/<sub>/<subset>_local_visual_localizer_samples.csv
<output_dir>/diagnostics/<sub>/<subset>_local_visual_input_summary.csv
<output_dir>/diagnostics/<sub>/<subset>_local_visual_searchlight_qc_model-<target>.csv
```

The QC file records the number of searchlights in mask, valid searchlights,
`NaN` searchlights, valid fraction, and summary statistics for the saved score
map.

## Running the analysis

Prepare a Biowulf-ready subset from the full data folder:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_local_visual/prepare_rsa_subset.py \
  --full_data_dir /Users/liuq13/SPCD_data \
  --subset_dir /Users/liuq13/SPCD_RSA_imagewise_odor_identity_local_visual_subset
```

The subset prep copies:

- this package
- `masks/gm_0.1.nii`
- `task_info/subject_info.csv`
- premeal probe trialwise GLM beta files and events
- character and place independent image-scan beta files and events

Run one local or Biowulf subject/subset from the subset root:

```bash
python -m rsa_imagewise_odor_identity_local_visual.run_rsa \
  --sub sub-01 \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_odor_identity_local_visual_subset \
  --cue_subset conditioned_odor_only \
  --target odor_identity \
  --save_visual_diagnostics
```

Default searchlight output:

```text
<base_dir>/rsa_maps/probe_imagewise_odor_identity_local_visual/premeal/
```

Create a swarm file:

```bash
python rsa_imagewise_odor_identity_local_visual/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_odor_identity_local_visual_subset \
  --targets odor_identity visual_image \
  --save_visual_diagnostics
```

For the main hypothesis, `odor_identity` is the primary target. Saving
`visual_image` is useful as QC/supplementary output. Regardless of which target
map is saved, the regression includes all four RDMs.

## Group aggregation

Group-level searchlight aggregation is handled by `aggregate_rsa_group.m`.
The default `model_tag` is `odor_identity`. To aggregate visual-image maps,
change:

```matlab
model_tag = 'odor_identity';
```

to:

```matlab
model_tag = 'visual_image';
```
