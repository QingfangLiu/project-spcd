# Premeal image-wise odor-identity RSA with visual-image control

This folder is for a third premeal image-wise odor-identity RSA variant. It is
intended to sit alongside:

- `scripts_MRI/rsa_imagewise_odor_identity/`
- `scripts_MRI/rsa_imagewise_odor_identity_no_image/`

The current goal is to replace the categorical `image_identity` regressor with
a graded visual-image dissimilarity RDM computed from independent image-scan
responses in visual cortex.

## Current motivation

The two existing odor-identity variants handle visual image identity in two
ways:

- `rsa_imagewise_odor_identity`: controls for image identity with a binary RDM
  where pairs from the same base cue are 0 and pairs from different base cues
  are 1.
- `rsa_imagewise_odor_identity_no_image`: omits image identity from the RSA
  regression.

This variant will use a third option:

```text
probe neural RDM ~ odor_identity + visual_image + button_presence + button_side
```

where `visual_image` is a subject-specific dissimilarity RDM derived from
independent character/place image-scan beta patterns.

## Visual mask source maps

The visual-cortex masks are derived from the group-level independent image-scan
decoding results:

```text
masks/independent_image_decoding_masks/char_spmT_0001.nii
masks/independent_image_decoding_masks/place_spmT_0001.nii
```

These maps are copied into this repository from the previous machine's
`/Users/liuq13/SPCD_data` analysis outputs. They correspond to the
`across_trials` normalization option from the independent image decoding
pipelines, not the `none` normalization option.

The source pipelines are:

```text
scripts_MRI/Decoding_character/
scripts_MRI/Decoding_place/
```

Those pipelines decoded image identity from independent scan trial-wise GLM beta
maps using an SVM searchlight, leave-one-run-out cross-validation, an 8 mm
searchlight radius, and a gray-matter mask thresholded at 0.1.

The SPM group-level aggregation scripts:

```text
scripts_MRI/Decoding_character/aggregate_searchlight_group_level_spm_char.m
scripts_MRI/Decoding_place/aggregate_searchlight_group_level_spm_place.m
```

processed subject-level searchlight accuracy maps by:

- reading maps from the `across_trials` normalization folder
- excluding `sub-19` and `sub-33`
- applying the gray-matter mask
- centering maps at chance, `1/6`
- smoothing centered maps with 6 mm FWHM
- running a one-sample SPM t-test for `Decoding > chance`

The two files in `masks/independent_image_decoding_masks/` are the resulting
SPM `spmT_0001` maps for character and place image decoding.

## Planned mask construction

The preferred visual mask for this RSA variant is a common union mask:

```text
visual_decoding_union = threshold(char_spmT_0001) OR threshold(place_spmT_0001)
```

Using one union mask keeps the visual feature space consistent across
conditioned and preconditioned cue subsets. This is preferable to using separate
character and place masks because each subject's conditioned and preconditioned
cues are counterbalanced across character/place category.

The thresholding rule for this analysis is:

- threshold each group t-map in xjView at `p < 0.001` uncorrected
- use `display intensity = Only +`, corresponding to `Decoding > chance`
- use `T = 3.374`, `df = 31`, and `cluster size >= 5`
- manually retain only plausible visual cortex clusters
- save category-specific visual-only masks
- create a union mask and intersect it with `masks/gm_0.1.nii`

Save the xjView-cleaned category-specific masks as:

```text
masks/independent_image_decoding_masks/char_p001_visual_only.hdr
masks/independent_image_decoding_masks/char_p001_visual_only.img
masks/independent_image_decoding_masks/place_p001_visual_only.hdr
masks/independent_image_decoding_masks/place_p001_visual_only.img
```

Then create the binary union mask with:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_visual_image/make_visual_union_mask.py \
  --char_mask masks/independent_image_decoding_masks/char_p001_visual_only.hdr \
  --place_mask masks/independent_image_decoding_masks/place_p001_visual_only.hdr \
  --gm_mask masks/gm_0.1.nii \
  --out masks/independent_image_decoding_masks/visual_union_p001_visual_only_gm0p1.nii
```

The script treats any finite value greater than zero as inside-mask, performs a
voxelwise OR across the character and place masks, intersects the union with
`gm_0.1`, writes a binary `uint8` NIfTI image, and prints voxel counts for the
input masks, overlap, union before GM intersection, and final GM-intersected
mask.

The generated union mask is:

```text
masks/independent_image_decoding_masks/visual_union_p001_visual_only_gm0p1.nii
```

Generated-mask voxel counts:

```text
character visual-only mask:          10511
place visual-only mask:              14416
character/place overlap:              8400
union before GM intersection:        16527
GM mask voxels:                     198081
union after GM intersection:         16527
```

The saved union mask has shape `(79, 95, 79)`, `uint8` datatype, binary values
`[0, 1]`, and 16,527 in-mask voxels. The GM intersection did not remove
additional voxels from the xjView-selected visual clusters.

## Planned visual-image RDM use

For the current subset-specific analyses, the visual RDM only needs to be
computed within each cue subset:

```text
conditioned_odor_only:    B11, B21, B12, B22
preconditioned_odor_only: A11, A21, A12, A22
```

For each subject and subset, the future code should:

- select the correct independent localizer category from `subject_info.csv`
- load independent-scan beta maps for the relevant cue indices
- extract patterns from the binary visual union mask
- average repeated presentations of the same base cue across runs
- compute `1 - correlation` between base-cue visual patterns
- expand the base-cue RDM to the probe image-wise sample order

The resulting `visual_image` RDM is then used as a control regressor in the
probe RSA regression.

## Future combined-cue model

This mask strategy is designed to transition later to a combined
conditioned-plus-preconditioned model. In that case, the same visual union mask
can support visual RDMs spanning both A and B cues:

```text
A11, A21, A12, A22, B11, B21, B12, B22
```

The combined model should likely include an additional cue-type or image-category
control so that graded visual similarity is not confounded with the broad
character-versus-place distinction.
