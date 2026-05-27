# Probe Image-Category Univariate Analysis

This folder contains a whole-brain univariate control analysis for probe image
category. It tests whether cue-evoked BOLD magnitude differs between character
and place cues.

This analysis is intended as a visual-category control for the probe cue-type
univariate analysis. It uses the same cue-wise beta maps, but groups cue indices
by image category instead of conditioned/preconditioned status.

## Script

`run_probe_imagecategory_univariate_group.m`

## Inputs

The script uses cue-wise beta maps from the existing cue-wise GLMs:

```text
/Users/liuq13/SPCD_data/GLM/premeal_cuewise_2mm/<sub>/both/
/Users/liuq13/SPCD_data/GLM/postmeal_cuewise_2mm/<sub>/both/
```

Expected beta filenames:

```text
beta_task-<session>_run-<run>_cue-<cue_index>.nii
```

The script also reads the repo-level file:

```text
/Users/liuq13/project-spcd/subject_info.csv
```

`subject_info.csv` is a truncated subject-info file copied from the full
subject-info file in the data folder. The `pairing` column is used to recover
the subject-specific image-category counterbalancing:

- `c`: conditioned/even cues are characters
- `p`: conditioned/even cues are places

With the current cue coding:

- conditioned cues are even `cue_index` values `2, 4, 6, 8, 10, 12`
- preconditioned cues are odd `cue_index` values `1, 3, 5, 7, 9, 11`

Therefore:

- if `pairing = c`, character cues are even and place cues are odd
- if `pairing = p`, character cues are odd and place cues are even

The analysis uses beta maps, not t-maps, because the question is about BOLD
response magnitude.

## Outputs

Outputs are written under:

```text
/Users/liuq13/SPCD_data/GLM/probe_cuewise_univariate_imagecategory/
```

Subject-level maps:

```text
subject_contrasts/premeal/sub-XX_session-premeal_contrast-character_gt_place_mode-both.nii
subject_contrasts/premeal/sub-XX_session-premeal_contrast-place_gt_character_mode-both.nii
subject_contrasts/postmeal/sub-XX_session-postmeal_contrast-character_gt_place_mode-both.nii
subject_contrasts/postmeal/sub-XX_session-postmeal_contrast-place_gt_character_mode-both.nii
subject_contrasts/collapsed/sub-XX_session-collapsed_contrast-character_gt_place_mode-both.nii
subject_contrasts/collapsed/sub-XX_session-collapsed_contrast-place_gt_character_mode-both.nii
```

Group t-maps:

```text
group_tmaps/premeal/group_session-premeal_contrast-character_gt_place.nii.gz
group_tmaps/premeal/group_session-premeal_contrast-place_gt_character.nii.gz
group_tmaps/postmeal/group_session-postmeal_contrast-character_gt_place.nii.gz
group_tmaps/postmeal/group_session-postmeal_contrast-place_gt_character.nii.gz
group_tmaps/collapsed/group_session-collapsed_contrast-character_gt_place.nii.gz
group_tmaps/collapsed/group_session-collapsed_contrast-place_gt_character.nii.gz
```

## Analysis Details

For each subject and session, the script averages available complete runs:

```text
mean(beta character cues) - mean(beta place cues)
```

It also writes the reverse contrast:

```text
mean(beta place cues) - mean(beta character cues)
```

For the collapsed analysis, each subject's premeal and postmeal contrast maps
are averaged first, then the group test is run across subjects.

## Masking And Group Tests

The group analysis is whole-brain within the explicit gray-matter mask:

```text
/Users/liuq13/SPCD_data/masks/gm_0.1.nii
```

Before group testing, subject maps are masked outside gray matter, smoothed with
a 6 mm FWHM kernel, and entered into SPM one-sample t-tests against zero.

The script skips `sub-19` and `sub-33`, matching the other group aggregation
scripts.

Existing subject maps and group outputs are skipped rather than overwritten.

## Run

From MATLAB:

```matlab
run('/Users/liuq13/project-spcd/scripts_MRI/UnivGLM_probe_imagecategory/run_probe_imagecategory_univariate_group.m')
```
