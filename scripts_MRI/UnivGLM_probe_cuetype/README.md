# Probe Cue-Type Univariate Analysis

This folder contains a whole-brain univariate follow-up for the probe cue-type RSA finding.

The analysis tests whether cue-evoked BOLD magnitude differs between:

- `conditioned` cues: even `cue_index` values `2, 4, 6, 8, 10, 12` (`B*`, `D*`)
- `preconditioned` cues: odd `cue_index` values `1, 3, 5, 7, 9, 11` (`A*`, `C*`)

## Script

`run_probe_cuetype_univariate_group.m`

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

The analysis uses beta maps, not t-maps, because the question is about BOLD response magnitude.

## Outputs

Outputs are written under:

```text
/Users/liuq13/SPCD_data/GLM/probe_cuewise_univariate_cuetype/
```

Subject-level maps:

```text
subject_contrasts/premeal/sub-XX_session-premeal_contrast-conditioned_gt_preconditioned_mode-both.nii
subject_contrasts/premeal/sub-XX_session-premeal_contrast-preconditioned_gt_conditioned_mode-both.nii
subject_contrasts/postmeal/sub-XX_session-postmeal_contrast-conditioned_gt_preconditioned_mode-both.nii
subject_contrasts/postmeal/sub-XX_session-postmeal_contrast-preconditioned_gt_conditioned_mode-both.nii
subject_contrasts/collapsed/sub-XX_session-collapsed_contrast-conditioned_gt_preconditioned_mode-both.nii
subject_contrasts/collapsed/sub-XX_session-collapsed_contrast-preconditioned_gt_conditioned_mode-both.nii
```

Group t-maps:

```text
group_tmaps/premeal/group_session-premeal_contrast-conditioned_gt_preconditioned.nii.gz
group_tmaps/premeal/group_session-premeal_contrast-preconditioned_gt_conditioned.nii.gz
group_tmaps/postmeal/group_session-postmeal_contrast-conditioned_gt_preconditioned.nii.gz
group_tmaps/postmeal/group_session-postmeal_contrast-preconditioned_gt_conditioned.nii.gz
group_tmaps/collapsed/group_session-collapsed_contrast-conditioned_gt_preconditioned.nii.gz
group_tmaps/collapsed/group_session-collapsed_contrast-preconditioned_gt_conditioned.nii.gz
```

## Analysis Details

For each subject and session, the script averages available complete runs:

```text
mean(beta conditioned cues) - mean(beta preconditioned cues)
```

It also writes the reverse contrast:

```text
mean(beta preconditioned cues) - mean(beta conditioned cues)
```

For the collapsed analysis, each subject's premeal and postmeal contrast maps are averaged first, then the group test is run across subjects.

## Masking And Group Tests

The group analysis is whole-brain within the explicit gray-matter mask:

```text
/Users/liuq13/SPCD_data/masks/gm_0.1.nii
```

Before group testing, subject maps are masked outside gray matter, smoothed with a 6 mm FWHM kernel, and entered into SPM one-sample t-tests against zero.

The script skips `sub-19` and `sub-33`, matching the RSA group aggregation script.

Existing subject maps and group outputs are skipped rather than overwritten.

## Run

From MATLAB:

```matlab
run('/Users/liuq13/project-spcd/scripts_MRI/UnivGLM_probe_cuetype/run_probe_cuetype_univariate_group.m')
```
