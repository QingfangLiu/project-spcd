# Passive odor → premeal cue rectangular Spearman RSA

This pipeline tests whether odor identity is represented similarly across the
passive odor scan and the premeal probe task in the two Kahnt OFC K=2 ROIs.
It is separate from, and does not overwrite, the existing cross-task decoding
or probe-only RSA analyses.

## Analysis definition

For each subject, ROI, hemisphere, and cue subset:

1. Load odor-1 and odor-2 trialwise beta patterns from passive odor-scan runs
   1 and 2.
2. Reuse the saved first-image-only cue patterns from the probe-only K=2
   Spearman RSA diagnostics. Conditioned and preconditioned cues are analyzed
   separately; clean-air cues are excluded.
3. Keep only voxels that are finite in the passive data and every requested
   cue subset. Center each voxel within each run and within each task.
4. Concatenate passive and cue patterns and calculate a correlation-distance
   neural RDM (`1 - Pearson r`).
5. Retain only the rectangular passive-trial × cue-trial block. Within-task
   pairs are excluded.
6. Spearman-correlate these neural distances with a binary odor-identity model
   RDM (0 = same odor, 1 = different odor). A positive coefficient means that
   different-odor cross-task pairs are more dissimilar than same-odor pairs.

The script also reports rank-biserial correlation/AUC and the mean-distance
contrast. RDM cells are dependent, so the pipeline does **not** use a
pair-level p-value. Inference is performed on one coefficient per subject.
Group summaries report two-sided and directional one-sample t tests,
Wilcoxon tests, sign tests, and the paired conditioned-versus-preconditioned
contrast. `sub-19` and `sub-33` are excluded at aggregation by default, matching
the existing analyses.

## Run on the Mac mini

From the repository root:

```bash
python scripts_MRI/crosstask_rectangular_rsa/run_ofc_k2_crosstask_spearman_rsa.py \
  --base_dir /path/to/SPCD_data \
  --strict
```

This runs both K=2 ROIs, both cue subsets, and the bilateral ROI. To inspect
the exact commands without accessing the external dataset:

```bash
python scripts_MRI/crosstask_rectangular_rsa/run_ofc_k2_crosstask_spearman_rsa.py \
  --base_dir /path/to/SPCD_data \
  --dry_run
```

Useful options include:

- `--clusters cluster2` for the central/anterior-lateral OFC analysis only.
- `--hemispheres bilateral left right` for bilateral and unilateral results.
- `--subjects sub-01 sub-02` for a small test run.
- `--save_pair_diagnostics` to save every passive×cue distance and label.
- `--skip_plots` to omit figure generation.

## Files that must be available on the Mac mini

Sync this repository, including:

- `scripts_MRI/crosstask_rectangular_rsa/`
- `scripts_MRI/Decoding_probe_cross_task/`
- `scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/`
- `rsa_roi/probe_imagewise_odor_identity_no_image_first_image/`
  `premeal_ofc_k2_spearman/cluster*/diagnostics/`

The external SPCD data root must contain:

- `GLM/odor_scan_trialwise_2mm/sub-*/beta_run*_trial*.nii[.gz]`
- `nii/sub-*/func_PAID/odor_scan/sub-*_task-odor_run-*_events.tsv`
- for `sub-14`, the original premeal event files under
  `nii/sub-14/func_PAID/premeal/` (or pass
  `--sub14_original_events_dir`)

The full premeal GLM is not needed because the repository diagnostics already
contain the selected first-image ROI patterns and their voxel coordinates.

Python dependencies are NumPy, pandas, SciPy, nibabel, scikit-learn, and
Matplotlib.

## Outputs

By default, results are written below:

`rsa_roi/crosstask_passive_odor_to_premeal_cues_spearman/`

Each ROI/hemisphere scope contains:

- `subject_crosstask_spearman_rsa.csv`: one row per subject and cue subset.
- `group_crosstask_spearman_rsa_summary.csv`: subject-level inference.
- `failures.csv`: subjects that failed when `--strict` was not used.
- `pair_diagnostics/`: optional pair-level audit tables.

Plots are saved in the output root's `plots/` directory.
