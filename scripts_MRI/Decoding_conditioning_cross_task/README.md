# Passive odor → conditioning cue decoding

Train the manuscript's two-odor classifier on passive odor delivery and test
whether it generalizes to expected odor identity during conditioning. This
complements the within-conditioning RSA: it tests transfer of the passive odor
discrimination to cue-evoked patterns. The agreed focus is **EXT5s in lateral
OFC**, with EARLY and medial OFC reported as comparisons.

## Run on the data machine

From the repository root, with the existing analysis Python environment:

```bash
python scripts_MRI/Decoding_conditioning_cross_task/run_decoding.py \
  --base_dir /Users/liuq13/SPCD_data \
  --strict
```

Change `--base_dir` to the MRI data location on that machine. All other paths
default to this checkout. This reads existing single-trial estimates; it does
not rerun GLMs or change input data. Dependencies are NumPy, pandas, nibabel,
SciPy, scikit-learn and Matplotlib, as used by the existing ROI analyses.

To inspect settings without MRI data or writing output, add `--dry_run`.
For an initial check, use `--subjects sub-01 sub-14` and **separate**
`--output_dir /tmp/conditioning-decoding-check/results` and
`--plot_dir /tmp/conditioning-decoding-check/plots`. Then run the command above
without `--subjects` to obtain the full cohort. A restricted subject run
replaces the selected folders' participant and group tables; it does not append
to previous tables.

## Defaults and analysis details

| Setting | Default |
| --- | --- |
| Passive training | Runs 1–2; 10 trials per food odor per run, 40 total |
| Conditioning test | All available runs 1–3; at least two usable runs |
| Test trials | Correct food-odor predictions with a recorded response and finite positive RT |
| Cue windows | `cueEARLY`: onset–2 s; `cueEXT5s`: onset–5 s, including prediction |
| ROIs | Bilateral Kahnt K2 lateral (`cluster2`) and medial (`cluster1`) OFC |
| Classifier | Linear SVC, fixed `C=1`; no tuning or feature selection based on decoding performance |
| Normalization | Voxelwise z scores across selected trials within each task, window and run |
| Primary measure | Participant balanced accuracy; chance = 0.5 |
| Group test | One-sided one-sample t test against 0.5; uncorrected |

EARLY and EXT5s use the intersection of eligible `(run, trial)` keys, in the
same order. Every contributing conditioning run must contain both food odors.
Clean-air, incorrect and missed trials are excluded using the keyed processed
behavioral data, rather than the potentially empty response/accuracy fields in
the GLM beta indexes. Known scan exclusion: **sub-35 run 3** is always omitted;
that participant can contribute runs 1–2.

The default participant exclusions match the retained conditioning RSA cohort:
sub-19, sub-31 and sub-33 (manuscript sample exclusions), sub-26 (sparse correct
conditioning trials), and sub-38 (loss of conditioning OFC coverage). That RSA
cohort had n = 29; the decoding sample is determined again after checking the
passive inputs and cross-task coverage. **Do not assume n = 29 before checking
the saved tables.** `--skip_subs` replaces the entire default list; a bare
`--skip_subs` disables participant exclusions but retains the sub-35 scan rule.

For each participant/ROI, retain only voxels finite across all selected passive
and conditioning trials, jointly across both cue windows. Standardize passive
runs once and each conditioning window/run independently; use population SD
(`ddof=0`), with SD below `1e-8` replaced by 1, as in the probe decoder.
Fit one classifier on all 40 passive samples and apply that same fitted model
to both windows. No conditioning labels enter classifier fitting or tuning.
Finite-voxel loss and loss of at least 20% of ROI voxels are recorded in QC;
the 20% flag does not automatically exclude a participant.

Balanced accuracy averages recall for the two odors, allowing unequal numbers
of correct conditioning trials. Also save ordinary accuracy, ROC AUC, and mean
signed distance from the classifier boundary (decision value divided by the
classifier weight norm). Leave-one-passive-run-out decoding is a diagnostic;
it does not determine inclusion. The shared probe summarizer additionally
reports two-sided t, Wilcoxon and sign tests. Each participant contributes one
score per ROI/window. P values are uncorrected; the EARLY/medial comparisons
and auxiliary metrics should be identified as such. A difference in
significance between ROIs or windows is not itself a test of their difference.

`--windows cueEXT5s` or `--clusters cluster2` restricts the analysis. Run both
windows together for matched trial and voxel sets: separate single-window
runs can use different trials and voxels. Use a different output directory
when comparing alternative settings.

## Inputs and odor labels

Under `--base_dir`:

```text
GLM/odor_scan_trialwise_2mm/<sub>/beta_run<r>_trial<NN>.nii[.gz]
nii/<sub>/func_PAID/odor_scan/<sub>_task-odor_run-<r>_events.tsv
GLM/odorpred_cueEARLY_trialwise_2mm/<sub>/*_beta_index.csv + betas
GLM/odorpred_cueEXT5s_trialwise_2mm/<sub>/*_beta_index.csv + betas
masks/gm_0.1.nii
```

From the repository:

```text
beh_processed_subs/<sub>/odor_pred_dat.csv
masks/OFC_clusterMap_K2_gm0p1/mask-OFC_K2_cluster*_gm0p1.nii.gz
```

Passive beta numbering follows **all rows** in each events TSV before filtering
out clean air. Passive `odor_1` and `odor_2` become class labels 0 and 1.
Conditioning trials are joined to behavior by `(run, trial)`, and cue/odor
consistency and image grids are checked. Normally conditioning `odor_num`
1/2 maps to the same class labels and must agree with behavior.

**Subject 14 needs special care.** Unlike RSA, cross-task decoding is sensitive
to a global label reversal. The passive event generator swaps this subject's
raw design odor labels; the conditioning event correction script additionally
swaps the behavior-derived conditioning labels. Following the manuscript's
probe decoder convention, use the **original conditioning events before that
additional GLM-event swap**:

```text
<base_dir>/nii/sub-14/func_PAID/odorpred/
  sub-14_task-odorpred_run-<r>_events_orig.tsv
```

The script selects `cue_stim` rows, joins by `(run, trial)`, checks `cue_idx`,
and requires the original odor labels to agree with the processed behavior.
It saves both beta-index and physical test labels. Missing originals,
duplicate/missing keys or inconsistent labels fail QC; the code never chooses
a label swap by maximizing decoding. Use `--sub14_original_events_dir` if the
backups are elsewhere. Before the full run, verify those backups exist and
are the originals created by
`scripts_MRI/MultivGLM_odorpred/swap_odor_index_odorpred_sub14.m`.
Do not regenerate or overwrite event files to get past a label error.

These are the same EARLY/EXT5s cue estimates used for conditioning RSA;
their existing GLMs include odor-delivery regressors. This decoder does not
change those models. Transfer would support shared odor-related information
across tasks, subject to the interpretation of the existing cue estimates.

## Saved output and handoff

Everything new is inside this repository by default:

```text
roi_decoding/conditioning_crosstask_odor/
  run_manifest.json
  {cueEARLY,cueEXT5s}/
    {cluster1_medial_posterior_lateral,cluster2_central_anterior_lateral}/
      subject_results.csv
      group_summary.csv
      qc.csv
      predictions/<sub>/
        conditioning_predictions.csv
        passive_cv_predictions.csv
plots/conditioning_crosstask_odor/
  {cueEARLY,cueEXT5s}/ofc_conditioning_decoding.{png,pdf}
```

Read the `cue_balanced_accuracy` row of `group_summary.csv`, specifically
`n`, `mean`, `t`, `df` and `p_greater_than_null`, for the primary group result.
The figures show participant scores, the mean and 95% t confidence interval.
There is no automatic manuscript update.

The manifest records settings, discovered subjects, failures and run status.
QC records exclusions, ineligibility, unexpected failures, missing/matched
trial counts, label source and voxel loss. `--strict` stops at the first
unexpected participant/ROI failure and preserves the available tables/QC.
Without it, the script continues to inspect other participants but still exits
with an error if any unexpected failure occurred. Fewer than two usable runs
or a run without both odors is recorded as ineligible. No usable results in
a selected ROI/window also makes the run fail. Plot failures are recorded too.

**Before interpreting or syncing:** require `run_status: complete`, review
every selected `qc.csv`, and confirm participant counts and sub-14 label
source in `subject_results.csv`. Participant/group tables and the current
manifest are authoritative: prediction files from an earlier invocation can
remain for participants excluded or failed on a rerun. Use a new output
directory when retaining a separate earlier analysis. Sync the result and
plot directories back along with this code.

The implementation imports passive loading, normalization, SVC metrics,
passive-run CV and group statistics from `Decoding_probe_cross_task`, and
conditioning loading, matching, eligibility and ROI extraction from
`rsa_conditioning_odor_identity`. Keep those existing folders when copying
the workflow to the data machine. No existing probe/RSA scripts or outputs
are changed.

Local validation uses synthetic NIfTI images, event files and behavior tables:

```bash
python -m unittest discover \
  -s scripts_MRI/Decoding_conditioning_cross_task -p 'test_*.py' -v
```

These checks cover known cross-task signal, subject-14 label orientation,
common finite voxels, independent normalization, passive event numbering,
invalid/missing inputs, sub-35's scan exclusion, group summaries and plots,
and failure persistence. Experimental MRI decoding must run on the data
machine; synthetic scores are not experimental results.
