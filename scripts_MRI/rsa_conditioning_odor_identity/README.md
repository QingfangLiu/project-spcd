# Odor-identity RSA during conditioning

This workflow uses existing conditioning cue betas to test whether lateral OFC
patterns distinguish the two expected food odors. The bilateral medial OFC
cluster is analyzed as a comparison. It runs on the machine holding the MRI
data; no GLMs are re-estimated.

## Status

As of November 12, 2026, the workflow has been run on the data machine
against the actual conditioning betas. The results are committed under
`rsa_roi/conditioning_odor_identity/` and `plots/conditioning_odor_identity/`;
see [Results](#results-november-12-2026).

- Before the run, the trial loader was checked against all 34 subjects'
  real beta indexes and behavioral records. Index/behavior joins, cue labels,
  matched EARLY/EXT5s trials, subject 14's global food-label swap, and
  subject 35's interrupted run 3 all behaved as intended. Beta images, both
  OFC masks, and `<base_dir>/masks/gm_0.1.nii` share an identical grid.
- That check found two subjects who are unsuitable for this analysis:
  sub-38 and sub-26. Both were excluded at run time; see
  [Analysis-specific exclusions](#analysis-specific-exclusions).
- Plotting originally failed on the data machine's Python (< 3.10) because
  `figure_style.py` used an `int | None` annotation. It was fixed in commit
  02a5b83 with `from __future__ import annotations`. The README still
  recommends Python 3.10+.

No new GLM estimation is needed for this workflow.

## Run

From the project clone on the data machine:

```bash
python scripts_MRI/rsa_conditioning_odor_identity/run_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --save_diagnostics \
  --skip_subs sub-19 sub-31 sub-33 sub-26 sub-38
```

This is the command that produced the committed results. `--skip_subs`
replaces the default exclusion list (19/31/33) rather than adding to it, so
those three must be listed alongside the two analysis-specific exclusions.

The default runs both cue windows, both run selections, both bilateral Kahnt
OFC K=2 regions, and both RDM comparison scopes. To stop at the first unexpected
input error, add `--strict`. Without it, other subjects/configurations continue,
failures are recorded in QC, and the command exits unsuccessfully if any
unexpected failures occurred. Known exclusions are not unexpected failures.

Preview the configuration on a machine without the MRI data:

```bash
python scripts_MRI/rsa_conditioning_odor_identity/run_rsa.py \
  --base_dir /Users/liuq13/SPCD_data --dry_run
```

For just the primary lateral OFC analyses, add:

```text
--analyses primary_runs2-3 --clusters cluster2
```

`--windows cueEARLY` or `--windows cueEXT5s` selects one window. Running both
together is recommended because the loader matches their selected trials.
Use `--subjects sub-01` for an initial single-subject check.

## Inputs and trial labels

| Window | Input GLM folder under `<base_dir>/GLM/` | Modeled interval |
| --- | --- | --- |
| `cueEARLY` | `odorpred_cueEARLY_trialwise_2mm` | Cue onset to 2 s |
| `cueEXT5s` | `odorpred_cueEXT5s_trialwise_2mm` | Cue onset to 5 s, including prediction |

Each subject/run requires `<sub>_odorpred_run-<run>_beta_index.csv` with
`beta_file`, `run`, `trial`, `trial_type`, `cue_idx`, and `odor_num`. Only
`trial_type == cue` rows are used. Paths to individual betas are resolved
relative to that subject's GLM directory.

Expected odor identity comes from the beta index's `odor_num`, checked against
the behavioral table's cue/odor associations. Codes 1 and 2 identify the two
food odors within each participant; code 0 is clean air and is excluded.
Ordinarily cues 2/8 predict odor 1 and cues 4/10 predict odor 2. Cue set 1
contains indices 2/4/6 and set 2 contains 8/10/12.

The loader joins `<project>/beh_processed_subs/<sub>/odor_pred_dat.csv` by
subject, run, and trial. Accuracy, response presence, and RT are recovered
from behavior rather than beta-index accuracy/RT, which can be missing due
to the old event generator's NaN arithmetic. Inclusion requires `acc == 1`,
a recorded response in {0,1,2}, and a finite positive RT. Missed and incorrect
trials are excluded. Model labels describe the cue-associated outcome; they
are not derived from button position.

Beta and behavior cues must match. A consistent global swap of food labels
1/2 is accepted and recorded as `global_food_label_swap`, accommodating
subject 14's correction without swapping again. Such a swap leaves the
same/different-odor RDM unchanged. Mixed label inconsistencies, duplicate
keys, missing behavioral matches, and missing selected beta images raise
errors. When both windows run together, their common valid trial keys are
used in the same order and any discarded unmatched trials are counted in QC.
Matched windows must agree on cue and odor labels.

## Run selection and RSA

- **`primary_runs2-3`:** conditioning runs 2 and 3; both must be usable.
- **`sensitivity_runs1-3`:** runs 1, 2, and 3; at least two must be usable.
  Incomplete requested-run coverage is explicitly flagged.
- **Subject 35 run 3** is always excluded because the scan was interrupted.
  Subject 35 is consequently ineligible for the primary analysis, but can
  contribute runs 1/2 to sensitivity analyses.
- Subjects **19, 31, and 33** are excluded by default, matching the final
  manuscript sample. Override only deliberately with `--skip_subs`.
- Every contributing run must retain trials for both food odors.

### Analysis-specific exclusions

Two more subjects are excluded from this analysis only, by passing them in
`--skip_subs` (see [Run](#run)). In QC they appear with the same generic reason
as 19/31/33 (`prespecified_subject_exclusion`), so the specific reasons are
recorded here:

- **sub-38: no OFC coverage.** Its conditioning betas (both windows, all
  three runs) are finite only in the bottom 26 slices of the 2 mm grid
  (z indices 0–25). That is about 89k voxels, versus about 384k for other
  subjects. Across the selected runs, 0 of 4,445 lateral (cluster2) voxels
  survive, and 28 of 5,500 medial (cluster1) voxels for runs 2–3 (4 for
  runs 1–3). Without the exclusion, the lateral ROI fails and a 28-voxel
  medial estimate would enter the group statistics, because the 20%
  voxel-loss check only flags. The raw `nii/sub-38/func/odorpred` files are
  full size, so the loss arises in preprocessing or the GLM mask; this has
  not been investigated.
- **sub-26: too few correct trials.** It is a late learner, near chance in
  runs 1–2 and accurate in run 3. In the primary selection, run 2 keeps one
  correct odor-1 trial and three odor-2 trials. Of the subjects not already
  excluded, it is the only one with fewer than three correct trials for
  either odor in any contributing run. The next lowest is sub-04, with four.

Individual trial patterns are retained, with no averaging across trials or
runs. Voxels nonfinite in any selected sample are dropped; within each run,
each retained voxel is mean-centered across the selected trials. Constant
centered patterns are rejected. The neural RDM is `1 - Pearson correlation`;
the odor model is 0 for the same odor and 1 for different odors.

Two separately reported pair scopes are computed:

1. **`cross_run`:** between-run comparisons, matching the probe RSA approach.
   Repeated presentations of the same visual cue can contribute.
2. **`cross_run_cross_set`:** between-run comparisons also restricted to
   different cue sets. Every retained pair therefore uses different visual
   images; this checks generalization across cues predicting the same odor.

The analysis imports the existing probe workflow's ROI loading, pattern
preparation, correlation-distance RDM, cross-run pair mask, rank-statistic,
and participant-level group-summary functions. Conditioning-specific input
handling, pair selection, orchestration, and plotting live here. The probe
scripts are unchanged.

Spearman rho is the primary statistic. Rank-biserial correlation, AUC, and
different-minus-same mean distance are saved as related descriptive checks,
not independent confirmations. There are no inferential tests on individual
RDM pairs. Group inference treats participants as the independent units and
includes one-sample t, Wilcoxon, and sign tests from the existing helper.
The directional prediction uses the greater-than-zero p-value.

The two primary lateral OFC cross-run Spearman tests (one per window) also
receive Holm correction, in `p_primary_lateral_holm`. The planned family
remains two tests when only one window is requested or one lacks a valid
group test. Other p-values are uncorrected; medial OFC, cross-set, and
all-run results are comparison/sensitivity analyses. Separate significance
in two windows or ROIs does not establish a difference between them.

These analyses inherit the existing GLMs' timing and nuisance specification:
cue–odor timing is fixed and odor events are pooled in the cue GLMs. A positive
RSA coefficient alone does not establish complete separation of anticipated
identity from later odor receipt, or separate sensory from semantic identity.

## Outputs

Scripts, summary tables, QC, and figures stay in the project clone:

```text
<project>/scripts_MRI/rsa_conditioning_odor_identity/
<project>/rsa_roi/conditioning_odor_identity/
  run_manifest.json
  {cueEARLY,cueEXT5s}/
    {primary_runs2-3,sensitivity_runs1-3}/
      {cluster1_medial_posterior_lateral,cluster2_central_anterior_lateral}/
        subject_results.csv
        group_summary.csv
        qc.csv
<project>/plots/conditioning_odor_identity/
  <window>/<analysis>/<cluster>.{pdf,png}
```

Both pair scopes are labeled in each results table. QC includes all discovered
subjects, including prespecified exclusions and ineligible/failed cases, plus
run/trial counts, missing runs, label checks, and voxel loss. At least 20%
voxel loss is flagged without automatic exclusion. The manifest records
paths, selection settings, and discovered subjects. Figures show individual
participants and means with t-based 95% confidence intervals; no significance
stars are added.

With `--save_diagnostics`, detailed arrays and trial/RDM-pair tables stay
beside the MRI data:

```text
<base_dir>/rsa_roi/conditioning_odor_identity/diagnostics/
  <window>/<analysis>/<cluster>/<sub>/
    samples.csv
    patterns_and_rdms.npz
    rdm_pairs.csv
```

`samples.csv` gives the row order for arrays and pair-table sample indices.
Both pair masks in the NPZ follow NumPy's strict lower-triangle order.

Paths can be overridden with `--behavior_dir`, `--mask_dir`,
`--repo_output_dir`, `--plot_dir`, and `--diagnostics_dir`. Project paths are
inferred from the script location or `--project_root`, not the current working
directory. Outputs for a selected configuration are replaced on rerun.
Diagnostics folders for subjects skipped in a later run are not removed. The
November 12 diagnostics tree still holds sub-26 and sub-38 folders from an
earlier run without those exclusions, so match diagnostics to
`subject_results.csv` before using them.

## Results (November 12, 2026)

Final samples are 28 for `primary_runs2-3` (31 manuscript-sample subjects,
minus sub-26 and sub-38, minus sub-35's interrupted run 3) and 29 for
`sensitivity_runs1-3`. They are the same across both windows and both ROIs.
No subject failed, the window trial sets matched exactly, and voxel loss was
at most 4.4% of either ROI. Subject-level rho recomputed independently from
the saved diagnostic RDMs matched `subject_results.csv` to within 1e-16.

Mean Spearman rho, with the one-sided (greater-than-zero) t-test p in
parentheses:

| Window | Runs | ROI | `cross_run` | `cross_run_cross_set` |
| --- | --- | --- | --- | --- |
| cueEARLY | 2–3 | lateral (cluster2) | **0.0065 (0.243)** | −0.0060 (0.775) |
| cueEARLY | 2–3 | medial (cluster1) | 0.0054 (0.274) | −0.0037 (0.608) |
| cueEXT5s | 2–3 | lateral (cluster2) | **0.0073 (0.214)** | −0.0092 (0.833) |
| cueEXT5s | 2–3 | medial (cluster1) | 0.0064 (0.210) | −0.0048 (0.698) |
| cueEARLY | 1–3 | lateral (cluster2) | 0.0115 (0.055) | 0.0037 (0.302) |
| cueEARLY | 1–3 | medial (cluster1) | 0.0032 (0.245) | 0.0015 (0.395) |
| cueEXT5s | 1–3 | lateral (cluster2) | 0.0131 (0.027) | 0.0042 (0.276) |
| cueEXT5s | 1–3 | medial (cluster1) | 0.0068 (0.120) | 0.0042 (0.246) |

The two primary tests (bold) are not significant. Holm-adjusted p is 0.428
for both. The only uncorrected p < 0.05 is the cueEXT5s lateral all-run
sensitivity result, which disappears when pairs are restricted to different
cue sets. Cross-set estimates are near zero or negative throughout.

A post hoc decomposition, computed from the diagnostics and not produced by
`run_rsa.py`, splits the across-run pairs four ways: same cue; same odor,
other set; different odor, same set; and different odor, other set. In the
primary selection, same-cue pairs were somewhat more similar than same-odor
pairs with different images. Paired t across subjects was 1.80 and 2.00 for
lateral OFC (cueEARLY, cueEXT5s) and 2.00 and 1.84 for medial OFC. Among
pairs whose images differ, rho was −0.005 for lateral OFC in both windows
(t ≈ −0.6), or about 0.006–0.007 (t ≈ 1.4) for runs 1–3. The small positive
`cross_run` values are therefore consistent with cue-specific (image)
similarity. The data give no evidence that lateral or medial OFC carries an
expected-odor code that generalizes across cues during conditioning.

## Local validation

Requirements: Python 3.10+, NumPy, pandas, SciPy, nibabel, scikit-learn (used
by the imported established RSA module), and Matplotlib for figures.

```bash
python -m unittest discover \
  -s scripts_MRI/rsa_conditioning_odor_identity -p 'test_*.py' -v
```

Tests use temporary synthetic NIfTIs and behavioral/index tables. They cover
trial joins and filtering, subject 14's global relabeling, invalid labels,
missing/duplicate inputs, matching between windows, subject 35's interrupted
scan, voxel loss, cross-set pair selection, recovery of an injected odor
signal, multiple-testing adjustment, both ROI/window/run-selection output
trees, figures, diagnostics, and offline dry-run behavior. Experimental results
must be generated on the data machine; no synthetic outputs are saved to the
repository's results folders.
