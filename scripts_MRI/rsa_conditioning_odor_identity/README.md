# Odor-identity RSA during conditioning

This workflow uses existing conditioning cue betas to test whether lateral OFC
patterns distinguish the two expected food odors. The bilateral medial OFC
cluster is analyzed as a comparison. It runs on the machine holding the MRI
data; no GLMs are re-estimated.

## Handoff status and next steps

Status as of November 12, 2026:

- Implementation is complete locally. All **14 synthetic tests passed**,
  including end-to-end CSV, diagnostic, and figure generation for both
  windows, both ROIs, and both run selections.
- The MRI images and actual conditioning beta indexes were **not available
  on the development machine**. No experimental MRI RSA results have been
  generated or interpreted by this workflow yet.
- A separate behavior-only audit checked the 31 final-sample participants
  using temporary indexes constructed from their behavioral records. It
  found 30 potentially eligible participants for runs 2/3; subject 35 lacks
  a usable third scan. Subject 14's global food-label swap was accepted as
  intended. This audit does not verify the actual MRI indexes, images,
  alignment, coverage, or final imaging sample size.

For the agent taking over on the data machine:

1. Read the project's root `AGENTS.md` and `CLAUDE.md`, then this README.
   Work from an updated full project clone: the new runner imports the
   existing probe RSA modules and `figure_style.py`, and uses the tracked
   OFC masks and behavioral records. Copying the runner alone is insufficient.
2. Confirm the actual MRI data root and Python dependencies. Run the local
   tests in the validation section and preview settings with `--dry_run`.
3. Run the command below with `--subjects sub-01 --strict` added for a first
   data-backed check. Keep both windows enabled to verify matched trial
   selection. Check index-to-behavior labels, correct-trial counts, mask/beta
   alignment, retained voxels, and the saved trial/RDM-pair diagnostics.
4. Run the full sample. Review every `qc.csv`, including ineligible/failed
   subjects, unmatched trials, missing runs, global label swaps, and voxel
   loss. Subject 35's primary exclusion and sensitivity runs 1/2 are expected.
   Resolve unexpected failures before interpreting group summaries.
5. Report final sample sizes separately by window, run selection, and ROI,
   along with Spearman estimates and the corresponding group p-values.
   Distinguish the two primary lateral-OFC tests from comparison/sensitivity
   results. Record any data-backed fixes and rerun the affected checks.

Agreed analysis settings are specified below. The immediate handoff task is
to validate and execute this conditioning RSA with the existing EARLY/EXT5s
betas. No new GLM estimation is needed for the implemented workflow.

## Run

From the project clone on the data machine:

```bash
python scripts_MRI/rsa_conditioning_odor_identity/run_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --save_diagnostics
```

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
