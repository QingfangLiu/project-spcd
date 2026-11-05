# Updated value searchlights

Run this workflow from a complete clone on the computer holding the beta
images. It calls `rsa_imagewise.searchlight.searchlight_rsa_single_subject`
directly; the existing estimator, model RDMs, ratings loader, sample loader,
and map writer are reused without changes. Older results are preserved.

The three analyses are pooled cues (`all`), conditioned B/D cues
(`conditioned_only`), and preconditioned A/C cues (`preconditioned_only`).
Each fits combined premeal/postmeal data, both cue presentations, and both
food odors plus clean air. The default radius is 8 mm (the old runner's
default; the saved group SPM does not itself record the original radius).
Patterns are not centered within run. The value model uses absolute
differences of the six jointly standardized phase-by-outcome ratings.
Image identity, button presence, and button side are nuisance regressors;
the pooled analysis also includes cue type. The constant cue-type regressor
is removed for single-class subsets by the existing code. There is an
intercept, and no odor-identity or phase regressor. Only different-run pairs
are fitted, including cross-phase pairs. Saving only the value target still
fits the full corresponding multiple-regression model.

## Prepare on the data computer

Use Python 3.10+ in a dedicated environment; do not upgrade the environment
of an already prepared run. MATLAB/SPM12 are needed only for the group stage.

```bash
python -m pip install -r scripts_MRI/rsa_imagewise_value_updated/requirements.txt
python -B scripts_MRI/rsa_imagewise_value_updated/run_searchlights.py prepare \
  --base_dir /Users/liuq13/SPCD_data \
  --output_dir /Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_value_updated_v1
```

Replace both paths for another computer. `base_dir` must contain the old
pipeline's `GLM/{premeal,postmeal}_trialwise_2mm/sub-XX/` beta images and
beta-index CSVs, `nii/sub-XX/func_PAID/{premeal,postmeal}/` events, and
`masks/gm_0.1.nii`. Preparation freezes a copy of that GM mask and a compiled
ratings table. Ratings default to the clone's
`beh_processed_subs/sub-XX/pleasantness_summary.csv`; `--ratings_file` accepts
the old wide compiled format. Do not copy only the old package with
`prepare_rsa_subset.py`: that older copier omits `ratings.py`. Use the clone.

The candidate list comes from the clone's participant folders. **Sub-10 is
included**, subject to validation. Sub-19, sub-31, and sub-33 are excluded.
Complete finite pre/post ratings are required; sub-36 currently lacks them.
The expected planned cohort is **30 participants**, including sub-38, whose
previously identified postmeal coverage problem must be examined on the
data computer. A failure is recorded, never silently converted to exclusion.
Final group n is not assumed from this planned count.

Preparation requires a fresh/empty output directory and writes
`analysis_config.json`, `subject_inclusion.csv`, `ratings.csv`, `gm_mask.nii`,
and `jobs.swarm`. Settings, source hashes, and input hashes are retained.
The runtime versions are recorded per participant; group preparation
requires identical versions across the cohort.

## Execute and inspect failures

Start with sub-10 to check the data and environment:

```bash
python -B scripts_MRI/rsa_imagewise_value_updated/run_searchlights.py run \
  --run_dir /Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_value_updated_v1 \
  --subjects sub-10
```

Then omit `--subjects sub-10` to attempt everyone. Successful participants
are skipped, and failed participants get a fresh numbered attempt that
preserves the previous logs/maps. The runner continues after participant
failures and exits nonzero if any requested participant fails. Logs and
status live under `subjects/sub-XX/attempt-001/`; each attempt includes
`run.log`, `status.json`, `input_coverage.csv`, `input_problems.json`, and,
after successful fits, `maps/sub-XX_model-odor_value_subset-<subset>.nii.gz`.
Original participant maps elsewhere are never reused implicitly.

Each participant is checked for all six runs, 24 trials/48 presentations
per run, both positions per trial, 24 presentations per cue class per run,
two repetitions of each cue identity at each position, matching events,
readable beta files, and agreement with the GM grid.
**Any nonfinite value inside the GM mask blocks the participant** and is
reported per presentation; no filling or mask changes are introduced.
This deliberately surfaces input problems that the old loader could skip
or downstream libraries could replace with zeros. A whole-mask constant
pattern also fails. These are validation failures to inspect, not automatic
scientific exclusions. Finite patterns can still be constant within an
individual sphere; the existing estimator's local behavior is retained.

Use `--n_jobs` during preparation to set parallel workers (default 4).
For Biowulf, prepare with paths valid there; `jobs.swarm` contains one command
per participant, each running all three subsets. It uses the Python
executable from preparation and absolute clone/data paths. Submit using
your usual allocation, requesting at least `n_jobs` CPUs per task; memory
requirements should be established from the initial participant. This
workflow does not submit jobs automatically. Do not run two jobs for the
same participant concurrently. An interrupted job can leave a `.running`
directory: verify that job has stopped before removing only that empty lock
directory and rerunning. No successful analysis is overwritten.

## Validate the shared cohort and run SPM

```bash
python -B scripts_MRI/rsa_imagewise_value_updated/run_searchlights.py check \
  --run_dir /Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_value_updated_v1
```

Inspect `check_report.csv`. Only a complete, validated cohort in all three
analyses produces `group_inputs.json`. No partial group model is allowed.
For unresolved failures, inspect/repair the inputs and rerun that participant.
If a scientific exclusion is later approved, prepare a fresh run with an
explicit `--subjects` list and document the reason separately; do not edit
the prepared cohort or silently drop participants at aggregation.

In MATLAB on the same computer:

```matlab
addpath('/path/to/project-spcd/scripts_MRI/rsa_imagewise_value_updated');
aggregate_group( ...
  '/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_value_updated_v1', ...
  '/path/to/spm12', ...
  '/path/to/the/python/environment/bin/python');
```

The function reruns the Python validation, then applies the old group's
GM-to-NaN masking, 6 mm smoothing, explicit GM mask, one-sample SPM model,
and positive t contrast. All three models use the identical ordered cohort.
It preserves full models and smoothed inputs at
`group_spm/subset-<subset>/SPM.mat`. Nonempty group outputs are refused, so
the old "skip existing SPM" behavior cannot retain an outdated cohort.
If a group attempt fails, preserve/move its `group_spm` directory before
retrying. This does not require recomputing subject searchlights.

For updated paper statistics, inspect each new `SPM.mat` in SPM Results and
repeat peak-level small-volume correction in the independent medial and
lateral masks used by the paper:

- `masks/OFC_clusterMap_K2_gm0p1/mask-OFC_K2_cluster1_medial_posterior_lateral_gm0p1.nii.gz`
- `masks/OFC_clusterMap_K2_gm0p1/mask-OFC_K2_cluster2_central_anterior_lateral_gm0p1.nii.gz`

The MATLAB function estimates group models; it does not automatically
calculate SVC p values, update figures, or change the manuscript. Retain
all regional results and account for the expanded testing family when
interpreting the separate cue analyses. A significant effect in one subset
alone does not establish a between-subset difference; that would require a
paired contrast, which is outside this three-analysis rerun.

## Local verification

```bash
python -B -m unittest discover \
  -s scripts_MRI/rsa_imagewise_value_updated -p 'test_*.py' -v
```

Tests use synthetic images, compare new-wrapper maps with direct calls to
the old searchlight, and check eligibility, failure reporting, retry
preservation, and common-cohort group gating. Real participant runs and
MATLAB/SPM execution require the data computer.
