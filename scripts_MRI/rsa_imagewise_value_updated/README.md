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

## Generate one Swarm file on Biowulf

From the repository clone on Biowulf, run:

```bash
python -B scripts_MRI/rsa_imagewise_value_updated/make_swarm.py
```

This uses the previously configured private Conda environment
`/data/liuq13/conda/envs/project_rsa` and data root **`/data/liuq13`** (the
parent of `/data/liuq13/GLM/`). It initializes Conda from
`/data/liuq13/conda/etc/profile.d/conda.sh`, checks Python 3.10+ and the RSA
dependencies there, and calls the existing preparation command below.
It does not install or upgrade packages.

The default destination is
`/data/liuq13/rsa_maps/probe_imagewise_value_updated_v1/`, which must be
fresh/empty. It writes **one `jobs.swarm` for the entire eligible cohort**,
with the same handling for sub-10 as everyone else. Each participant's job
runs pooled, conditioned-only, and preconditioned-only analyses in sequence.
Every command explicitly activates the Conda environment and sets
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` to 1, while the searchlight uses four workers by default.
No separate pilot file is generated.

The script creates `swarm_logs/`, records the submission settings and hashes
in `submission_config.json`, and prints the exact command for **you to submit**:

```bash
swarm -f /data/liuq13/rsa_maps/probe_imagewise_value_updated_v1/jobs.swarm \
  -g 16 -t 4 --time 24:00:00 \
  --logdir /data/liuq13/rsa_maps/probe_imagewise_value_updated_v1/swarm_logs
```

It never submits Swarm jobs or runs the analyses during generation. The
16 GB/24 hour defaults are initial allocation requests, not benchmarked
requirements for these data. Set them to your intended allocation when
generating; `-t` always matches `--n_jobs`:

```bash
python -B scripts_MRI/rsa_imagewise_value_updated/make_swarm.py \
  --base_dir /data/liuq13 \
  --output_dir /data/liuq13/rsa_maps/probe_imagewise_value_updated_v2 \
  --n_jobs 4 --mem_gb 24 --time 48:00:00
```

`--conda_root`, `--conda_env`, `--ratings_file`, `--subjects`, and `--radius`
are also configurable. Defaults use the current clone's participant ratings
and the existing paper exclusion rules. Keep the clone and environment in
place throughout execution. After the Swarm finishes, use `check` and the
SPM group stage below.

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

For execution without Swarm, run all planned participants with:

```bash
python -B scripts_MRI/rsa_imagewise_value_updated/run_searchlights.py run \
  --run_dir /Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_value_updated_v1
```

An optional `--subjects sub-XX ...` restricts execution to selected planned
participants. Successful participants are skipped, and failed participants
get a fresh numbered attempt that
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
For Biowulf, use `make_swarm.py` above to add the Conda setup and thread caps
to every command. The generic `prepare` command writes plain Python commands;
it is useful outside Biowulf or when you manage the environment yourself.
Both use absolute clone/data paths and the target Python executable.
This workflow does not submit jobs automatically. Do not run two jobs for the
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
preservation, common-cohort group gating, and Swarm environment/CPU setup.
Real participant runs and
MATLAB/SPM execution require the data computer.
