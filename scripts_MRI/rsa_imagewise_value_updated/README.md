# Updated value searchlights using the original RSA pipeline

This folder only adapts preparation, Swarm generation, and group aggregation
from `rsa_imagewise`. Every job directly runs **the unchanged
`python -m rsa_imagewise.run_rsa`**. There is no additional subject runner,
preparation job, submission helper, or scheduler dependency.

The three analyses use `--session both --target odor_value` and cue subsets
`all`, `conditioned_only`, and `preconditioned_only`. They retain both cue
positions, food odors and clean air, 8 mm spheres, the original value/nuisance
RDMs, cross-run pairs including cross-phase pairs, and no within-run centering.
The existing runner drops the constant cue-type RDM in single-class subsets.
Saving only the value target still fits the full applicable regression model.

The candidate list defaults to the local clone's participant folders. Apply
the current paper exclusions (sub-19, sub-31, sub-33) and require complete finite
pre/post ratings. This currently gives **30 participants**, including sub-10
and sub-38; sub-36 lacks ratings. No participant is removed because a job
later fails. The generated `subjects.txt` records the planned group cohort.

## 1. Prepare locally

Local preparation needs pandas. Biowulf needs Python 3.10+ and the original
RSA dependencies listed in `requirements.txt`, using your existing environment.

From the repository root on the computer containing the full local dataset:

```bash
python scripts_MRI/rsa_imagewise_value_updated/prepare_rsa_subset.py \
  --full_data_dir /Users/liuq13/SPCD_data \
  --subset_dir /Users/liuq13/SPCD_value_rsa_subset
```

Replace the local paths if needed. This reuses the original subset-copy
functions to stage:

- `rsa_imagewise/`: the original execution package, including `ratings.py`,
  which the older copier omitted.
- `compiled_pleasantness_ratings.csv` from the current local clone.
- Premeal/postmeal probe event TSVs under `nii/sub-XX/func_PAID/<phase>/`.
- `subjects.txt`, `subject_selection.csv`, `missing_events.csv`, and the
  adapted `aggregate_rsa_group.m` for later local group analysis.

Review `missing_events.csv` before proceeding. The old cross-task mini-data
preparation copied odor-scan events, so the probe events may need adding.
The script copies no betas or GM mask by default: Biowulf already has them.
Optional `--copy_betas` and `--copy_mask` reuse the original copying helpers
if those inputs need staging too. Neither script contacts Biowulf.

The compiled ratings format is unchanged: `sub`, `pre_odor1`, `pre_odor2`,
`pre_clean`, `post_odor1`, `post_odor2`, and `post_clean`. Both scripts accept
`--ratings_file` and `--subjects` for explicit local inputs; use the same
arguments for preparation and Swarm generation.

## 2. Make one Swarm file locally

```bash
python scripts_MRI/rsa_imagewise_value_updated/make_swarm.py \
  --base_dir /data/liuq13/SPCD_mini \
  --out /Users/liuq13/SPCD_value_rsa_subset/value_rsa.swarm \
  --n_jobs 4
```

`--base_dir` is the **Biowulf** root shown in your screenshot; it is not
opened locally. The local output is **one Swarm file with 90 commands**:
30 participants × three cue subsets. Each command activates the established
`/data/liuq13/conda/envs/project_rsa` environment, caps numerical-library
threads at 1, changes into `SPCD_mini`, and calls the original RSA runner.
`subjects.txt` and `subject_selection.csv` are written beside the Swarm file.

Maps go to the fresh analysis location:

```text
/data/liuq13/SPCD_mini/rsa_maps/probe_imagewise_value_updated/both/
  sub-XX_model-odor_value_subset-all.nii.gz
  sub-XX_model-odor_value_subset-conditioned_only.nii.gz
  sub-XX_model-odor_value_subset-preconditioned_only.nii.gz
```

Use a fresh `--map_dir` for another analysis version: the original map writer
overwrites an existing map with the same filename when rerun. The radius,
worker count, Conda paths, and printed resource requests are configurable.

## 3. Copy and submit as before

On the local computer:

```bash
rsync -avP /Users/liuq13/SPCD_value_rsa_subset/ biowulf:/data/liuq13/SPCD_mini/
```

This merges the staged code, ratings, and events into the existing mini
dataset, preserving other remote files. It does not delete remote files.
On Biowulf:

```bash
cd /data/liuq13/SPCD_mini
mkdir -p logs_value_rsa_updated
swarm --no-comment -f value_rsa.swarm -g 16 -t 4 \
  --time 24:00:00 --logdir logs_value_rsa_updated
```

The generator also prints this command. Match `-t` to `--n_jobs`; the
16 GB/24 hour requests are initial allocations, not measured requirements.
Swarm saves ordinary stdout/stderr logs, with the participant and subset
printed by the original runner. Inspect failures and `[SKIP]` messages and
confirm both phases/all intended runs were loaded before group inference.
The original loader and nonfinite-value handling are unchanged. Sub-38's
known postmeal coverage issue still needs inspection; a successful exit
alone does not resolve that data problem.

## 4. Group analysis locally

Copy `rsa_maps/probe_imagewise_value_updated/` back to the local data folder.
Edit the paths at the top of the staged `aggregate_rsa_group.m`, then run
that script in MATLAB with SPM12, as in the original workflow. It expects
`subjects.txt` beside it and checks every planned map in all three subsets
before estimating group models. Missing jobs cause an explicit error rather
than a smaller, silently changed cohort. The same GM masking, 6 mm smoothing,
one-sample model, and positive contrast are used; full SPM models are saved
under `both/group_spm/model-odor_value/subset-<subset>/`.

Inspect the new models in SPM Results and repeat the paper's medial/lateral
OFC small-volume corrections. This does not automatically change the paper,
figures, or multiple-testing correction. The separate subset tests do not
establish a between-subset difference; that requires a paired contrast.

## Verification

```bash
python -B -m unittest discover \
  -s scripts_MRI/rsa_imagewise_value_updated -p 'test_*.py' -v
```

Tests check participant selection, event/code staging, generated commands,
and execution against direct calls to the original searchlight on synthetic
data. Real Biowulf submission and MATLAB/SPM require the other computers.
