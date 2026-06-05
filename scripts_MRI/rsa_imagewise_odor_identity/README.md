# Premeal image-wise odor-identity RSA

This is a separate variant of `rsa_imagewise` for testing odor identity
encoding in odor-predictive probe cues. It intentionally excludes clean-air
cues and does not include an odor-value model.

## Scope

- Session: `premeal` only
- Cue subsets:
  - `conditioned_odor_only`: `B11`, `B21`, `B12`, `B22`
  - `preconditioned_odor_only`: `A11`, `A21`, `A12`, `A22`
- Clean-air-associated cues are excluded before RSA model construction.
- The current `rsa_imagewise` folder is unchanged.

## Models

The multiple-regression RSA model includes:

- `odor_identity`: odor1 vs odor2 identity distance
- `image_identity`: cue/image identity, ignoring first-vs-second presentation
  position
- `button_presence`: pressed vs none nuisance
- `button_side`: left vs right, with none treated as different

`odor_value`, `cue_type`, `phase`, and clean-air odor-group models are not part
of this variant. Positive `odor_identity` RSA beta values mean local neural
pattern dissimilarity is higher for pairs linked to different odor identities
than for pairs linked to the same odor identity, controlling for the other RDMs.

## Single-subject runs

Conditioned odor-predictive cues:

```bash
python -m rsa_imagewise_odor_identity.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --cue_subset conditioned_odor_only \
  --target odor_identity
```

Preconditioned odor-predictive cues:

```bash
python -m rsa_imagewise_odor_identity.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --cue_subset preconditioned_odor_only \
  --target odor_identity
```

Default output location:

```text
<base_dir>/rsa_maps/probe_imagewise_odor_identity/premeal/
```

Example subject-level maps:

```text
sub-01_model-odor_identity_subset-conditioned_odor_only.nii.gz
sub-01_model-odor_identity_subset-preconditioned_odor_only.nii.gz
```

## Biowulf swarm

The swarm generator defaults to one `odor_identity` job for each subject and
each odor-only cue subset:

```bash
python scripts_MRI/rsa_imagewise_odor_identity/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_odor_identity_subset \
  --out imagewise_odor_identity_premeal.swarm
```

Use `--split_targets` only if you also want output maps for nuisance/control
regressors.

## Group aggregation

`aggregate_rsa_group.m` runs second-level SPM one-sample tests for:

- `model-odor_identity/subset-conditioned_odor_only`
- `model-odor_identity/subset-preconditioned_odor_only`

It reads subject maps from:

```text
/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_odor_identity/premeal/
```

and writes persistent SPM folders under:

```text
/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_odor_identity/premeal/group_spm/
```

## Local ROI RSA

For ROI-based RSA, use `run_roi_rsa.py`. This runs the same regression model as
the searchlight analysis, but extracts one pattern matrix from a predefined ROI
mask instead of looping over searchlight spheres:

```text
neural RDM ~ odor_identity + image_identity + button_presence + button_side
```

Example using the lateral OFC ROI label:

```bash
python scripts_MRI/rsa_imagewise_odor_identity/run_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --roi_mask masks/MiddleFrontalGyrus_from_spmT_0001_rsa_cue_type.hdr \
  --roi_name lateral_ofc \
  --center_within_run \
  --save_diagnostics
```

`--roi_mask` may be an absolute path or a path relative to `base_dir`. The ROI
is intersected with `<base_dir>/masks/gm_0.1.nii` when that gray-matter mask is
available. The script writes:

```text
<base_dir>/rsa_roi/probe_imagewise_odor_identity/premeal/
  roi-lateral_ofc_subject_betas.csv
```

The CSV includes one row per subject and cue subset, with `odor_identity_beta`,
nuisance/control betas, ROI voxel counts, sample counts, and cross-run pair
counts.

With `--save_diagnostics`, the script also writes per-subject diagnostics:

```text
<base_dir>/rsa_roi/probe_imagewise_odor_identity/premeal/
  diagnostics/
    sub-15/
      conditioned_odor_only_samples.csv
      conditioned_odor_only_nan_report.csv
      conditioned_odor_only_roi_patterns.npz
      conditioned_odor_only_neural_y.csv
      preconditioned_odor_only_samples.csv
      preconditioned_odor_only_nan_report.csv
      preconditioned_odor_only_roi_patterns.npz
      preconditioned_odor_only_neural_y.csv
```

The `.npz` file contains both `raw_patterns` and `analysis_patterns`; when
`--center_within_run` is used, `analysis_patterns` is the centered matrix used
for the neural RDM. The `neural_y.csv` file contains the cross-run neural
dissimilarity vector and the corresponding model-RDM columns used in the
regression.

To summarize subject-level ROI betas:

```bash
python scripts_MRI/rsa_imagewise_odor_identity/aggregate_roi_rsa.py \
  --base_dir /Users/liuq13/SPCD_data \
  --roi_name lateral_ofc
```

This writes:

```text
<base_dir>/rsa_roi/probe_imagewise_odor_identity/premeal/
  roi-lateral_ofc_group_summary.csv
```

The summary includes one-sample tests for each cue subset and paired
`conditioned_odor_only - preconditioned_odor_only` tests. By default it excludes
`sub-19` and `sub-33`, matching the whole-brain aggregation scripts.
