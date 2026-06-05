# Premeal image-wise odor-identity RSA without image identity

This is a separate variant of `rsa_imagewise_odor_identity` for testing odor
identity encoding in odor-predictive probe cues while omitting the
`image_identity` regressor. It intentionally excludes clean-air cues and does
not include an odor-value model.

## Scope

- Session: `premeal` only
- Cue subsets:
  - `conditioned_odor_only`: `B11`, `B21`, `B12`, `B22`
  - `preconditioned_odor_only`: `A11`, `A21`, `A12`, `A22`
- Clean-air-associated cues are excluded before RSA model construction.
- The current `rsa_imagewise_odor_identity` folder is unchanged.

## Models

The multiple-regression RSA model includes:

- `odor_identity_no_image`: odor1 vs odor2 identity distance
- `button_presence`: pressed vs none nuisance
- `button_side`: left vs right, with none treated as different

`image_identity`, `odor_value`, `cue_type`, `phase`, and clean-air odor-group
models are not part of this variant. Positive `odor_identity_no_image` RSA beta
values mean neural pattern dissimilarity is higher for pairs linked to different
odor identities than for pairs linked to the same odor identity, controlling for
button nuisance RDMs but not controlling for cue/image identity.

## Single-subject runs

Conditioned odor-predictive cues:

```bash
python -m rsa_imagewise_odor_identity_no_image.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --cue_subset conditioned_odor_only \
  --target odor_identity_no_image
```

Preconditioned odor-predictive cues:

```bash
python -m rsa_imagewise_odor_identity_no_image.run_rsa \
  --sub sub-01 \
  --session premeal \
  --base_dir /path/to/SPCD_data \
  --cue_subset preconditioned_odor_only \
  --target odor_identity_no_image
```

Default output location:

```text
<base_dir>/rsa_maps/probe_imagewise_odor_identity_no_image/premeal/
```

Example subject-level maps:

```text
sub-01_model-odor_identity_no_image_subset-conditioned_odor_only.nii.gz
sub-01_model-odor_identity_no_image_subset-preconditioned_odor_only.nii.gz
```

## Biowulf swarm

The swarm generator defaults to one `odor_identity_no_image` job for each
subject and each odor-only cue subset:

```bash
python scripts_MRI/rsa_imagewise_odor_identity_no_image/make_swarm_imagewise.py \
  --base_dir /data/liuq13/SPCD_RSA_imagewise_odor_identity_no_image_subset \
  --out imagewise_odor_identity_no_image_premeal.swarm \
  --center_within_run
```

Use `--split_targets` only if you also want output maps for nuisance/control
regressors.

## Group aggregation

`aggregate_rsa_group.m` runs second-level SPM one-sample tests for:

- `model-odor_identity_no_image/subset-conditioned_odor_only`
- `model-odor_identity_no_image/subset-preconditioned_odor_only`

It reads subject maps from:

```text
/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_odor_identity_no_image/premeal/
```

and writes persistent SPM folders under:

```text
/Users/liuq13/SPCD_data/rsa_maps/probe_imagewise_odor_identity_no_image/premeal/group_spm/
```
