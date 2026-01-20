# RSA Searchlight Analysis for Probe Task

This code performs whole-brain searchlight Representational Similarity Analysis (RSA) on probe task data to test:
1. **Association effects**: Neural similarity reflecting learned A-B pairings during preconditioning
2. **Odor identity effects**: Neural similarity reflecting shared odor outcomes

## Overview

The analysis:
- Loads trial-wise betas for 12 conditions (A11, A21, C1, B11, B21, D1, A12, A22, C2, B12, B22, D2)
- Z-scores voxels within each run (across all trials)
- Averages trials within each condition
- Computes neural RDM (12×12) at each searchlight location using Pearson correlation
- Fits multiple regression: `neural_RDM = β₁×Association_RDM + β₂×Identity_RDM + ε`
- Outputs separate brain maps for β₁ (association) and β₂ (identity) effects

## Files

- **`rsa_utils.py`**: Core functions for data loading, RDM creation, and searchlight RSA
- **`run_rsa_searchlight.py`**: Main script to run analysis for one subject
- **`README_RSA.md`**: This file

## Data Requirements

Expected directory structure:
```
<base_dir>/
  GLM/
    premeal_trialwise_2mm/
      sub-01/
        sub-01_premeal_run-1_beta_index.csv
        sub-01_premeal_run-2_beta_index.csv
        sub-01_premeal_run-3_beta_index.csv
        beta_run1_img*.nii
        beta_run2_img*.nii
        beta_run3_img*.nii
      sub-02/
        ...
    postmeal_trialwise_2mm/
      (same structure)
  masks/
    gm_0.1.nii
```

Each CSV must contain:
- `beta_file`: filename of beta image
- `run`: run number (1, 2, 3)
- `cue_index`: 1-12
- `odor_index`: 0 (clean air), 1 (odor1), 2 (odor2)
- `set`: 1 or 2

## Condition Mapping

| Condition | cue_index | odor_index | Set | Description |
|-----------|-----------|------------|-----|-------------|
| A11 | 1 | 1 | 1 | Set1, precond, odor1 |
| A21 | 3 | 2 | 1 | Set1, precond, odor2 |
| C1 | 5 | 0 | 1 | Set1, precond, clean |
| B11 | 2 | 1 | 1 | Set1, cond, odor1 |
| B21 | 4 | 2 | 1 | Set1, cond, odor2 |
| D1 | 6 | 0 | 1 | Set1, cond, clean |
| A12 | 7 | 1 | 2 | Set2, precond, odor1 |
| A22 | 9 | 2 | 2 | Set2, precond, odor2 |
| C2 | 11 | 0 | 2 | Set2, precond, clean |
| B12 | 8 | 1 | 2 | Set2, cond, odor1 |
| B22 | 10 | 2 | 2 | Set2, cond, odor2 |
| D2 | 12 | 0 | 2 | Set2, cond, clean |

## Model RDMs

### Association Model (12×12)
Value = 1 for paired cues within same set:
- A11-B11, A21-B21, C1-D1 (Set 1)
- A12-B12, A22-B22, C2-D2 (Set 2)

All other pairs = 0

### Odor Identity Model (12×12)
Value = 1 for cues sharing the same odor:
- **Odor1**: A11, B11, A12, B12
- **Odor2**: A21, B21, A22, B22
- **Clean**: C1, D1, C2, D2

All other pairs = 0

## Usage

```bash
python run_rsa_searchlight.py \
    --sub sub-01 \
    --session premeal \
    --base_dir /data/username/SPCD_data \
    --radius 8 \
    --n_jobs 16
```

## Output

Results saved to:
```
<base_dir>/rsa_maps/probe/
  sub-01_session-premeal_model-association_radius-8mm.nii.gz
  sub-01_session-premeal_model-identity_radius-8mm.nii.gz
  sub-01_session-postmeal_model-association_radius-8mm.nii.gz
  sub-01_session-postmeal_model-identity_radius-8mm.nii.gz
  ...
```

Each map contains regression beta values at each voxel:
- **association map**: β₁ values (unique association effect, controlling for identity)
- **identity map**: β₂ values (unique identity effect, controlling for association)


## Parameters

- **Searchlight radius**: 8mm (default, matches decoding analysis)
- **Distance metric**: Pearson correlation for neural RDM
- **Normalization**: Z-score within each run (across all trials)
- **Trial averaging**: Mean across all trials per condition
- **Mask**: Gray matter mask (gm_0.1.nii)


