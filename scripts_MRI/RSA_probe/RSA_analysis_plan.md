# Final RSA Analysis Plan

## 1. Data Preparation

- Extract single-trial beta estimates from probe task (pre-meal and post-meal separately)
- **12 conditions**: A11, A21, C1, B11, B21, D1, A12, A22, C2, B12, B22, D2
- **Z-score normalization**: Within each run, z-score each voxel across trials
- **Average trials**: Within each condition (12 conditions per participant)
- Output: (n_participants × 12_conditions × n_voxels) for pre-meal and post-meal separately

## 2. Create Model RDMs (12×12)

### Association Model RDM

- Value = 1 for paired cues within same set:
  - A11-B11, A21-B21, C1-D1, A12-B12, A22-B22, C2-D2 (6 pairs)
- Value = 0 for all other pairs (66 - 6 = 60 pairs in lower triangle)

### Odor Identity Model RDM

- Value = 1 for all same-odor pairs:
  - **Sweet** (6 pairs): A11-A12, A11-B11, A11-B12, A12-B11, A12-B12, B11-B12
  - **Savory** (6 pairs): A21-A22, A21-B21, A21-B22, A22-B21, A22-B22, B21-B22
  - **Clean air** (6 pairs): C1-C2, C1-D1, C1-D2, C2-D1, C2-D2, D1-D2
  - Total: 18 same-odor pairs
- Value = 0 for different-odor pairs (66 - 18 = 48 pairs)

Both model RDMs will be represented as similarity (1 = similar, 0 = dissimilar)

## 3. Searchlight RSA

### Searchlight parameters

- Radius: 8mm
- Whole brain (gray matter mask)

### For each searchlight sphere

1. Extract neural patterns for 12 conditions (averaged across trials, z-scored)
2. Compute neural RDM (12×12):
   - Use **Pearson correlation** between condition patterns
   - Keep as similarity matrix (correlation values)
3. Vectorize lower triangle (66 values) of:
   - Neural RDM
   - Association model RDM  
   - Odor Identity model RDM
4. Fit multiple regression:
   ```
   neural_RDM_vector = β₁ × Association_RDM_vector + β₂ × Identity_RDM_vector + intercept + ε
   ```
5. Store β₁ and β₂ values for this searchlight center voxel

## 4. Individual Subject Output

For each participant, create two 3D brain maps:
- **β₁ map**: Association effect (controlling for identity)
- **β₂ map**: Odor identity effect (controlling for association)

Run this separately for:
- Pre-meal probe data
- Post-meal probe data

## 5. Group-Level Analysis

For each of β₁ and β₂ maps (pre-meal and post-meal separately):
1. Smooth individual maps with 6mm FWHM Gaussian kernel
2. One-sample t-test across participants: test if β > 0
3. Apply cluster-based correction or FWE correction for multiple comparisons
4. Threshold and visualize significant clusters

## 6. Final Outputs

Four statistical brain maps:
1. Pre-meal association effects (β₁)
2. Pre-meal identity effects (β₂)
3. Post-meal association effects (β₁)
4. Post-meal identity effects (β₂)

---

## Implementation Notes

- Use Python with: nilearn (searchlight), scipy (regression/stats), nibabel (nifti I/O)
- For efficiency: parallelize searchlight across voxels
- Save intermediate outputs (individual β maps) before group stats
