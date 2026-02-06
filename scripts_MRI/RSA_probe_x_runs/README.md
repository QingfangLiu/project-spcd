# RSA Searchlight (Cross-Run, Cuewise GLM)

This repository implements a **cross-run RSA searchlight** for cuewise GLM maps
(beta or t-maps). The goal is to test whether local multivoxel patterns encode
model-predicted similarity structure (association, identity, value) while
reducing inflated similarity due to within-run autocorrelation.

---

## Key design choices

- **Samples = (run × condition)**  
  Each sample is a cuewise GLM map from a specific run and condition. Maps are
  *not* averaged across runs.

- **Cross-run neural similarity only**  
  Neural similarity between conditions is computed by correlating patterns
  across *different runs only* (ra ≠ rb). Within-run correlations are excluded.

- **Diagonal excluded from RSA statistics**  
  RSA scores are computed from the lower triangle excluding the diagonal.
  Diagonal elements (cross-run self-similarity / reliability) are not mixed
  with representational structure.

- **Optional condition subsets**
  - `all` (12 conditions)
  - `conditioned` (B/D cues)
  - `preconditioned` (A/C cues)

- **Optional pair masking**
  Restrict RSA to odor–odor pairs only (exclude clean-involving pairs).

---

## Main entry point

### `run_rsa.py`
Command-line interface for running RSA searchlight for one subject and session.

Key options:
- `--session premeal|postmeal`
- `--stat_type beta|tmap`
- `--mode both|img1only|img2only`
- `--models association identity value`
- `--rsa_method regression|spearman|kendall`
- `--cue_subset all|conditioned|preconditioned`
- `--use_pair_mask`
- `--runs 1,2,3`
- `--radius 8`

Example:
```bash
python run_rsa.py   --sub sub-01   --session premeal   --base_dir ~/SPCD_data   --stat_type tmap   --mode both   --rsa_method spearman   --models identity value   --cue_subset all   --runs 1,2,3   --radius 8
```

---

## Data flow overview

1. **Load cuewise GLM maps** (`io.py`)  
   Cuewise beta or t-maps are loaded per run using per-run index CSV files.

2. **Build 4D image** (`searchlight.py`)  
   All available (run, condition) maps are stacked into a 4D NIfTI:
   `(x, y, z, n_samples)`.
   Parallel arrays `run_ids` and `cond_ids` define the mapping from samples to
   runs and conditions.

3. **Searchlight extraction** (nilearn)  
   For each voxel center, SearchLight extracts a matrix:
   `X.shape = (n_samples, n_voxels_in_sphere)`.

4. **Cross-run neural similarity** (`neural_rdm.py`)  
   An `(n_conditions × n_conditions)` similarity matrix is built by averaging
   Pearson correlations across all run pairs with `ra ≠ rb`.

5. **RSA scoring** (`rsa_stats.py`, `estimator.py`)  
   Neural similarity is compared to model RDMs using:
   - multiple regression (beta weights), or
   - Spearman / Kendall correlation.

6. **Save outputs** (`save_maps.py`)  
   One NIfTI map per model, with filenames encoding session, input type, mode,
   model, RSA method, radius, and optional subset/mask.

---

## Scripts and their roles

- `run_rsa.py`  
  CLI entry point; parses arguments and launches analysis.

- `searchlight.py`  
  Orchestrates subject-level analysis: data loading, condition subsetting,
  model slicing, SearchLight execution.

- `io.py`  
  Loads cuewise GLM maps using per-run index CSV files.

- `models.py`  
  Defines model RDMs (association, identity, value) and canonical condition
  order.

- `masking.py`  
  Builds boolean masks over condition pairs (e.g., odor-only pairs).

- `neural_rdm.py`  
  Computes cross-run neural similarity matrices.

- `rsa_stats.py`  
  Implements regression RSA and single-model correlation RSA (diagonal excluded).

- `estimator.py`  
  scikit-learn–compatible estimator used by nilearn SearchLight.

- `save_maps.py`  
  Handles standardized output naming and NIfTI writing.

---

## Notes on diagonal values

Diagonal entries reflect **cross-run reliability**, not representational
geometry. They are explicitly excluded from RSA statistics to avoid spurious
inflation of similarity, following standard RSA practice.

If reliability is of interest, diagonal values should be analyzed separately,
not included in RSA correlations.

---

## Outputs

Output maps are saved with filenames encoding:
- session
- input type (`beta` or `tmap`)
- mode
- model name
- RSA method
- searchlight radius
- optional condition subset and pair mask

Example:
```
sub-01_session-premeal_input-tmap_mode-both_identity_spearman_radius-8mm.nii.gz
```
