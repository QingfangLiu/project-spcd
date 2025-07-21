# SPCD Odor Decoding Pipeline

This pipeline decodes brain activity patterns for two food odors (`odor_1` vs. `odor_2`) in the SPCD study, using trial-wise GLMs, searchlight decoding, and group-level statistical analysis.

---

## Step 1: Trialwise Beta Extraction

- Single-trial GLMs generate beta maps for each odor trial.
- Labels (`odor_1` → 0, `odor_2` → 1) are extracted from `*_events.tsv`.
- Implemented in [`decoding_utils.py`](./decoding_utils.py):
  - `load_betas()`, `load_decoding_data()`

---

## Step 2: Searchlight Decoding (Subject Level)

- Linear SVM + Leave-One-Run-Out CV
- Radius: 3 voxels  
- Run via [`run_searchlight_all.py`](./run_searchlight_all.py)
- Input: gray matter mask (`gm_0.1.nii`)
- Output: `searchlight_maps/sub-XX_searchlight.nii.gz`

---

## Step 3: Group-Level Analysis

- Smooth individual decoding maps (e.g., 6mm FWHM)
- Notebook: [`aggregate_searchlight_group_level.ipynb`](./aggregate_searchlight_group_level.ipynb)

---

## Output Summary

- Individual maps: `searchlight_maps/`
- Group-level: SPM t-maps showing above-chance decoding

