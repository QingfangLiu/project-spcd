# SPCD Odor Decoding Pipeline

This pipeline decodes brain activity patterns for two food odors (`odor_1` vs. `odor_2`) in the SPCD study, using trial-wise GLMs, searchlight decoding, and group-level statistical analysis.

---

## Step 1: Trialwise Beta Extraction

- Single-trial GLMs generate beta maps for each odor trial.
- Labels (`odor_1` → 0, `odor_2` → 1) are extracted from `*_events.tsv`.
- Implemented in [`decoding_utils.py`](./decoding_utils.py):
  - `load_betas()`, `load_decoding_data()`
  - Three normalization options supported: none, across trials, across voxels
  - Use `prepare_decoding_subset.py` to create subject-level decoding inputs.
  
---

## Step 2: Searchlight Decoding (Subject Level)

- Linear SVM + Leave-One-Run-Out CV
- Radius: 3 voxels  
- Run via [`run_searchlight_one.py`](./run_searchlight_one.py), which performs decoding for a single subject and one normalization option
- Use `make_searchlight_swarm.py` to launch batch jobs across all subjects/options on Biowulf

---

## Step 3: Smoothing

- Applies spatial smoothing (e.g., 6mm FWHM) to individual decoding maps
- Run via `run_smooth_one.py`
- Use `make_smooth_swarm.py` to batch the smoothing jobs

---

## Step 4: Group-Level Analysis

- Smoothed decoding maps are aggregated across subjects
- One-sample t-test is used to identify regions with above-chance decoding
- Notebook: [`aggregate_searchlight_group_level.ipynb`](./aggregate_searchlight_group_level.ipynb)

---

## Output

- Individual maps: `searchlight_maps/odor_scan_2odors`
- Group-level: SPM t-maps showing above-chance decoding


---

## Normalization Notes

We tested three normalization options for trial-wise beta images:

- **none**: raw beta images without normalization  
- **across_trials**: voxel-wise z-scoring across all trials within a run  
- **across_voxels**: image-wise z-scoring across all voxels per trial

Both `none` and `across_trials` yielded clear above-chance decoding in expected olfactory regions, especially the **left piriform cortex**. In contrast, `across_voxels` severely reduced decoding accuracy, likely due to the loss of spatial pattern information critical for searchlight decoding. Therefore, we focus on the first two normalization options in group-level analyses.
