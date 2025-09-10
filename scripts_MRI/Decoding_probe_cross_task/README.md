# Cross-task odor→probe searchlight decoding

Train on **Day-1 odor scan betas** and test on **Day-2 probe betas** (cue type A or B) using a whole-brain **searchlight (8 mm)** with a linear SVM.

This analysis tests whether neural representations of odor identity generalize across **tasks** (odor perception → cue-based probe inference).

---

## Data expected

```
GLM/odor_scan_trialwise_2mm/sub-XX/
  beta_run1_trial**.nii
  beta_run2_trial**.nii
  sub-XX_task-odor_run-1_events.tsv
  sub-XX_task-odor_run-2_events.tsv

premeal_trialwise_2mm/sub-XX/
  sub-XX_premeal_run-1_beta_index.csv
  sub-XX_premeal_run-2_beta_index.csv
  sub-XX_premeal_run-3_beta_index.csv
  beta_run1_img***.nii
  beta_run2_img***.nii
  beta_run3_img***.nii
```

- **Odor events TSV:** provides `trial_type` (`odor_1`, `odor_2`, `clean_air`)  
- **Probe CSVs:** must include `beta_file`, `run`, `cue_index`, `odor_index`, `set`  
- **Cue rules:**  
  - Cue type: A = odd `cue_index`, B = even `cue_index`  
  - Labels: from `odor_index`  
    - **2-class:** keep {1,2} (odor_1 vs odor_2)  
    - **3-class:** keep {0,1,2} (clean_air vs odor_1 vs odor_2)

---

## Normalization

**Leakage-safe within-dataset normalization (voxel-wise z-score):**

- **Odor training data:** z-scored across trials within the odor dataset (Day-1 runs).  
- **Probe test data:** z-scored within each run/set (Day-2 probe).  
- Ensures no statistics are shared across train/test tasks.  

---

## Scripts

### `odor_training_utils.py`
- `load_decoding_data_odor(...)`  
  Loads Day-1 odor betas, builds labels from events TSV, applies within-dataset voxelwise z-scoring, returns 4D NIfTI + labels.

### `probe_decoding_utils.py`
- `load_decoding_data_cue(...)`  
  Loads Day-2 probe betas by cue type, applies run-wise set-wise z-scoring, returns 4D NIfTI + labels + run/set IDs.

### `run_searchlight_crosstask_odor2probe.py`
Runs whole-brain searchlight decoding for one subject.

- **Train:** odor betas (Day-1 runs 1–2)  
- **Test:** probe betas (Day-2 runs 1–3, cue type A or B)  
- **Classes:** 2-class (odor_1 vs odor_2) or 3-class (clean_air vs odor_1 vs odor_2)  
- **Searchlight:** 8 mm radius, linear SVM (C=1), user-supplied mask (e.g. GM mask)  

---

## Outputs

Saved to:
```
<base_dir>/searchlight_maps/probe_crosstask/
  sub-XX_session-<premeal|postmeal>_mode-crosstask_cue-<A|B>_classes-<2|3>_radius-8mm.nii.gz
```

- Map values = accuracy per searchlight sphere (single train/test split: odor→probe).  
- Chance levels: 2-class = 0.5, 3-class ≈ 0.33.

---

## Defaults & notes

- **Session:** currently pre-meal (`premeal_trialwise_2mm`), but script supports `postmeal`.  
- **Runs:** odor uses runs 1–2; probe uses runs 1–3 (skips missing).  
- **Normalization:** performed separately within odor vs probe datasets; avoids leakage.  
- **Performance:** `n_jobs` controls parallelism for cluster runs.  
