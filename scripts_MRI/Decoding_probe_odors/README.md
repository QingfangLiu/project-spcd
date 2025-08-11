# Cue-based cross-set searchlight decoding

Train on **Set 1** cues and test on **Set 2** cues of the **same cue type** (A or B) using a whole-brain **searchlight (8 mm)** with a linear SVM. Splits are **run-matched** by default: for each run, train on Set‑1 from the other runs and test on Set‑2 from the left‑out run. Only the **averaged accuracy map** is saved.

---

## Data expected

```
premeal_trialwise_2mm/sub-XX/
  sub-XX_premeal_run-1_beta_index.csv
  sub-XX_premeal_run-2_beta_index.csv
  sub-XX_premeal_run-3_beta_index.csv
  beta_run1_img***.nii
  beta_run2_img***.nii
  beta_run3_img***.nii
```

Each CSV must include at least:
- `beta_file` — filename of the beta image (relative to the subject folder)
- `run` — 1 / 2 / 3
- `cue_index` — 1–12
- `odor_index` — 0 = clean_air, 1 = odor_1, 2 = odor_2
- `set` — 1 (Set 1) or 2 (Set 2)

**Cue rules**
- **Cue type:** A = odd `cue_index`; B = even `cue_index`  
- **Set:** 1 → `cue_index` 1–6, 2 → `cue_index` 7–12  
- **Labels:** derived from `odor_index`  
  - **2‑class:** keep `odor_index ∈ {1,2}` (odor_1 vs odor_2)  
  - **3‑class:** keep `odor_index ∈ {0,1,2}` (odor_1 vs odor_2 vs clean_air)

---

## Normalization

**Run‑wise across trials (voxel‑wise z‑score):**
- For each run independently, compute per‑voxel mean/std across **all trials** in that run (A, B, odors, clean air).  
- Normalize trials from that run with its own stats.  
- Prevents train/test leakage and removes run‑level baselines/drift.

---

## Scripts

### `odor_decoding_utils.py`
Utilities to load and normalize trial‑wise betas for cue‑based decoding.

- `load_betas_cue(sub, run, base_dir, session_folder='premeal_trialwise_2mm', session_tag='premeal', cue_type='A'|'B', n_classes=2|3, normalize=True)`  
  Loads one run for a subject, filters A/B by `cue_index` parity, builds labels from `odor_index`, applies **run‑wise** normalization, and returns: list of NIfTI images, labels, and metadata (run/set/cue_index/odor_index/beta_path).

- `load_decoding_data_cue(sub, base_dir, runs=(1,2,3), cue_type='A'|'B', n_classes=2|3, normalize=True, session_folder='premeal_trialwise_2mm', session_tag='premeal')`  
  Wraps multiple runs. Concatenates images to a 4D NIfTI and returns: labels, `run_ids`, and `set_ids`. Skips missing runs gracefully.

### `run_searchlight_cue.py`
Runs whole‑brain searchlight decoding for one subject with flexible options.

- **Default mode: `strict` (run‑matched cross‑set)**  
  For each available run _r_:  
  - **Train:** all trials with `set==1` from runs ≠ r  
  - **Test:** all trials with `set==2` from run r  
  Saves **only the fold‑averaged accuracy map**.

- **Alternate mode: `tune_final`**  
  Tune C on `set==1` via LOGO across runs (tiny grid), refit on all `set==1`, test on all `set==2` (pooled).

- **Class setups**  
  - **2‑class (default):** odor_1 vs odor_2 (drops clean_air)  
  - **3‑class:** odor_1 vs odor_2 vs clean_air

- **Searchlight settings**  
  - Radius: **8 mm**  
  - Mask: gray‑matter mask (e.g., `gm_0.1.nii`) in the **same space** as the betas  
  - Estimator: Linear SVM (C=1 in `strict` mode; tuned in `tune_final`)

---

## Outputs

Saved to:
```
<base_dir>/searchlight_maps/cue_crossset/
  sub-XX_session-premeal_mode-<strict|tune_final>_cue-<A|B>_classes-<2|3>_radius-8mm.nii.gz
```

- Map values are **accuracy** per searchlight sphere (averaged across folds in `strict`).  
- Chance levels: **2‑class = 0.5**, **3‑class = 1/3**.

---

## Defaults & notes

- **Session:** premeal only (postmeal stored elsewhere)  
- **Runs:** checks runs 1–3; folds with missing data are skipped automatically  
- **Normalization:** run‑wise across all trial types (A/B/odors/air)  
- **Performance:** use `n_jobs` to parallelize searchlight when running on a cluster
