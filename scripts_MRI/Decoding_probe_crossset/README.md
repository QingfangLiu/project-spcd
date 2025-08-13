# Cue-based cross-set searchlight decoding

Train on **Set 1** cues and test on **Set 2** cues (or vice versa) of the **same cue type** (A or B) using a whole-brain **searchlight (8 mm)** with a linear SVM.

Two decoding modes are supported:

- **strict** — run-matched cross-set CV: for each run, train on one set from the *other runs*, test on the opposite set from the held-out run. Controls for run-specific effects.  
- **set2set** — pooled set-to-set generalization: train on all runs of one set, test on all runs of the other set (then swap). Maximizes training data but does not control for run effects.


```  
STRICT (for each held-out run r)
  Fold A_r: Train = Set1 from runs != r  --> Test = Set2 from run r
  Fold B_r: Train = Set2 from runs != r  --> Test = Set1 from run r

SET2SET
  Fold 1: Train = Set1 (all runs)        --> Test = Set2 (all runs)
  Fold 2: Train = Set2 (all runs)        --> Test = Set1 (all runs)
```  

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

Each CSV must include:
- `beta_file` — filename of the beta image (relative to the subject folder)  
- `run` — 1 / 2 / 3  
- `cue_index` — 1–12  
- `odor_index` — 0 = clean_air, 1 = odor_1, 2 = odor_2  
- `set` — 1 (Set 1) or 2 (Set 2)  

**Cue rules**  
- **Cue type:** A = odd `cue_index`; B = even `cue_index`  
- **Set:** 1 → `cue_index` 1–6, 2 → `cue_index` 7–12  
- **Labels:** from `odor_index`:  
  - **2‑class:** keep {1,2} (odor_1 vs odor_2)  
  - **3‑class:** keep {0,1,2} (odor_1 vs odor_2 vs clean_air)

---

## Normalization

**Leakage-safe set-wise normalization (voxel-wise z‑score):**
- Within each run, compute per‑voxel mean/std **separately for each set** using all trials from that set.  
- Normalize only trials from that set using its own stats.  
- Prevents train/test leakage when evaluating cross‑set.

---

## Scripts

### `probe_decoding_utils.py`
Utilities to load and normalize trial‑wise betas for cue‑based decoding.

- `load_betas_cue(...)`  
  Loads one run for a subject, filters A/B by `cue_index` parity, builds labels from `odor_index`, applies **set-wise** normalization, and returns: list of NIfTI images, labels, and metadata (run/set/cue_index/odor_index/beta_path).

- `load_decoding_data_cue(...)`  
  Wraps multiple runs. Concatenates images to a 4D NIfTI and returns: labels, `run_ids`, and `set_ids`. Skips missing runs gracefully.

### `run_searchlight_cue.py`
Runs whole‑brain searchlight decoding for one subject with flexible options.

- **Modes:**  
  - `strict` — run-matched CV folds across sets.  
  - `set2set` — pooled set-to-set generalization.  

- **Class setups:**  
  - **2‑class (default):** odor_1 vs odor_2 (drops clean_air)  
  - **3‑class:** odor_1 vs odor_2 vs clean_air

- **Searchlight settings:**  
  - Radius: **8 mm**  
  - Mask: gray‑matter mask (e.g., `gm_0.1.nii`) in the **same space** as the betas  
  - Estimator: Linear SVM (C=1 for both modes)

---

## Outputs

Saved to:
```
<base_dir>/searchlight_maps/probe_crossset/
  sub-XX_session-premeal_mode-<strict|set2set>_cue-<A|B>_classes-<2|3>_radius-8mm.nii.gz
```

- Map values are **accuracy** per searchlight sphere (averaged across folds in `strict` or across swaps in `set2set`).  
- Chance levels: **2‑class = 0.5**, **3‑class ≈ 1/3**.

---

## Defaults & notes

- **Session:** premeal only (postmeal stored elsewhere)  
- **Runs:** checks runs 1–3; folds with missing data are skipped automatically  
- **Normalization:** set‑wise within each run; avoids leakage  
- **Performance:** use `n_jobs` to parallelize searchlight on a cluster

---

## Summary of findings

### 1. Set2set – Cue B – 3 classes
- **Extensive activation** across frontal, parietal, and temporal cortices.
- Strong bilateral clusters in **orbitofrontal cortex (OFC)**, particularly ventral/lateral OFC.
- Additional engagement in dorsomedial prefrontal cortex, posterior cingulate/precuneus, lateral parietal, and ventral temporal regions.
- OFC signal is broad and symmetrical across hemispheres.

### 2. Strict – Cue B – 3 classes
- **More focal** activation compared to Set2set.
- Frontal activation is smaller and less extensive in OFC, but still present in right lateral/ventrolateral regions.
- Clusters in posterior cingulate, ventral temporal, and occipital cortex.
- Reduced coverage suggests stricter run-matching controls reduced variance but also statistical power.

### 3. Set2set – Cue A – 3 classes
- Widespread activation across frontal, temporal, and parietal regions.
- OFC activation present, especially in right ventrolateral OFC, but **less bilateral symmetry** compared to Cue B.
- Strong midline activity (medial prefrontal/anterior cingulate) and large clusters in ventral temporal cortex.
- More posterior activity (parietal, occipital) than Strict Cue A.

### 4. Strict – Cue A – 3 classes
- **Least extensive** activation pattern among the four.
- OFC involvement is minimal; only a small right ventrolateral cluster may overlap OFC.
- Strongest activity in medial prefrontal/anterior cingulate, right ventral temporal cortex, and cerebellum.
- More focal posterior activations.

---

**Overall pattern:**
- **Set2set** mode yields **more widespread and robust activation**, especially in OFC and distributed associative networks, but doesn’t control for run-specific effects.
- **Strict** mode produces **more conservative, focal maps**, with reduced OFC engagement.
- Cue B generally shows stronger and more bilateral OFC activation than Cue A in both modes.


