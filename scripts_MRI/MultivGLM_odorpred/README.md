# SPCD Odor Prediction Event & GLM

---

## 1. Odor Prediction Event File Generator

**Script:** `generate_bids_events_from_design.m`

### Purpose
Converts subject-level behavioral data for the odor prediction task into BIDS-like `events.tsv` files. These capture the full trial structure (cue, prediction, delay/odor, fixation, and button press).

### Inputs
- `<subID>/odor_pred_dat.csv` containing: run, trial, cue index, odor identity, response, RT, accuracy, ITI.

### Outputs
- `sub-XX_task-odorpred_run-<r>_events.tsv` under `SPCD_data/nii/<subID>/func_PAID/odorpred/`

### Trial logic
- Trials start with a **cue stimulus**, followed by a **prediction period**.
- If responded: prediction, button, delay, odor are included.
- If missed: fixation replaces delay+odor.
- ITI duration advances the run clock.

### Columns
`onset`, `duration`, `trial_type`, `run`, `trial`, `cue_idx`, `odor_num`, `response_label`, `rt`, `acc`

---

## 2. Subject-specific Odor Index Correction

**Script:** `swap_odor_index_odorpred_sub14.m`

### Purpose
Fixes a labeling error for **sub-14**, where odor indices `1` and `2` were swapped.

### Behavior
- Reads subject’s `events.tsv` files generated in Step 1.
- Swaps odor indices (`1 ↔ 2`).
- Overwrites with corrected versions (originals saved with `_orig` suffix).

**⚠️ This step must be run *before* GLM estimation (Step 3) to ensure the corrected indices are used.**

---

## 3. Trial-wise GLM Design (Least-Squares Single-Trial)

**Script:** `run_trialwise_glms_odorpred.m`

### Goal
Estimate a **separate beta per odor event** using least-squares single-trial (LSS) GLMs.

### Model specification (per odor event *k*)
- **Regressor of interest:** target odor *k* (2 s)
- **Task regressors:** pooled other odors (2 s), prediction (3 s), delay (2 s, responded only), button (0 s sticks, responded only)
- **Nuisance regressors:** subject-specific motion/outlier regressors (z-scored)

### Inputs
- Preprocessed NIfTI runs
- Corrected `events.tsv` files (after swap if sub-14)
- Nuisance regressor text files

### Outputs
- Trial-wise betas: `beta_run-<r>_trial-<k>.nii`
- Beta index log (per run): `<subID>_odorpred_run-<r>_beta_index.csv`

### Notes
- Canonical HRF, AR(1), HPF = 128 s, 2 mm smoothing
- Cue and fixation regressors omitted (modeled implicitly)

---

## 4. Trial-wise GLM Design for Cues (Least-Squares Single-Trial)

**Scripts:**  
- `run_trialwise_glms_cueEARLY.m`  
- `run_trialwise_glms_cueEXT5s.m`

### Goal  
Estimate a **separate beta per cue presentation** (`cue_stim`) to study visual cue representations and possible associative reactivation.

### Model specification (per cue event *k*)  

**Cue EARLY (2 s window)**  
- **Regressor of interest:** `cue_target_2s` (2 s at cue onset, *t → t+2*)  
- **Task regressors:**  
  - `cue_others_2s` (all other cues, 2 s)  
  - `prediction_all` (3 s, *t+2 → t+5*)  
  - `delay_all` (2 s, responded only)  
  - `odor_all` (2 s, responded only)  
  - `fixation_miss` (4 s, missed only)  
  - `button_all` (sticks, responded only)  
- **Nuisance regressors:** subject-specific motion/outlier regressors (z-scored)

**Cue EXT5s (5 s window, no prediction regressor)**  
- **Regressor of interest:** `cue_target_5s` (5 s at cue onset, *t → t+5*)  
- **Task regressors:**  
  - `cue_others_5s` (all other cues, 5 s)  
  - `delay_all` (2 s, responded only)  
  - `odor_all` (2 s, responded only)  
  - `fixation_miss` (4 s, missed only)  
  - `button_all` (sticks, responded only)  
- **Nuisance regressors:** subject-specific motion/outlier regressors (z-scored)

### Inputs  
- Preprocessed NIfTI runs  
- Corrected `events.tsv` files (after swap if sub-14)  
- Nuisance regressor text files  

### Outputs  
- Trial-wise betas: `beta_run-<r>_trial-<k>.nii`  
- Beta index log (per run): `<subID>_odorpred_run-<r>_beta_index.csv`  

### Notes  
- Canonical HRF, AR(1), HPF = 128 s, 2 mm smoothing  
- **All cue_stim trials are modeled as targets** (responded + missed)  
- Cue betas named by **trial number** (consistent with odor GLM)  
- `cueEARLY` variant emphasizes **visual cue onset**, while `cueEXT5s` variant captures **visual + predictive processes**

---

### Workflow Summary  
1. Generate `events.tsv` for each subject/run  
2. Apply **odor index swap for sub-14** (if applicable)  
3. Run odorpred GLM 
3. Run both cue GLM variants 



