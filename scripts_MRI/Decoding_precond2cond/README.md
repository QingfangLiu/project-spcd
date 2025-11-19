# 🎯 Goal 4 – Pre-conditioned → Conditioned Cue Reactivation (Odor Prediction)

Train on **Day 1 pre-conditioned cues (A1–A6)** and test on **Day 2 conditioned cues (B1–B6)** from the **odor prediction task**, using a whole-brain **8 mm searchlight** with a linear SVM.

This analysis asks whether **conditioned cues presented in the odor-prediction task** evoke neural patterns similar to their associated **pre-conditioned cues**, indicating **associative reinstatement** of A-cue representations during B-cue processing.

---

## 🗂 Data Expected

```
GLM/character_trialwise_2mm/sub-XX/
GLM/place_trialwise_2mm/sub-XX/
GLM/odorpred_cueEARLY_trialwise_2mm/sub-XX/
GLM/odorpred_cueEXT5s_trialwise_2mm/sub-XX/
nii/sub-XX/func_PAID/{character|place}/sub-XX_task-<localizer>_run-*_events.tsv
task_info/subject_info.csv
```

### **Day 1 – Pre-conditioned A cues (Training)**
- Source: `character_trialwise_2mm` **or** `place_trialwise_2mm`
- Determined by subject’s `pairing` in `subject_info.csv`
  - `c` → A cues come from *place* scan
  - `p` → A cues come from *character* scan
- From localizer events:
  - Keep only **odd-numbered cues** (`img_1, img_3, …, img_11`) → A₁…A₆
  - Map A cue to its paired B identity → **6 classes (0–5)**

### **Day 2 – Odor Prediction B cues (Testing)**
- Two possible GLM folders:
  - `odorpred_cueEARLY_trialwise_2mm`
  - `odorpred_cueEXT5s_trialwise_2mm`
- From the per-run `*_beta_index.csv`:
  - Keep only **B cues** (`cue_idx ∈ {2,4,6,8,10,12}`)
  - Label each B cue by its identity class (0–5)
- **Special case:** `sub-35` → exclude run 3 (partial scan)

---

## ⚙️ Normalization

**Leakage-safe voxelwise z-scoring**

- **Day 1 A-cue GLM:**  
  Run-wise z-scoring using `_zscore_4d_by_voxel`
- **Day 2 odor prediction B cues:**  
  Run-wise z-scoring applied separately for the EARLY and EXT5s GLM variants

No information is shared across training/testing phases.

---

## 🧩 Scripts

### `decoding_utils.py`
- `load_decoding_data_preconditioned_A_localizer(...)`  
  Loads Day 1 A-cue training data from the correct localizer type  
- `load_decoding_data_conditioned_B_odorpred(...)`  
  Loads B-cue testing data from *odor prediction* GLMs  
- Handles 6-class label mapping (A→B identity)

---

### `run_searchlight_crosstask_precond2cond_odorpred.py`
Runs single-subject decoding:

- **Train:** Day 1 A-cue betas (character/place localizer)
- **Test:** Day 2 odor-prediction B-cue betas
- **Classes:** 6 (B₁…B₆)
- **Searchlight:** 8 mm radius, linear SVM
- **Mask:** explicit GM mask (`masks/gm_0.1.nii`)
- Automatically handles `sub-35` run-3 exclusion

---

### `make_searchlight_swarm_precond2cond_odorpred.py`
Creates a Biowulf swarm file for all subjects with:

- valid Day 1 A-cue GLMs
- valid Day 2 odorpred GLM (EARLY or EXT5s)

Each swarm job:

```
python run_searchlight_crosstask_precond2cond_odorpred.py ...
```

---

### `prepare_decoding_precond2cond_odorpred_subset.py` (simple)
Builds a minimal dataset containing:

- Day 1 character + place GLMs (for A cues)
- Day 2 odorpred GLMs (EARLY or EXT5s)
- Localizer event TSVs
- Mask + subject_info

Designed to sync cleanly to Biowulf.

---

## 📈 Outputs

Searchlight maps are written to:

```
searchlight_maps/odorpred_crosstask/
  sub-XX_session-odorpred_mode-precond2cond_<glm_tag>_classes-6_radius-8mm.nii.gz
```

Each voxel = decoding accuracy for predicting B-cue identity using A-cue-trained patterns.

Chance = 1/6 ≈ 0.167.

---

## 🧠 Interpretation

Above-chance accuracy means:

- When subjects process **conditioned cues** (B) during the odor-prediction task,
- Their neural activity partially resembles the **associated pre-conditioned cues** (A),

→ indicating **associative reinstatement** and **model-based reactivation** of A-cue representations.

This provides evidence that **learned A→B associations** influence neural processing during conditioned cue presentation.

---

## 🧪 Notes

- Compatible with both **EARLY** and **EXT5s** GLM variants.
- Special handling for **sub-35** (odor-prediction run 3 excluded).
- Group-level analysis (centering, smoothing, one-sample t-test) implemented in  
  `aggregate_precond2cond_odorpred_group_level_spm.m`.
- Results include centered/smoothed maps and SPM t-maps identifying regions where associative A→B reinstatement is strongest.
