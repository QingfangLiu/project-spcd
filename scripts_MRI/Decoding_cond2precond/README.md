# 🎯 Goal 1 – Pre-conditioned → Conditioned Cue Reactivation

Train on **Day 1 conditioned cues (B1–B6)** and test on **Day 2 pre-conditioned cues (A1–A6)** using a whole-brain **searchlight (8 mm)** with a linear SVM.

This analysis tests whether neural patterns evoked by *pre-conditioned cues* (A) resemble those of their *associated conditioned cues* (B), indicating associative reactivation via **model-based inference**.

---

## 🗂 Data Expected

```
GLM/character_trialwise_2mm/sub-XX/
GLM/place_trialwise_2mm/sub-XX/
GLM/premeal_trialwise_2mm/sub-XX/
nii/sub-XX/func_PAID/{character|place}/sub-XX_task-{character|place}_run-*_events.tsv
task_info/subject_info.csv
```

- **Day 1 localizer (Conditioned B cues)**  
  - Source: `character_trialwise_2mm` *or* `place_trialwise_2mm`  
  - Choice determined by `pairing` column in `subject_info.csv`:  
    - `c` → B = character
    - `p` → B = place  
  - Event TSVs (`nii/sub-XX/func_PAID/...`) contain `trial_type = img_#`  
  - Keep only **even-numbered** cues (`img_2, img_4, …, img_12`) → B₁ … B₆  
  - Labels (6 classes): `0–5`

- **Day 2 probe (Pre-conditioned A cues)**  
  - Source: `premeal_trialwise_2mm/sub-XX/` with per-run CSV  
  - Keep only **odd-numbered** cues (`cue_index ∈ {1, 3, 5, 7, 9, 11}`)  
  - Each A cue labeled by its paired B class (A₁→B₁→class 0 … A₆→B₆→class 5)

---

## ⚙️ Normalization

**Leakage-safe voxel-wise z-scoring**  

- **Day 1 localizer:** per-run z-score across B-cue trials (_within dataset_).  
- **Day 2 probe:** per-run + per-set z-score (_within dataset_) as in probe decoding.  
- Ensures no statistics are shared across train/test tasks.

---

## 🧩 Scripts

### `decoding_utils.py`
- `load_decoding_data_conditioned_B(...)` → Day 1 B-cue training set  
- `load_decoding_data_preconditioned_A(...)` → Day 2 A-cue test set  
- Handles subject-specific pairing (character vs place) and 6-class label mapping.

### `run_searchlight_crosstask_cond2precond.py`
Runs the searchlight decoding for one subject.

- **Train:** Day 1 B (localizer)  
- **Test:** Day 2 A (pre-meal probe)  
- **Classes:** 6 (B₁ – B₆)  
- **Searchlight:** 8 mm radius, linear SVM (C = 1)  
- **Mask:** e.g. `masks/gm_0.1.nii`

### `make_searchlight_swarm_cond2precond.py`
Generates a Biowulf swarm file for all subjects with both Day 1 (localizer) and Day 2 (pre-meal) data.

---

## 📈 Outputs

```
<base_dir>/searchlight_maps/cue_reactivation/
  sub-XX_session-premeal_mode-cond2precond_classes-6_radius-8mm.nii.gz
```

Each map voxel = searchlight accuracy for classifying A-cue patterns by their B-cue identities.

Chance level = 1 / 6 ≈ 0.167.

---

## 🧠 Interpretation

Above-chance accuracy indicates **associative reactivation of conditioned cue representations** during presentation of their linked pre-conditioned cues, consistent with **model-based inference**.

---

## 🧪 Notes

- Compatible with subjects using either **character** or **place** Day-1 localizers.  
- The same decoding pipeline was also applied to **post-meal** sessions (`--session_tag postmeal`).
- Group-level **smoothing and aggregation** were performed using `aggregate_searchlight_group_level.py`, which generates smoothed mean-accuracy and t-maps across subjects.  
- Preliminary findings highlight **bilateral inferior temporal cortex** as the most promising site of associative cue reactivation, with some **differences between pre-meal and post-meal sessions** to be explored.






