# SPCD Character Scan Decoding

This pipeline performs multivariate pattern analysis (MVPA) using trial-wise beta estimates to decode image identity during the character scan in the SPCD study.

---

## 📂 Folder structure

- `GLM/character_trialwise_2mm/`  
  Contains single-trial GLM outputs (beta maps) for each character image trial, smoothed at 2mm.

- `searchlight_maps/character_scan/`  
  Contains subject-level searchlight decoding maps for the character scan.

- `searchlight_maps/character_scan/group_stats/`  
  Contains group-level statistical results, including t-maps and p-maps.

---

## 🧠 Analysis summary

- **Trial-wise GLM**: Each trial is modeled with its own regressor, with motion and other nuisance regressors included.  
- **Searchlight decoding**:
  - Classifier: SVM
  - Cross-validation: Leave-one-run-out
  - Radius: 4 voxels (8mm)
  - Gray matter mask applied with threshold = 0.1
- **Multiclass decoding** across 6 unique character images.
- **Group-level** analysis:
  - Each subject’s accuracy map is smoothed (6mm FWHM).
  - One-sample t-test vs. chance level (1/6).

---

## 🧪 Notes

- Mask: Gray matter mask thresholded at 0.1.
- Only decoding trials with valid image presentations.
- Subject-wise decoding results stored as NIfTI (`sub-*_searchlight.nii.gz`).