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

GLM/premeal_trialwise_2mm/sub-XX/
  sub-XX_premeal_run-1_beta_index.csv
  sub-XX_premeal_run-2_beta_index.csv
  sub-XX_premeal_run-3_beta_index.csv
  beta_run1_img***.nii
  beta_run2_img***.nii
  beta_run3_img***.nii
```

- **Odor events TSV:** provides `trial_type` (`odor_1`, `odor_2`, `clean_air`)  
- **Probe CSVs:** must include `beta_file`, `run`, `cue_index`, `odor_index`, `set`, `img_index`
- **Cue rules:**  
  - Cue type: A = odd `cue_index`, B = even `cue_index`  
  - Labels: from `odor_index`  
    - **2-class:** keep {1,2} (odor_1 vs odor_2)  
    - **3-class:** keep {0,1,2} (clean_air vs odor_1 vs odor_2)

---

## Normalization

**Leakage-safe within-dataset normalization (voxel-wise z-score):**

- **Odor training data:** z-scored across trials within each run.  
- **Probe test data:** z-scored within each run/set.  
- Ensures no statistics are shared across train/test tasks.
- Consistent with normalization procedure when decoding within each task. 

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

### First-image-only searchlight variant

`run_searchlight_crosstask_odor2probe_first_image.py` is the version to use
when rerunning this analysis. It keeps only `img_index == 1` from every probe
trial. The filter is applied before run/set normalization, so second-image
betas affect neither the test samples nor the probe normalization statistics.

The original runner and loader remain unchanged as the prior two-image
analysis. The first-image runner uses the dedicated
`probe_decoding_utils_first_image.py` loader and therefore has no switch back
to the two-image behavior.

Generate the first-image AB, three-class Biowulf jobs with:

```bash
python scripts_MRI/Decoding_probe_cross_task/make_searchlight_swarm_crosstask_first_image.py \
  --base_dir /data/liuq13/SPCD_mini \
  --script_name scripts_MRI/Decoding_probe_cross_task/run_searchlight_crosstask_odor2probe_first_image.py \
  --n_jobs 4
```

On Biowulf, submit with `-t 4` so the allocated CPUs match `--n_jobs 4` in
each generated command.

For group analysis, use
`aggregate_searchlight_group_level_spm_crosstask_first_image.m`. It searches
only for first-image map names and writes group results beneath an
`images-first` analysis directory.

---

## Outputs

Saved to:
```
<base_dir>/searchlight_maps/probe_crosstask/
  sub-XX_session-<premeal|postmeal>_mode-crosstask_cue-<A|B|AB>_classes-<2|3>_radius-8mm.nii.gz
  sub-XX_session-<premeal|postmeal>_mode-crosstask_cue-<A|B|AB>_classes-<2|3>_images-first_radius-8mm.nii.gz
```

The first pattern is the original two-image output. The `_images-first_`
pattern is produced only by the new first-image runner, so existing maps are
never overwritten.

- Map values = accuracy per searchlight sphere (single train/test split: odor→probe).  
- Chance levels: 2-class = 0.5, 3-class ≈ 0.33.

---

## Defaults & notes

- **Session:** currently pre-meal (`premeal_trialwise_2mm`), but script supports `postmeal`.  
- **Runs:** odor uses runs 1–2; probe uses runs 1–3 (skips missing).  
- **Normalization:** performed separately within odor vs probe datasets; avoids leakage.  
- **Performance:** `n_jobs` controls parallelism for cluster runs.  

---

## Findings
- **No lateral OFC clusters** detected.  
- **Fusiform gyrus** appeared consistently, with additional effects in **posterior OFC** and **piriform cortex**.  
- **Effects modest overall**; some emerging trends but not strongly robust.  

---

## Next Steps
- **Cue collapsing:** Combine cue A and cue B to increase statistical power, assuming both elicit odor outcome expectations similarly.  
- **Expanded decoding:** Add an option for decoding **odors (odor_1 or odor_2) vs non-odors**.
- **Temporal dynamics:** Examine time-course of decoding accuracy, restricted to **pre-selected ROIs**.  

---

## ROI passive odor to first-image cue decoding

`run_roi_crosstask_odor2preconditioned.py` is the ROI-only validation of
the first-image odor-identity RSA. It trains a linear SVM on passive
odor-smelling trials and tests the exact premeal cue samples saved in one
ROI's RSA diagnostics.

The default test set is `preconditioned_odor_only`. The conditioned subset
can be added as a secondary comparison.

### Inputs

- Passive events:
  `<base_dir>/nii/<sub>/func_PAID/odor_scan/<sub>_task-odor_run-<run>_events.tsv`
- Passive trial betas:
  `<base_dir>/GLM/odor_scan_trialwise_2mm/<sub>/beta_run<run>_trial<NN>.nii`
- One ROI's saved RSA diagnostics:
  `<diagnostics_dir>/<sub>/preconditioned_odor_only_{samples.csv,roi_patterns.npz}`
- For `sub-14`, original premeal events:
  `<base_dir>/nii/sub-14/func_PAID/premeal/sub-14_task-premeal_run-<run>_events_orig.tsv`

The saved RSA voxel coordinates define the ROI, so the decoder does not
need a separate mask argument. It uses `raw_patterns` from the diagnostic
NPZ and applies decoding-specific normalization.

### Subject analysis

Example for the primary central/anterior-lateral OFC K=2 ROI:

```bash
python scripts_MRI/Decoding_probe_cross_task/run_roi_crosstask_odor2preconditioned.py \
  --base_dir /path/to/SPCD_data \
  --diagnostics_dir /path/to/project-spcd/rsa_roi/probe_imagewise_odor_identity_no_image_first_image/premeal_ofc_k2_spearman/cluster2_central_anterior_lateral/diagnostics \
  --roi_name OFC_K2_cluster2_central_anterior_lateral \
  --save_predictions \
  --strict
```

Run the comparison ROI by changing the diagnostics directory and ROI name
to `cluster1_medial_posterior_lateral` and
`OFC_K2_cluster1_medial_posterior_lateral`.

To include conditioned cues as a secondary comparison:

```bash
  --cue_subsets preconditioned_odor_only conditioned_odor_only
```

### Left- and right-hemisphere decoding

The bilateral RSA diagnostics can also be decoded separately by hemisphere.
No new RSA extraction is required: the decoder uses the saved raw pattern
matrix and voxel indices, transforms those indices with each subject's
passive-beta affine, and reruns all decoding preprocessing after selecting:

- left hemisphere: MNI `x < 0`;
- right hemisphere: MNI `x > 0`;
- bilateral: all ROI voxels, which remains the default.

Voxels at MNI `x = 0` are excluded from both unilateral analyses. Left/right
runs automatically append `_left` or `_right` to the ROI label and output
filename, preventing them from overwriting the bilateral results.

From the project clone on the data computer, define:

```bash
PROJECT_ROOT="$PWD"
SPCD_DATA="/path/to/SPCD_data"
```

Run both hemispheres for medial/posterior-lateral OFC:

```bash
for HEMISPHERE in left right; do
  python scripts_MRI/Decoding_probe_cross_task/run_roi_crosstask_odor2preconditioned.py \
    --base_dir "$SPCD_DATA" \
    --diagnostics_dir "$PROJECT_ROOT/rsa_roi/probe_imagewise_odor_identity_no_image_first_image/premeal_ofc_k2_spearman/cluster1_medial_posterior_lateral/diagnostics" \
    --roi_name OFC_K2_cluster1_medial_posterior_lateral \
    --hemisphere "$HEMISPHERE" \
    --cue_subsets preconditioned_odor_only conditioned_odor_only \
    --save_predictions \
    --strict
done
```

Run both hemispheres for central/anterior-lateral OFC:

```bash
for HEMISPHERE in left right; do
  python scripts_MRI/Decoding_probe_cross_task/run_roi_crosstask_odor2preconditioned.py \
    --base_dir "$SPCD_DATA" \
    --diagnostics_dir "$PROJECT_ROOT/rsa_roi/probe_imagewise_odor_identity_no_image_first_image/premeal_ofc_k2_spearman/cluster2_central_anterior_lateral/diagnostics" \
    --roi_name OFC_K2_cluster2_central_anterior_lateral \
    --hemisphere "$HEMISPHERE" \
    --cue_subsets preconditioned_odor_only conditioned_odor_only \
    --save_predictions \
    --strict
done
```

Aggregate all four unilateral result files:

```bash
for ROI in \
  OFC_K2_cluster1_medial_posterior_lateral \
  OFC_K2_cluster2_central_anterior_lateral; do
  for HEMISPHERE in left right; do
    python scripts_MRI/Decoding_probe_cross_task/aggregate_roi_crosstask_odor2preconditioned.py \
      --input_csv "$SPCD_DATA/roi_decoding/probe_crosstask_odor2preconditioned/roi-${ROI}_${HEMISPHERE}_subject_results.csv"
  done
done
```

The subject CSVs report the hemisphere rule, bilateral voxel count, midline
voxel count, number excluded by the hemisphere split, and post-QC voxel
count. A significant result in one hemisphere but not the other is not
evidence for lateralization; that conclusion requires a direct paired
left-minus-right test. Because the atlas halves differ in voxel count, any
paired hemisphere difference should also receive a voxel-count-matched
sensitivity analysis.

If the original `sub-14` TSVs are stored elsewhere, provide:

```bash
  --sub14_original_events_dir /path/to/sub-14/func_PAID/premeal
```

### Label and beta alignment

- Passive `odor_1` and `odor_2` labels are read as recorded, with no
  subject-specific relabeling.
- `sub-14` cue labels are matched by `run`, `trial`, and `img_index` to the
  `*_events_orig.tsv` files. This keeps the passive and premeal tasks in
  the original physical-odor convention.
- Passive beta numbers preserve row positions in the full event table.
  Filtering out clean-air trials does not renumber the corresponding beta
  files.

### Preprocessing and metrics

- Voxelwise z-scoring is performed separately within every passive run
  and every cue run.
- Voxels nonfinite in any passive or selected cue sample are dropped
  before normalization.
- Classifier: linear SVM with fixed `C=1`.
- Primary cross-task measure: cue balanced accuracy against `0.5`.
- Secondary measures: cue accuracy, AUC, and mean signed geometric
  distance from the SVM boundary.
- Passive leave-one-run-out metrics are saved as a training-data
  manipulation check.

Subject results default to:

```text
<base_dir>/roi_decoding/probe_crosstask_odor2preconditioned/
  roi-<roi_name>_subject_results.csv
  roi-<roi_name>_predictions/
```

### Group analysis

```bash
python scripts_MRI/Decoding_probe_cross_task/aggregate_roi_crosstask_odor2preconditioned.py \
  --input_csv /path/to/SPCD_data/roi_decoding/probe_crosstask_odor2preconditioned/roi-OFC_K2_cluster2_central_anterior_lateral_subject_results.csv
```

The group script tests accuracy and AUC against `0.5`, and signed distance
against `0`, using one-sample t tests, Wilcoxon tests, and sign tests. The
primary aggregation excludes `sub-19` and `sub-33`, matching the RSA
group analysis, while reporting voxel-loss and passive-decoding QC flags
without additional automatic exclusions.


