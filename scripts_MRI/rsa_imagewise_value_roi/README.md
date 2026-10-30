# Combined-phase value RSA in vmPFC and OFC

Run from a clone of this repository on the computer containing the trialwise
beta data using Python 3.10 or newer. The raw images remain outside the repository; results are written
inside the clone by default. No MATLAB or Nilearn installation is required.

```bash
python -m pip install -r scripts_MRI/rsa_imagewise_value_roi/requirements.txt
python -B scripts_MRI/rsa_imagewise_value_roi/run_roi_rsa.py \
  --base_dir /path/to/SPCD_data
```

The script infers the project root from its own location. `--base_dir` must
contain the same inputs used by the existing image-wise value searchlight:

- `GLM/{premeal,postmeal}_trialwise_2mm/sub-XX/` with beta images and
  `sub-XX_{premeal,postmeal}_run-{1,2,3}_beta_index.csv` files;
- `nii/sub-XX/func_PAID/{premeal,postmeal}/` with
  `sub-XX_task-{premeal,postmeal}_run-{1,2,3}_events.tsv` files;
- `masks/gm_0.1.nii`, the existing binary gray-matter analysis mask.

Beta-index CSVs must contain `beta_file`, `trial`, `img_index`, `cue_index`,
and `odor_index`. Events files must contain `trial`, `trial_type`, and
`response_hand`, as in the existing ROI pipeline. A missing beta-index CSV
marks an unavailable run; a present but broken run fails explicitly. Both
phases must contribute samples. Incomplete run counts are flagged, not an
additional automatic exclusion.

## Analysis and reuse

The analysis uses all conditioned/preconditioned and clean-air cues, both cue
presentation positions, and combined premeal/postmeal data. It reuses the
current `rsa_imagewise` value RDM, nuisance RDMs, correlation-distance function,
ratings loader, and OLS regression. It also reuses mask-space checks, beta and
events loading, ROI pattern extraction, nonfinite-voxel handling, and cross-run
pair selection from `rsa_imagewise_odor_identity/run_roi_rsa.py`.

Six phase-by-outcome ratings are jointly standardized within participant.
Image identity ignores presentation position; button labels apply only to the
second presentation. The model includes value, cue type, image identity,
button presence, and button side, with an intercept. It includes neither an
odor-identity nor a phase nuisance RDM. Postmeal run IDs are offset by 100, so
all cross-phase pairs are eligible. Within-run pairs are excluded.

Patterns are **not centered within run by default**, matching the current
value searchlight's default. `--center_within_run` enables a sensitivity
analysis and defaults to a separate `both_centered` output folder. The
odor-identity analysis in the paper uses within-run centering; that choice
is not silently transferred to the value model. Voxels nonfinite in any
selected presentation are removed separately per participant/ROI. Constant
spatial patterns or unidentifiable value regressors produce explicit failures.

The ROI coefficient is fitted to the neural RDM across the whole ROI, rather
than obtained by averaging searchlight coefficients. Input beta images are
not additionally smoothed and ROI coefficients do not undergo the 6 mm
group-map smoothing used for the searchlight results.

## ROIs

- `Bartra_vmPFC`: the connected component containing MNI `[0, 44, -6]` in
  the Bartra Fig. 3D positive-versus-negative meta-analysis mask, exactly the
  independent component used by the existing vmPFC figure/SVC workflow.
  The source component must contain 1,245 native voxels. Striatal and PCC
  components are excluded without consulting the current value statistic map.
- `OFC_K2_cluster1_medial_posterior_lateral`: the bilateral medial OFC mask
  used in the current paper.
- `OFC_K2_cluster2_central_anterior_lateral`: the bilateral lateral OFC mask
  used in the current paper.

Masks are aligned to the gray-matter analysis grid with nearest-neighbor
resampling if needed and intersected with `gm_0.1.nii > 0`. Beta images must
match that grid; beta images are not silently resampled. Use `--gm_mask` to
specify the same analysis mask explicitly when stored elsewhere.

## Participant inclusion

The default candidate list comes from `beh_processed_subs/sub-*` in the clone.
The current paper exclusions are **sub-19, sub-31, and sub-33**, matching the
current behavior and main fMRI figure scripts rather than the older ROI
aggregation defaults. Complete pre/post pleasantness ratings are also required.
With the current repository this yields an expected 30 participants, because
sub-36 lacks ratings. Final counts depend on successful beta-data validation.

Ratings default to `beh_processed_subs/sub-XX/pleasantness_summary.csv` in the
clone. `--ratings_file /path/to/compiled_pleasantness_ratings.csv` accepts the
existing wide format (`sub`, `pre_odor1`, `pre_odor2`, `pre_clean`,
`post_odor1`, `post_odor2`, `post_clean`). `--subjects sub-01 sub-02 ...` can
restrict a trial run. `--skip_subs` overrides the paper exclusion list.

A participant enters the coefficient table only after all three ROIs succeed.
Any unexpected eligible-participant failure prevents group inference by
default; inspect `subject_inclusion.csv` and fix inputs before rerunning.
`--allow_partial` explicitly permits a summary of the successful common set.
Voxel loss of at least 20% is flagged but does not itself exclude participants.

## Outputs

Default destination: `rsa_roi/probe_imagewise_value/both/` inside this clone.

- `subject_betas.csv`: one row per included participant and ROI, with all five
  RSA coefficients, sample/run/pair counts, voxel counts, and QC flags.
- `group_summary.csv`: one value test per ROI, with the same participants in
  all three, mean, SD, SE, 95% t confidence interval, t/df, one-sided and
  two-sided uncorrected p values, Cohen's dz, and Holm-adjusted p values across
  the three ROI tests. Nuisance coefficients are not treated as primary tests.
- `subject_inclusion.csv`: paper exclusions, missing ratings, failures, and
  the final common included participants; includes ratings provenance.
- `analysis_config.json`: settings, paths, exclusions, final participant list,
  failures, and mask provenance.
- `mask_manifest.csv` and `masks/`: source hashes, effective voxel counts,
  and the three actual analysis masks.

Add `--save_diagnostics` to retain selected sample metadata, raw/analysis ROI
patterns, voxel-selection masks, neural/model RDMs, and the cross-run pair
mask under `diagnostics/`. These can be large and contain absolute data paths;
they are optional. All outputs are local files; no Git commit or push is automatic.

An existing nonempty output directory is refused to prevent mixing results
from different runs. Use a fresh `--output_dir` to rerun or test a subset:

```bash
python -B scripts_MRI/rsa_imagewise_value_roi/run_roi_rsa.py \
  --base_dir /path/to/SPCD_data \
  --subjects sub-01 sub-02 \
  --output_dir rsa_roi/probe_imagewise_value/smoke_test
```

To regenerate group summaries from a completed coefficient table:

```bash
python -B scripts_MRI/rsa_imagewise_value_roi/aggregate_roi_rsa.py \
  --input_csv rsa_roi/probe_imagewise_value/both/subject_betas.csv
```

The standalone aggregator verifies a balanced three-ROI table and applies the
current paper exclusions again. It cannot infer participants absent from the
input table; consult the accompanying inclusion report.

## Export raw patterns for analysis on another computer

Use the standalone exporter on the computer containing the beta images:

```bash
python -B scripts_MRI/rsa_imagewise_value_roi/export_roi_patterns.py \
  --base_dir /path/to/SPCD_data
```

It reuses the three independent ROI definitions and writes to
`rsa_roi/probe_imagewise_value/raw_patterns/` inside the clone. Copy or sync
that entire directory back to the analysis computer. No RSA regression is
performed during export. Existing `both/` results are preserved. The same
dependency list applies; use a fresh `--output_dir` for subsequent exports.

The default paper exclusions remain sub-19, sub-31, and sub-33. Missing ratings
do **not** block raw extraction: sub-36 can be exported and is flagged as
having incomplete ratings, so later value analyses must apply that additional
eligibility rule. Sub-38's raw values are saved even if every voxel fails the
all-presentations RSA check. The exporter attempts every ROI for every sample;
it never requires an earlier ROI or a model fit to succeed.

Each participant has a `samples.csv` and one `roi-<ROI>_patterns.npz` archive
per ROI. Archives contain the raw **presentations × voxels** matrix as float64,
preserving the scaled beta-image values, including NaN/Inf. There is no
centering, normalization, averaging, or nonfinite-voxel removal. Voxel columns
are in NumPy mask-index order; `voxel_ijk`, `voxel_mni`, and `affine` retain
their exact locations on the analysis grid. Archives contain numeric/string
arrays and can be opened with `np.load(..., allow_pickle=False)`.

`samples.csv` preserves row indices, participant, phase (`session`), unique
run ID, within-phase run, trial, presentation position, cue/odor indices,
condition/image labels, odor group, cue type, cue set, and response labels.
Both the trial response and the RSA button label are retained; the latter
is `none` for first presentations. Source beta paths, beta-index row numbers,
and index-file hashes are included. Phase/run/sample labels are also embedded
in each archive, and its `sample_index` aligns with the CSV rows.

The bundle also includes:

- `ratings.csv`: unstandardized pre/post pleasantness for each outcome and
  participant, with explicit availability flags;
- `subject_export.csv`: paper exclusions, export status, missing runs, sample
  counts, rating completeness/provenance, and exported archive paths;
- `coverage_summary.csv`: finite-voxel intersections and all-nonfinite sample
  counts for all samples, each phase/run, each phase/presentation position,
  and first/second presentations across phases;
- per-ROI `roi-<ROI>_sample_coverage.csv` files: finite, NaN, and Inf counts for
  each beta image, making the specific problematic phase/run/trial identifiable;
- `mask_manifest.csv`, `masks/`, `export_config.json`, and `input_issues.csv`:
  analysis masks and provenance, export settings/schema version, and any
  unreadable or malformed inputs.

Absent beta-index CSVs are recorded as missing runs; available phases/runs are
still exported. A missing or unreadable beta image, or a grid mismatch, gets an
explicit `beta_read_status=error` and an all-NaN placeholder row, preserving
sample alignment. Missing events are flagged and response labels remain
`unknown`; a later analysis requiring those labels must resolve that issue.
The exporter continues other samples, ROIs, and participants, saves its reports,
and exits with an error if unexpected input problems occurred. Readable images
containing NaN/Inf are exported successfully and flagged in coverage reports.

For sub-38, inspect coverage rows with `scope=all`, `phase`, `phase_run`, and
`phase_image`, then the per-sample reports. This distinguishes coverage loss
across all samples from a particular beta image or phase. Later analyses must
choose usable voxel intersections after selecting their samples, preserving
the raw exports and documenting all analysis variants.

The bundle supports changes to sample selection, averaging, run centering,
model RDMs, and correlation-distance RSA locally. Properly noise-normalized
distances would additionally require GLM noise estimates, which are not part
of this exporter.

## Verification

```bash
python -B -m unittest discover -s scripts_MRI/rsa_imagewise_value_roi -p 'test_*.py' -v
```

Tests use temporary synthetic NIfTI images and the repository's independent
Bartra source mask. They check end-to-end extraction against the existing
searchlight estimator and an independent least-squares fit, run/phase pair
selection, paper exclusions, missing ratings, voxel QC, optional centering,
wide-format ratings, refusal to overwrite outputs, and common-participant
handling when only one ROI fails. No real participant analysis is performed.
Exporter tests additionally verify raw-value/coordinate alignment, retention
of nonfinite values and missing-rating participants, first-cue coverage
recovery, and continued export after unreadable or mismatched beta images.
