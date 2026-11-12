# Conditioning identity results: PI review draft

This folder holds a proposed Results section for PI discussion before a
decision about integration into the main paper or supplement. It sits beside
`writing/main/` and `writing/supp/`. The folder name describes the analysis
without implying that it has been selected for the main manuscript.

## Files

- **`results.pdf`**: standalone copy of the agreed two-paragraph draft for
  sharing with the PI. The selected section heading is "Lateral OFC
  represents reward identity during conditioning."
- **`results.tex`**: editable source and canonical version of the draft.
- **`methods.pdf`**: two-paragraph Methods addition for PI review; no
  allocation to main-text versus supplementary Methods has been made.
- **`methods.tex`**: editable source focused on the retained 5 s conditioning
  GLM, pooled odor-delivery regressor, trial/run inclusion, and RSA inference.
  General preprocessing, nuisance regressors, ROI definitions, and RSA
  procedures are referenced back to the existing probe Methods.
- **`README.md`**: analysis provenance and working notes; these notes are
  separate from the proposed Results prose.

The concluding sentence cites Howard et al. (2015) and Howard and Kahnt
(2017), using the main paper's existing `howard2015identity` and
`howard2017identity` keys. The standalone PDF reads their entries directly
from `../main/sn-bibliography.bib` and includes a short reference list.
The Methods draft is written as an addition to the paper's existing Methods
and does not repeat their general descriptions or citations. Additional
implementation details remain in this README and the linked analysis notes.

The text follows the main paper's Results structure: question and purpose,
brief analysis description, statistical results, and a restrained implication.
The main and supplementary manuscripts have not been edited for this draft.

The motivation connects the probe finding to the resource-limited account:
inferred outcome identities may receive priority when both cue classes share
limited resources. Conditioning provides a complementary task context because
only directly conditioned cues are presented, and participants explicitly
report which odor each cue predicts. The question is whether odor-related
information is detectable for conditioned cues in that context, where
preconditioned cues are not also presented. This is a contextual extension of
the interpretation, rather than a demonstrated task-phase difference or a new
formal prediction derived from the model equations.

## Result selected for discussion

The selected result is **EXT5s, all available conditioning runs, cross-run
RSA**, using the full eligible sample. This is the analysis saved as
`cueEXT5s/sensitivity_runs1-3`, not the separately designated runs 2/3 primary
analysis. The user chose to retain this result after reviewing both run
selections. Its original analysis designation should remain transparent if
the section is integrated into the paper.

| ROI | n | Mean Spearman rho | t (df = 28) | One-sided, uncorrected p |
| --- | --- | --- | --- | --- |
| Lateral OFC | 29 | 0.0130742357 | 2.01025099 | 0.02706335 |
| Medial OFC | 29 | 0.0068176640 | 1.20082800 | 0.11993915 |

The draft rounds these to rho = 0.0131 / 0.0068, t = 2.01 / 1.20, and
p = 0.027 / 0.120. These are group one-sample t-tests on participant-level
RSA coefficients. The inferential sample size is the number of participants,
not the number of RDM pairs. The lateral two-sided p-value is 0.0541.

Source files, relative to this folder:

- [Lateral OFC group summary](../../rsa_roi/conditioning_odor_identity/cueEXT5s/sensitivity_runs1-3/cluster2_central_anterior_lateral/group_summary.csv)
- [Lateral OFC subject results](../../rsa_roi/conditioning_odor_identity/cueEXT5s/sensitivity_runs1-3/cluster2_central_anterior_lateral/subject_results.csv)
- [Medial OFC group summary](../../rsa_roi/conditioning_odor_identity/cueEXT5s/sensitivity_runs1-3/cluster1_medial_posterior_lateral/group_summary.csv)
- [Medial OFC subject results](../../rsa_roi/conditioning_odor_identity/cueEXT5s/sensitivity_runs1-3/cluster1_medial_posterior_lateral/subject_results.csv)
- [Lateral OFC figure](../../plots/conditioning_odor_identity/cueEXT5s/sensitivity_runs1-3/cluster2_central_anterior_lateral.pdf)
- [Medial OFC figure](../../plots/conditioning_odor_identity/cueEXT5s/sensitivity_runs1-3/cluster1_medial_posterior_lateral.pdf)
- [Run manifest](../../rsa_roi/conditioning_odor_identity/run_manifest.json)
- [Analysis implementation and detailed notes](../../scripts_MRI/rsa_conditioning_odor_identity/README.md)

## Brief method and sample notes

- Existing least-squares-separate conditioning betas are used; the analysis
  does not re-estimate GLMs. EXT5s models the first 5 s after cue onset,
  covering the initial cue and explicit odor prediction period.
- The same independently defined bilateral Kahnt K=2 OFC regions are used
  as in the probe analysis: cluster2 is lateral OFC and cluster1 is medial.
- Inclusion requires a correct response, recorded response, and finite
  positive RT. Clean-air trials are excluded. Accuracy comes from the
  behavioral records because the old beta indexes may contain missing values.
- Individual trial patterns are retained. Each voxel is centered across
  selected presentations within each run. Nonfinite voxels are dropped.
- Neural dissimilarity is one minus Pearson pattern correlation; the model
  codes same expected odor as 0 and different expected odors as 1. Spearman
  RSA uses only between-run pairs. The selected result includes repeated
  presentations of the same visual cue.
- Of 34 task completers, subjects 19/31/33 were excluded in the manuscript
  sample. This conditioning analysis additionally excludes subject 26 for
  sparse correct trials and subject 38 for severe OFC coverage loss.
  These additional exclusions are described in the analysis README.
- **Twenty-eight participants contribute runs 1/2/3; subject 35 contributes
  runs 1/2 only**, because the third scan was interrupted. Accordingly the
  prose says "all available conditioning runs," rather than three complete
  runs from every participant.

## Interpretation and integration notes

- This is a small positive association with the odor-identity RDM. The draft
  uses "suggestive evidence" and "odor-related information"; it does not
  claim a demonstrated image-independent sensory representation.
- The corresponding lateral OFC cross-set result is rho = 0.0042,
  one-sided p = 0.276. That restriction compares different cue images as
  well as different runs. Its nonsignificance leaves generalization across
  cues unresolved. Repeated-image similarity can contribute to the selected
  unrestricted result.
- The originally designated runs 2/3 lateral OFC EXT5s result is rho = 0.0073,
  one-sided p = 0.214 (n = 28). The all-run analysis uses more trial data;
  the difference in significance does not itself establish a difference
  between learning phases or run selections.
- Significance in lateral OFC and nonsignificance in medial OFC do not
  establish a difference between the regions. No direct regional contrast
  is included in this draft. Likewise, task dependence has not been tested
  by a formal conditioning-versus-probe interaction.
- "Only conditioned cues" refers to which visual cues were presented during
  conditioning. The cover story still encouraged participants to think about
  their paired, absent preconditioned cues. The draft therefore does not
  establish that internal retrieval of preconditioned cues or competition
  for resources was eliminated. Cue composition, explicit response demands,
  learning, and odor delivery also differ between conditioning and probe;
  their individual contributions have not been isolated.
- Existing cue GLMs include a pooled odor-delivery regressor, rather than
  separate odor-identity delivery regressors. Anticipation and receipt have
  fixed timing and overlapping modeled responses. Signal loss or leakage
  has not been quantified; the draft makes no claim that the GLM either
  removed all odor-identity information or completely isolated anticipation.
- Before manuscript integration, decide placement and figure presentation,
  and merge the unified Methods draft with the paper's existing Methods.
  Reconcile wording about the null conditioned-cue probe result with the
  new conditioning result. The Methods draft documents the selected GLM,
  trial filtering, available runs, exclusions, partial data, and inference.
- Detailed pattern/RDM diagnostics remain with the MRI data on the other
  machine; this folder contains the writing draft and notes only.

## Methods provenance and verification notes

The Methods draft was checked against the implementation and synced tables,
not reconstructed from a fitted SPM design on this machine. Source links:

- [EARLY cue GLM](../../scripts_MRI/MultivGLM_odorpred/run_trialwise_glms_cueEARLY.m)
- [EXT5s cue GLM](../../scripts_MRI/MultivGLM_odorpred/run_trialwise_glms_cueEXT5s.m)
- [Trial timing and delivery event generation](../../scripts_MRI/MultivGLM_odorpred/generate_bids_events_from_design.m)
- [Motion and image-artifact nuisance regressors](../../scripts_MRI/preproc_organize_imaging/Do_6_plot_motion_saveNR.m)
- [Respiratory regressors](../../scripts_MRI/preproc_organize_imaging/Do_5_process_labchart.m)
- [Conditioning trial selection, RSA, and group orchestration](../../scripts_MRI/rsa_conditioning_odor_identity/run_rsa.py)
- [Shared ROI extraction and within-run centering](../../scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_roi_rsa.py)
- [Shared Spearman RSA calculation](../../scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/run_roi_rsa_spearman.py)
- [Shared participant-level group tests](../../scripts_MRI/rsa_imagewise_odor_identity_no_image_first_image/aggregate_roi_rsa_spearman.py)
- [ROI preparation and mask provenance](../../scripts_MRI/roi_mask_prep/README_OFC_clusterMap_K2.md)

Two details are deliberately recorded rather than assumed:

- The GLM headers describe AR(1), but the executable batches do not set the
  `cvi` field explicitly. The Methods draft gives the settings explicitly
  specified in the scripts and does not assert a verified autocorrelation
  model. Confirm the fitted/default SPM setting on the data machine if that
  detail is to be added. Temporary per-trial GLM work files were deleted by
  the original scripts after saving each target beta.
- The odor regressor uses the programmed 7 s onset and 2 s duration. The
  event generator does not add the approximately 2 s tubing transit described
  elsewhere in the task documentation. The Methods reports the implemented
  timing; no timing correction or new GLM was introduced in this writing pass.

## Build

From this folder:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error results.tex methods.tex
```

Compile again after editing the source so the PDF stays synchronized.
The existing repository ignore rules cover the LaTeX auxiliary files.
