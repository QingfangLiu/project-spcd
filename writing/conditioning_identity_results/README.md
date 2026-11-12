# Conditioning identity results: PI review draft

This folder holds proposed Results and Methods text for PI discussion before a
decision about integration into the main paper or supplement. It sits beside
`writing/main/` and `writing/supp/`. The folder name describes the analysis
without implying that it has been selected for the main manuscript.

## Files

- **`conditioning_identity.pdf`**: combined PI-review document, with Results
  first, Figure X and its caption on a separate page, followed by the
  two-paragraph Methods addition and references.
  The title is "Lateral OFC represents reward identity during conditioning."
- **`conditioning_identity.tex`**: single editable source and canonical
  version of the combined document. The Methods cover the EARLY and EXT5s
  conditioning GLMs, pooled odor-delivery regressor, trial/run inclusion,
  RSA, and decoding. General preprocessing, nuisance regressors, ROI
  definitions, and RSA procedures refer back to the existing probe Methods.
- **Figure X** is loaded from [the shared figure PDF](../../fig/figx.pdf),
  with [a PNG preview](../../fig/figx.png). Its layout has the conditioning
  schematic on the left and both result plots on the right. It displays
  EXT5s cross-run RSA and EARLY balanced accuracy, with the cue windows
  specified in the caption and significance notation matching Figures 3/4.
  The caption is included in this document's source; its standalone copy
  is [here](../../fig/figx_caption.txt). Keep these caption copies consistent.
- **`README.md`**: analysis provenance and working notes; these notes are
  separate from the proposed Results prose.

The Results concluding sentence cites Howard et al. (2015) and Howard and Kahnt
(2017), using the main paper's existing `howard2015identity` and
`howard2017identity` keys. The standalone PDF reads their entries directly
from `../main/sn-bibliography.bib` and includes a short reference list.
The combined document replaces the separate Results and Methods documents.
No allocation to main-text versus supplementary Methods has been made.
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

## RSA result selected for discussion

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

## Cross-task decoding results selected for discussion

The user chose to report **balanced accuracy**, the primary decoding measure,
for both EARLY and EXT5s. The decoder was trained on the same two-run passive
odor localizer used for the manuscript's probe analysis (40 trials, 20 per
odor; linear SVM, fixed C = 1), then tested without further fitting on correct
conditioning trials. The 29 participants are the same as in the retained RSA
sample. EARLY covers 0--2 s after cue onset, whereas EXT5s covers 0--5 s.

| ROI | Window | n | Mean balanced accuracy | t (df = 28) | One-sided, uncorrected p |
| --- | --- | --- | --- | --- | --- |
| Lateral OFC | EARLY | 29 | 52.1181681% | 1.833238 | 0.038710 |
| Lateral OFC | EXT5s | 29 | 51.8711704% | 1.451910 | 0.078820 |
| Medial OFC | EARLY | 29 | 49.2549038% | -0.576073 | 0.715416 |
| Medial OFC | EXT5s | 29 | 50.0534632% | 0.039517 | 0.484379 |

The draft rounds accuracy to two decimal places and p to three. EARLY
lateral OFC is above chance at the uncorrected one-sided threshold; EXT5s
is numerically positive but does not reach p < 0.05. Both windows are
reported together. EXT5s lateral OFC was the designated focus before running
this decoder, with EARLY and medial OFC designated comparisons; balanced
accuracy remains the primary measure for every ROI/window. The signed-distance
measure is secondary and is not included in the proposed Results text.

The manuscript's existing cross-task Methods already describe passive
training, fixed C = 1, and the absence of test-task fitting or tuning. The
conditioning Methods addition references that procedure and specifies the
conditioning windows, trial matching, common finite voxels, separate
task/window/run normalization, and balanced-accuracy inference.

Source files, relative to this folder:

- [EARLY lateral OFC group summary](../../roi_decoding/conditioning_crosstask_odor/cueEARLY/cluster2_central_anterior_lateral/group_summary.csv)
- [EXT5s lateral OFC group summary](../../roi_decoding/conditioning_crosstask_odor/cueEXT5s/cluster2_central_anterior_lateral/group_summary.csv)
- [EARLY medial OFC group summary](../../roi_decoding/conditioning_crosstask_odor/cueEARLY/cluster1_medial_posterior_lateral/group_summary.csv)
- [EXT5s medial OFC group summary](../../roi_decoding/conditioning_crosstask_odor/cueEXT5s/cluster1_medial_posterior_lateral/group_summary.csv)
- [EARLY figure](../../plots/conditioning_crosstask_odor/cueEARLY/ofc_conditioning_decoding.pdf)
- [EXT5s figure](../../plots/conditioning_crosstask_odor/cueEXT5s/ofc_conditioning_decoding.pdf)
- [Decoding run manifest](../../roi_decoding/conditioning_crosstask_odor/run_manifest.json)
- [Decoding implementation and handoff notes](../../scripts_MRI/Decoding_conditioning_cross_task/README.md)

The synced run has status `complete`, no unexpected failures, matched cue
trials and the same finite voxel set between windows, and no 20% voxel-loss
flags. Participant metrics were independently reproduced from
all 116 ROI/window prediction tables, and group statistics were checked
against the participant scores. Subject 14 used the original conditioning
event labels to maintain the physical-odor convention of passive training;
subject 35 contributed only conditioning runs 1--2.

The EARLY and EXT5s lateral accuracy estimates are similar. An exploratory
paired two-sided test of their difference gives p = 0.752; significance in
only EARLY does not establish a timing difference. The primary draft does
not add this post-run comparison. Passive-localizer leave-one-run-out
balanced accuracy was 48.10% in lateral OFC (one-sided p = 0.825) and 51.29%
in medial OFC (p = 0.281), also not above chance. The existing probe workflow
showed similarly weak passive cross-validation. These diagnostics temper
the strength of a shared sensory-code interpretation without changing the
predefined sample or primary outcome.

## Brief RSA method and shared sample notes

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
  is included in this draft for either RSA or decoding. Likewise, task
  dependence has not been tested by a formal conditioning-versus-probe
  interaction.
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
  new conditioning result. The Methods draft documents both cue GLMs,
  trial filtering, available runs, exclusions, partial data, RSA and decoding
  inference.
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
- [Conditioning cross-task decoding and group orchestration](../../scripts_MRI/Decoding_conditioning_cross_task/run_decoding.py)
- [Shared passive-localizer loading, normalization, and classifier metrics](../../scripts_MRI/Decoding_probe_cross_task/run_roi_crosstask_odor2preconditioned.py)
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
latexmk -pdf -interaction=nonstopmode -halt-on-error conditioning_identity.tex
```

If the figure changes, rebuild it first from the repository root:

```bash
python3 fig/make_figx.py
```

The [figure generator](../../fig/make_figx.py) validates the plotted statistics
against the synced group tables and trims only the unused bottom margin.

Compile again after editing the source so the PDF stays synchronized.
The existing repository ignore rules cover the LaTeX auxiliary files.
