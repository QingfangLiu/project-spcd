# Passive-odor to probe-cue ROI decoding

Train a linear SVM (`C=1`) on the two passive odor-scan runs and test either
premeal or postmeal first-cue responses. Both conditioned and preconditioned
odor-predictive cues are evaluated separately; clean-air cues and second cue
presentations are excluded. No searchlight analysis is run.

## Run on the machine containing SPCD_data

From the project clone, using the Python environment with nibabel, numpy,
pandas, scipy, scikit-learn, and matplotlib installed:

```bash
python scripts_MRI/Decoding_probe_cross_task/run_ofc_k2_crosstask_decoding.py \
  --base_dir /path/to/SPCD_data \
  --phase postmeal \
  --strict
```

This runs both bilateral Kahnt K=2 OFC ROIs and both cue subsets, saves
trial-level predictions, aggregates participant results, and creates PDF/PNG
plots. Add `--dry_run` to preview commands without accessing the MRI data or
writing results. Use `--subjects sub-01 sub-02` for a smaller run.

Required inputs:

- Passive events under `SPCD_data/nii/<sub>/func_PAID/odor_scan/` and betas
  under `SPCD_data/GLM/odor_scan_trialwise_2mm/<sub>/`.
- Postmeal per-run indices named `<sub>_postmeal_run-<run>_beta_index.csv`
  and their beta images under `SPCD_data/GLM/postmeal_trialwise_2mm/<sub>/`.
- Existing premeal RSA diagnostics under the clone's
  `rsa_roi/probe_imagewise_odor_identity_no_image_first_image/premeal_ofc_k2_spearman/`,
  with `<cluster>/diagnostics/<sub>/<cue_subset>_roi_patterns.npz` and
  `<cue_subset>_samples.csv`. Override this root with `--diagnostics_root`.
- For sub-14, original postmeal event TSVs named
  `sub-14_task-postmeal_run-<run>_events_orig.tsv` under
  `SPCD_data/nii/sub-14/func_PAID/postmeal/`.

Postmeal decoding uses the diagnostics' **raw ROI voxel coordinates**, but
reads cue responses from the actual postmeal beta images. A postmeal RSA
rerun is unnecessary. Probe image shape and affine must match the passive
images. Missing run indices are recorded in `missing_cue_runs`; fewer than
two available runs, missing beta files, malformed inputs, or missing sub-14
original labels cause an analysis failure. Sub-36's postmeal run 3 is included
when present; the premeal missing-run convention is not carried over.

Physical odor labels follow the same conventions as the premeal decoder.
In particular, sub-14 uses original events matched by run, trial, and image
position rather than the corrected `odor_index` column in the beta indices.
No guessed label swap or fallback is applied.

## Outputs and statistics

Default postmeal outputs are under the clone's
`roi_decoding/probe_crosstask_odor2preconditioned_postmeal/`:

- Per-ROI participant results, group summaries, and failure CSVs.
- Trial predictions in each ROI's predictions directory.
- Phase and input provenance in participant results.

Plots are saved under `plots/OFC_K2_two_roi_crosstask_decoding_accuracy_postmeal_1x4.{pdf,png}`.
Use `--output_dir` or `--plot_dir` to change destinations.

The established processing is retained: select finite voxels jointly across
passive and selected cue data; z-score voxel responses separately within each
task/run with stimulus sets pooled; train on both passive runs; and evaluate
balanced accuracy against 50%. Passive leave-one-run-out performance is also
reported. Participants sub-19 and sub-33 are excluded during group aggregation
and plotting, with `--skip_subs` available to change that list. T, Wilcoxon,
and sign-test results are reported without multiplicity correction; the plot
retains the existing one-sided Wilcoxon annotations.

The decoder's default phase remains `premeal`, using the original cached cue
patterns and default output location. `--phase postmeal` selects the new loader
and a separate output location. The aggregation and plotting scripts reject
mixed-phase inputs and treat legacy results without a phase column as premeal.

These are separate phase-specific analyses. They do not test a pre/post
difference or phase-by-cue-type interaction. Such a comparison should use
paired participants and a shared finite-voxel set across phases so that the
trained classifier is identical; phase-specific finite-voxel selection can
otherwise change its feature set. A significant premeal result and a
nonsignificant postmeal result do not establish a phase difference.

## Tests without MRI data

```bash
python -m unittest discover \
  -s scripts_MRI/Decoding_probe_cross_task \
  -p 'test_roi_crosstask_odor2preconditioned*.py'
```
