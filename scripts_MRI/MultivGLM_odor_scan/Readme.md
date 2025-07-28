# MultivGLM_odor_scan

Scripts for trial-wise GLM analysis of the SPCD odor scan task.

## Scripts

- **generate_bids_events_from_design.m**  
  Converts subject Excel designs into BIDS-compatible `events.tsv`.  
  Includes special label fix for `sub-14` (odor_1 ↔ odor_2).

- **run_trialwise_glms.m**  
  Runs single-trial GLMs per subject and run for multivoxel decoding.

- **RunCode.m**  
  Wrapper script to run the full pipeline across subjects.

