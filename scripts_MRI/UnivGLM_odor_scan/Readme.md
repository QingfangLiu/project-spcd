# Odor Scan GLM Analysis

This project contains code for running a General Linear Model (GLM) analysis on fMRI data from the odor scan task. The analysis includes onset specification, and model estimation using SPM. Both subject and group level. 

## Overview

- Used 4D fMRI data (smoothed using a 6mm kernel)
- Onsets were defined per run using task design files.
- Using 3 conditions: odor 0, 1, 2, duration of 2s
- Nuisance regressors (e.g., motion parameters) were included in the GLM.

## Group-Level Results

We conducted a second-level random-effects analysis for the contrast **odor > clean air**. The resulting group-level statistical map revealed significant activation in expected olfactory and value-related regions, including:

- **Primary olfactory (piriform) cortex**
- **Orbitofrontal cortex (OFC)**
- **Amygdala and adjacent medial temporal areas**

These findings confirm robust engagement of the olfactory network during odor presentation and validate the task design and GLM specification.



## Known Issues

- **sub-35** and **sub-36** were **not processed** successfully due to missing nuisance regressors.
- These regressors need to be generated.


## Next Steps

- [ ] Set up or verify nuisance regressors for `sub-35` and `sub-36`
- [ ] Re-run the GLM pipeline for those subjects
- [ ] Also need to recode odor 1 and 2 for that subject with meal feed issue.
- [ ] Now the folders contain both nii files and nii.gz files; need to clean up to save space
- [ ] To add an external gray matter mask on group level analysis


