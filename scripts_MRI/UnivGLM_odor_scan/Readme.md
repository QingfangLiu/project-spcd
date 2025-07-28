# Odor Scan GLM Analysis

This section contains code for running a General Linear Model (GLM) analysis on fMRI data from the odor scan task. The analysis includes onset specification, and model estimation using SPM. Both subject and group level. 

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

## Note

- In this analysis, no correction was made on sub-14 with swapped odor 1 and 2, but it does not matter because we are interested in odor>clean-air in the contrast.
- [ ] To add an external gray matter mask on group level analysis
