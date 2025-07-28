# Odor Scan GLM Analysis

This section contains code for running a General Linear Model (GLM) analysis on fMRI data from the odor scan task. The analysis includes onset specification, and model estimation using SPM. Both subject and group level. 

## Overview

- Used 4D fMRI data (smoothed using a 6mm FWHM Gaussian kernel)
- Data were analyzed in normalized MNI space
- Two runs of the odor scan task were modeled separately per subject
- Each run included 3 conditions (odor 0, 1, 2), with 10 trials per condition
- Trial duration was modeled as 2 seconds
- Onsets were defined per run using subject-specific task design files
- Nuisance regressors (e.g., motion parameters) were included in the GLM

## Group-Level Results

We conducted a second-level random-effects analysis for the contrast **odor > clean air**,  
using an external gray matter mask (`gm_0.1.nii`).

The resulting group-level statistical map revealed significant activation in expected olfactory and value-related regions, including:

- **Primary olfactory (piriform) cortex**
- **Orbitofrontal cortex (OFC)**
- **Amygdala and adjacent medial temporal areas**

These findings demonstrate robust engagement of the olfactory network during odor presentation and validate both the task design and GLM specification.

## Note

- In this analysis, no correction was made for sub-14, which had odor 1 and 2 labels swapped.  
  However, this does not affect interpretation since we are only interested in the **odor > clean air** contrast.
