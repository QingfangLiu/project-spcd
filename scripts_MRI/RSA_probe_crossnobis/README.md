This is an updated version of the `RSA_probe_x_runs` pipeline with
support for crossnobis dissimilarity in addition to the original
cross-run correlation approach.

------------------------------------------------------------------------

## Major Changes

-   **Added a new parameter `neural_metric`**
    -   Accepts two options:
        -   `corr` (default): cross-run Pearson correlation
            (similarity-based RDM)\
        -   `crossnobis`: cross-validated Mahalanobis distance
            (distance-based RDM)
    -   This parameter is exposed at the top-level script and propagated
        through the searchlight and estimator modules.
-   **Added `compute_crossnobis_neural_rdm` in `neural_rdm.py`**
    -   Provides an alternative to `compute_crossrun_neural_rdm`.
    -   Uses `rsatoolbox` to compute crossnobis dissimilarity.
    -   Returns a **distance matrix** with diagonal ≈ 0 (not forced to 1
        as in correlation mode).
-   **Added model conversion utilities in `models.py`**
    -   Introduced helper functions to convert similarity-based model
        RDMs into distance form.

    -   Diagonal entries are explicitly set to 0 for consistency with
        crossnobis RDMs.

    -   This ensures neural and model RDMs are in the same
        representational space (both distance-based).


### Noise Estimation for Crossnobis

- In the current implementation, the `noise` argument in `compute_crossnobis_neural_rdm` is set to `None`.

- A pre-whitening step is performed prior to the searchlight analysis. Voxel-wise noise variance is estimated from the GLM residual image (`ResMS.nii`) and used to whiten beta estimates before computing crossnobis distances.

- Accordingly, noise normalization is handled at the preprocessing stage rather than being passed explicitly to the crossnobis function.

------------------------------------------------------------------------

## Important Behavioral Differences

-   In `corr` mode:
    -   Neural RDM = similarity matrix
    -   Diagonal is explicitly set to 1
    -   Model RDMs remain similarity matrices
-   In `crossnobis` mode:
    -   Neural RDM = distance matrix
    -   Diagonal ≈ 0 (not manually forced)
    -   Model RDMs are converted to distance matrices (diagonal = 0)
    -   Distances can be negative (expected under noise)


