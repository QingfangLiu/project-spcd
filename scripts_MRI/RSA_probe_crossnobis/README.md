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

    -   When `neural_metric == "crossnobis"`, all model RDMs are
        converted via:

            distance = 1 - similarity

    -   Diagonal entries are explicitly set to 0 for consistency with
        crossnobis RDMs.

    -   This ensures neural and model RDMs are in the same
        representational space (both distance-based).

------------------------------------------------------------------------

## Small Changes

-   **Removed the `runs = (1,2,3)` parameter**
    -   The pipeline already infers run structure from the data.
    -   The explicit `runs` parameter was redundant and not functionally
        necessary.

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

------------------------------------------------------------------------

## What Remains To Be Implemented

-   **Noise estimation for crossnobis**
    -   Currently, the `noise` argument in
        `compute_crossnobis_neural_rdm` is set to `None` to ensure the
        code runs end-to-end.
    -   A proper implementation should:
        -   Estimate voxel-wise noise covariance (ideally from GLM
            residuals), or
        -   Use a shrinkage/diagonal approximation if residuals are not
            available.
    -   Incorporating noise normalization will improve statistical
        validity and interpretability of crossnobis distances.

- Also think about how to organize new files and name new maps
     - the code has multiple options, but the cross-run option has been run without specifying the option 

