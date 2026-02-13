
"""
Utilities for masking condition pairs in RSA analyses.

This module provides helper functions to construct boolean masks
over vectorized RDM entries, typically used to restrict RSA
computations to specific condition subsets (e.g., odor–odor pairs).
"""

import numpy as np
np.seterr(all='raise') # This turns warnings into errors so you can see the traceback

def build_odor_pair_mask(condition_order, odor_set):
    """
    Construct boolean mask for selecting odor–odor condition pairs.

    Mask is aligned with the vectorized lower triangle of an RDM.

    Returns
    -------
    mask : np.ndarray (n_pairs,)
        Boolean mask indicating which condition pairs to keep.
    """
    n = len(condition_order)
    i, j = np.tril_indices(n, k=-1)
    return np.array(
        [(condition_order[ii] in odor_set) and (condition_order[jj] in odor_set)
         for ii, jj in zip(i, j)],
        dtype=bool
    )
