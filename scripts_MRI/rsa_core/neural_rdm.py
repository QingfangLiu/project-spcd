"""
Neural representational dissimilarity matrix computation.

This module converts condition-wise neural activation patterns
into neural RDMs using correlation- or distance-based metrics.

Functions here operate purely in memory on NumPy arrays and
contain no assumptions about task structure, models, or analysis
pipelines.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

def compute_neural_rdm(patterns, metric='correlation'):
    """
    Compute neural RDM from condition patterns.
    
    Parameters
    ----------
    patterns : np.ndarray (n_conditions, n_voxels)
        Neural patterns for each condition
    metric : str
        'correlation' (default) or 'euclidean'
    
    Returns
    -------
    rdm : np.ndarray (n_conditions, n_conditions)
        Neural similarity matrix (correlation values if metric='correlation')
    """
    n_cond = patterns.shape[0]
    
    if metric == 'correlation': # in [-1, 1]
        # Compute correlation-based similarity (guardrail added)
        rdm = np.corrcoef(patterns) # correlation between all pairs of rows
        rdm = np.nan_to_num(rdm, nan=0.0) # replace NANs with 0
        np.fill_diagonal(rdm, 1.0) # force diagonal to be 1.0
    elif metric == 'euclidean': # in [0, 1]
        # Compute pairwise distances and convert to similarity
        distances = squareform(pdist(patterns, metric='euclidean'))
        # Convert to similarity (invert and normalize)
        max_dist = distances.max()
        if max_dist > 0:
            rdm = 1 - (distances / max_dist)
        else:
            rdm = np.ones((n_cond, n_cond))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return rdm


   