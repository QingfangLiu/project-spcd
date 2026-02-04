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


def compute_crossrun_neural_rdm(
    patterns: np.ndarray,
    run_ids: np.ndarray,
    cond_ids: np.ndarray,
    n_cond: int,
    metric: str = "correlation",
):
    """
    Compute an (n_cond x n_cond) similarity matrix using ONLY cross-run pairs.

    patterns: (n_samples, n_voxels), each sample is one (run, condition) map
    run_ids:  (n_samples,) run label per sample
    cond_ids: (n_samples,) condition index [0..n_cond-1] per sample
    """
    if metric != "correlation":
        raise ValueError("compute_crossrun_neural_rdm currently supports metric='correlation' only")

    patterns = np.asarray(patterns)
    run_ids = np.asarray(run_ids).astype(int)
    cond_ids = np.asarray(cond_ids).astype(int)

    # index by (run, cond) -> sample row
    idx = {}
    for k in range(patterns.shape[0]):
        idx[(int(run_ids[k]), int(cond_ids[k]))] = k

    runs = sorted(set(run_ids.tolist()))
    eps = 1e-12

    def _corr(a, b):
        a = a.astype(float, copy=False)
        b = b.astype(float, copy=False)
        a = a - a.mean()
        b = b - b.mean()
        da = np.sqrt(np.sum(a * a))
        db = np.sqrt(np.sum(b * b))
        if da < eps or db < eps:
            return 0.0
        return float(np.dot(a, b) / (da * db))

    rdm = np.zeros((n_cond, n_cond), dtype=float)

    for i in range(n_cond):
        for j in range(n_cond):
            s = 0.0
            c = 0
            for ra in runs:
                for rb in runs:
                    if ra == rb:
                        continue  # EXCLUDE within-run
                    ka = idx.get((ra, i), None)
                    kb = idx.get((rb, j), None)
                    if ka is None or kb is None:
                        continue
                    s += _corr(patterns[ka], patterns[kb])
                    c += 1
            rdm[i, j] = (s / c) if c > 0 else 0.0

    np.fill_diagonal(rdm, 1.0)
    return rdm



   