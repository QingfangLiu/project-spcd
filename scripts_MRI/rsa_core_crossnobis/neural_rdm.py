"""
Neural representational dissimilarity matrix computation.

This module converts condition-wise neural activation patterns
into neural RDMs using correlation- or distance-based metrics.
"""

import numpy as np

def compute_crossrun_neural_rdm(
    patterns: np.ndarray,
    run_ids: np.ndarray,
    cond_ids: np.ndarray,
    n_cond: int,
):
    """
    Compute an (n_cond x n_cond) similarity matrix using ONLY cross-run pairs.

    patterns: (n_samples, n_voxels), each sample is one (run, condition) map
    run_ids:  (n_samples,) run label per sample
    cond_ids: (n_samples,) condition index [0..n_cond-1] per sample
    n_cond: number of unique conditions
    """
    
    patterns = np.asarray(patterns)
    run_ids = np.asarray(run_ids).astype(int)
    cond_ids = np.asarray(cond_ids).astype(int)

    # index by (run, cond) -> sample row
    # a fast look up table to know the location of the sample
    idx = {}
    for k in range(patterns.shape[0]):
        idx[(int(run_ids[k]), int(cond_ids[k]))] = k

    # get sorted, unique runs
    runs = sorted(set(run_ids.tolist()))
    eps = 1e-12

    # A manual Pearson correlation
    def _corr(a, b):
        a = a.astype(float, copy=False)
        b = b.astype(float, copy=False)
        a = a - a.mean()
        b = b - b.mean()
        da = np.sqrt(np.sum(a * a))
        db = np.sqrt(np.sum(b * b))
        if da < eps or db < eps: # avoid zero in the denominator
            return 0.0
        return float(np.dot(a, b) / (da * db))

    # Initialize the rdm output
    rdm = np.zeros((n_cond, n_cond), dtype=float)

    # fill each (i, j) by averging cross-run correlations
    for i in range(n_cond):
        for j in range(n_cond):
            s = 0.0
            c = 0
            for ra in runs:
                for rb in runs:
                    if ra == rb:
                        continue  # EXCLUDE within-run
                    ka = idx.get((ra, i), None) # default to "None"
                    kb = idx.get((rb, j), None)
                    if ka is None or kb is None:
                        continue
                    s += _corr(patterns[ka], patterns[kb])
                    c += 1
            rdm[i, j] = (s / c) if c > 0 else 0.0

    np.fill_diagonal(rdm, 1.0) # enforce diagonal to be 1 & not including afterwards
    return rdm


def compute_crossnobis_neural_rdm(
    patterns: np.ndarray,
    run_ids: np.ndarray,
    cond_ids: np.ndarray,
    noise=None,
):
    """
    Compute an (n_cond x n_cond) crossnobis *distance* matrix using rsatoolbox.

    patterns: (n_samples, n_voxels), each sample is one (run, condition) map
    run_ids:  (n_samples,) run label per sample (used as CV folds)
    cond_ids: (n_samples,) condition index [0..n_cond-1] per sample
    noise: optional noise normalization info (start with None to get it running)
    """
    import rsatoolbox as rsa

    # note: here n_samples = n_runs * n_conditions
    patterns = np.asarray(patterns, dtype=float)

    # If any non-finite values, crossnobis will blow up in rsatoolbox
    if not np.isfinite(patterns).all():
        bad = np.sum(~np.isfinite(patterns))
        print(f"[crossnobis] non-finite patterns: {bad}/{patterns.size}", flush=True)
        # safest for searchlight: return an RDM full of NaNs (or zeros) so scoring returns 0
        n_cond = len(np.unique(cond_ids))
        rdm = np.full((n_cond, n_cond), np.nan, dtype=float)
        return rdm

    run_ids = np.asarray(run_ids).astype(int)
    cond_ids = np.asarray(cond_ids).astype(int)
    # revise here: not forcing run_ids and cond_ids to int
    # instead make them strings, so numpy will not report error

    ds = rsa.data.Dataset(
        measurements=patterns,
        obs_descriptors={
            "run": run_ids,
            "cond": cond_ids,
        },
    )

    rdm_obj = rsa.rdm.calc_rdm(
        ds,
        method="crossnobis",
        descriptor="cond",
        cv_descriptor="run",
        noise=noise,
    )

    # rsatoolbox stores RDMs in vector form; get square
    rdm = rdm_obj.get_matrices()[0]  # shape (n_cond, n_cond)

    return rdm


   