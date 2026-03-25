"""Neural RDM construction for image-wise beta patterns."""

from __future__ import annotations

import numpy as np


def _pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    """Pairwise correlation distance RDM across sample patterns."""
    patterns = np.asarray(patterns, dtype=float)
    n_samples = patterns.shape[0]
    rdm = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            r = _pearson_similarity(patterns[i], patterns[j])
            d = 1.0 - r
            rdm[i, j] = d
            rdm[j, i] = d

    np.fill_diagonal(rdm, 0.0)
    return rdm
