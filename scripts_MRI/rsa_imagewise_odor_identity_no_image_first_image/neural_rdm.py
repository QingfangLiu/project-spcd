"""Neural RDM construction for image-wise beta patterns."""

from __future__ import annotations

import numpy as np

def compute_correlation_rdm(patterns: np.ndarray) -> np.ndarray:
    """Pairwise correlation distance RDM across sample patterns."""
    # CHANGE: vectorized implementation
    X = np.asarray(patterns, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")

    # center each sample pattern
    X = X - X.mean(axis=1, keepdims=True)

    # normalize each sample pattern
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # correlation similarity matrix
    sim = X @ X.T
    sim = np.clip(sim, -1.0, 1.0)

    # convert to correlation distance
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm

