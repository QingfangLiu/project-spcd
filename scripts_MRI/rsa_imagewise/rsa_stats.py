"""RSA regression utilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

def vectorize_lower_triangle(rdm: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    n = rdm.shape[0]
    tri = np.tril_indices(n, k=-1)
    vec = np.asarray(rdm, dtype=float)[tri]
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != vec.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match vectorized lower triangle {vec.shape}."
            )
        vec = vec[mask]
    return vec


def rsa_multiple_regression(
    neural_rdm: np.ndarray,
    model_rdms: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """OLS regression of the neural RDM on one or more model RDMs."""
    y = vectorize_lower_triangle(neural_rdm, mask=mask)
    X = np.column_stack([vectorize_lower_triangle(rdm, mask=mask) for rdm in model_rdms.values()])

    if X.shape[0] == 0:
        raise ValueError("No RDM pairs available for regression.")

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    return {name: float(beta) for name, beta in zip(model_rdms.keys(), reg.coef_)}
