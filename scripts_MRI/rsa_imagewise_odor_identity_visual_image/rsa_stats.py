"""RSA regression utilities."""

import numpy as np
from sklearn.linear_model import LinearRegression


def vectorize_lower_triangle(rdm: np.ndarray, pair_mask=None) -> np.ndarray:
    n = rdm.shape[0]
    tri = np.tril_indices(n, k=-1)
    vec = np.asarray(rdm, dtype=float)[tri]

    if pair_mask is not None:
        pair_mask = np.asarray(pair_mask, dtype=bool)
        if pair_mask.shape != vec.shape:
            raise ValueError(
                f"pair_mask shape {pair_mask.shape} does not match "
                f"vectorized lower triangle {vec.shape}."
            )
        vec = vec[pair_mask]

    return vec


def rsa_multiple_regression(
    neural_rdm: np.ndarray,
    model_rdms: dict,
    pair_mask=None,
) -> dict:
    """
    OLS regression of the neural RDM on one or more model RDMs.

    pair_mask should be a boolean vector with length equal to the
    lower triangle of the RDM, indicating which pairs to keep.
    """
    y = vectorize_lower_triangle(neural_rdm, pair_mask=pair_mask)
    X = np.column_stack([
        vectorize_lower_triangle(rdm, pair_mask=pair_mask)
        for rdm in model_rdms.values()
    ])

    if X.shape[0] == 0:
        raise ValueError("No RDM pairs available for regression.")

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)

    return {
        name: float(beta)
        for name, beta in zip(model_rdms.keys(), reg.coef_)
    }