
"""
Statistical comparison between neural and model RDMs.

This module implements RSA scoring methods, including multiple
regression RSA and single-model correlation-based RSA.

All functions assume RDM inputs and return scalar similarity
metrics. 
"""

import numpy as np

from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression

# ============================================================================
# RSA Multiple Regression
# ============================================================================

def rsa_multiple_regression(
    neural_rdm, 
    model_rdms, 
    pair_mask=None):
    """
    Fit multiple regression: neural_RDM = β1*model1 + β2*model2 + ... + ε
    
    Uses lower triangle (excluding diagonal) vectorized values.
    
    Parameters
    ----------
    neural_rdm : np.ndarray (n, n)
        Neural similarity matrix
    model_rdms : dict
        Dictionary of model RDMs, e.g., {'association': rdm1, 'identity': rdm2}
    
    Returns
    -------
    betas : dict
        Regression coefficients for each model
    """
    n = neural_rdm.shape[0]
    
    # Vectorize lower triangle (excluding diagonal)
    tril_idx = np.tril_indices(n, k=-1) # k=-1: exclude the diagonal
    y = neural_rdm[tril_idx]
    
    # Stack model RDM vectors
    X = np.column_stack([model_rdm[tril_idx] for model_rdm in model_rdms.values()])

    # optionally drop pairs not of interest
    if pair_mask is not None:
        y = y[pair_mask]
        X = X[pair_mask, :]
    
    # Fit linear regression
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    
    # Get betas
    betas = {name: beta for name, beta in zip(model_rdms.keys(), reg.coef_)}
    
    return betas


# ============================================================================
# RSA single model correlation
# ============================================================================

def rsa_single_model_correlation(
    neural_rdm, 
    model_rdm, 
    method: str = "spearman", 
    pair_mask=None):
    """
    Correlate neural similarity matrix with ONE model similarity matrix
    using lower triangle (excluding diagonal).
    
    method: "spearman" or "kendall" (kendall = tau-b in scipy)
    """

    if neural_rdm.shape != model_rdm.shape:
        raise ValueError(f"Shape mismatch: neural_rdm {neural_rdm.shape} vs model_rdm {model_rdm.shape}")

    n = neural_rdm.shape[0]

    if n < 2:
        return 0.0

    tril_idx = np.tril_indices(n, k=-1)

    # Force float arrays (avoids weird dtype edge-cases)
    y = np.asarray(neural_rdm[tril_idx], dtype=float)
    x = np.asarray(model_rdm[tril_idx], dtype=float)

    # drop pairs not of interest
    if pair_mask is not None:
        y = y[pair_mask]
        x = x[pair_mask]

    # Constant input -> correlation undefined (prevents ConstantInputWarning)
    # Use exact check because x is typically 0/1 model distances; for y a tolerance helps.
    if np.all(x == x[0]) or np.allclose(y, y[0], atol=0.0, rtol=0.0):
        return 0.0

    if method == "spearman":
        r, _ = spearmanr(x, y)
    elif method == "kendall":
        r, _ = kendalltau(x, y)  # scipy's kendalltau is tau-b (tie-corrected)
    else:
        raise ValueError(f"Unknown RSA correlation method: {method}")

    # be safe in searchlight: return 0.0 if nan
    return 0.0 if np.isnan(r) else float(r)







