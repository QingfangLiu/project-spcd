
"""
rsa_stats.py

Statistical scoring functions for Representational Similarity Analysis (RSA).

This module provides two complementary RSA approaches:

1) Multiple-regression RSA
   - Fits the neural RDM as a linear combination of one or more model RDMs
     using ordinary least squares.
   - Returns regression coefficients (betas) for each model.
   - Suitable for testing unique model contributions (subject to collinearity).

2) Single-model correlation RSA
   - Computes rank-based correlation (Spearman or Kendall) between
     neural and model RDMs.
   - Robust and stable for searchlight analyses.
   - Recommended when models may be collinear.

Implementation details
----------------------
- Only the strict lower triangle (excluding diagonal) of RDMs is used.
- Optional masking (e.g., odor–odor pairs only) is supported.
- All returned scores are safe for searchlight usage (no NaNs).

Assumptions
-----------
- Neural and model RDMs are symmetric and aligned in condition order.
- Model RDMs encode dissimilarity (0 = similar, 1 = dissimilar).
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression

def rsa_multiple_regression(
	neural_rdm,
	model_rdms,
	odor_mask=None,
	):
	"""
	Multiple-regression RSA between a neural RDM and one or more model RDMs.

	This function performs ordinary least squares regression:

	    neural_RDM_vector = β1 * model1 + β2 * model2 + ... + ε

	where RDMs are vectorized using the strict lower triangle
	(excluding the diagonal).

	Parameters
	----------
	neural_rdm : np.ndarray, shape (n_cond, n_cond)
		Symmetric neural dissimilarity matrix.

	model_rdms : dict[str, np.ndarray]
		Dictionary mapping model name -> model RDM.
		Each model RDM must be symmetric and match the shape
		of `neural_rdm`.

	odor_mask : np.ndarray of bool, optional
		Boolean mask over lower-triangle pairs.
		If provided, regression is performed only on masked pairs
		(e.g., odor–odor comparisons only).

	Returns
	-------
	betas : dict[str, float]
		Regression coefficients for each model (no intercept returned).

	Notes
	-----
	- Intercept is included in the regression model but not returned.
	- No regularization is applied (ordinary least squares).
	- If model RDMs are highly collinear, beta estimates may be unstable.
	- Interpretation assumes model RDMs are dissimilarity matrices
	  aligned to the neural RDM.
	"""

	n = neural_rdm.shape[0] # number of conditions

	tril_idx = np.tril_indices(n, k=-1) # k=-1 to exclude diagonal
	y = neural_rdm[tril_idx]

	# stack model RDM vectors
	X = np.column_stack([
		model_rdm[tril_idx]
		for model_rdm in model_rdms.values()
		]) # dim of X: (n_pairs, n_models)

	if odor_mask is not None:
		y = y[odor_mask]
		X = X[odor_mask, :]

	# fit linear regression
	reg = LinearRegression(fit_intercept=True)
	reg.fit(X, y)

	betas = {name: beta for name, beta in zip(model_rdms.keys(), reg.coef_)}

	return betas


def rsa_single_model_correlation(
	neural_rdm,
	model_rdm,
	method: str = "spearman",
	odor_mask=None,
	):
	"""
	Single-model RSA using rank-based correlation.

	Computes correlation between the vectorized lower triangle
	(excluding diagonal) of the neural RDM and a single model RDM.

	Parameters
	----------
	neural_rdm : np.ndarray, shape (n_cond, n_cond)
		Symmetric neural dissimilarity matrix.

	model_rdm : np.ndarray, shape (n_cond, n_cond)
		Symmetric model dissimilarity matrix aligned to `neural_rdm`.

	method : {"spearman", "kendall"}, default="spearman"
		Correlation method:
		- "spearman": Spearman rank correlation
		- "kendall": Kendall's tau

	odor_mask : np.ndarray of bool, optional
		Boolean mask over lower-triangle pairs.
		If provided, correlation is computed only on masked pairs.

	Returns
	-------
	score : float
		Correlation coefficient. Returns 0.0 if:
		- fewer than 2 conditions,
		- input is constant,
		- correlation is undefined (NaN).

	Notes
	-----
	- Diagonal elements are excluded from analysis.
	- Designed to be safe inside searchlight loops (never returns NaN).
	- Interpretation assumes model RDM encodes dissimilarity
	  (0 = similar, 1 = dissimilar).
	"""

	n = neural_rdm.shape[0]

	if n < 2:
		return 0.0

	tril_idx = np.tril_indices(n, k=-1) # k=-1 to exclude diagonal

	y = np.asarray(neural_rdm[tril_idx], dtype=float)
	x = np.asarray(model_rdm[tril_idx], dtype=float)

	if odor_mask is not None:
		y = y[odor_mask]
		x = x[odor_mask]

	# check constant input
	if np.all(x == x[0]) or np.allclose(y, y[0], atol=0.0, rtol=0.0):
		return 0.0

	if method == "spearman":
		r, _ = spearmanr(x, y)
	elif method == "kendall":
		r, _ = kendalltau(x, y)
	else:
		raise ValueError(f"Unknown RSA correlation method: {method}")

	# be safe in searchlight: return 0.0 if nan
	return 0.0 if np.isnan(r) else float(r)

	







