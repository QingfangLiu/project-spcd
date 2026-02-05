# estimator.py

"""
SearchLight-compatible RSA estimators.

This estimator computes neural RDMs using cross-run correlations only
(no within-run pairs) to avoid inflated similarity from temporal autocorrelation.
"""

import numpy as np
from sklearn.base import BaseEstimator

from .neural_rdm import compute_crossrun_neural_rdm
from .rsa_stats import rsa_multiple_regression, rsa_single_model_correlation


class SingleModelRSA(BaseEstimator):
    """
    SearchLight-compatible estimator for single-model RSA.
    """
    def __init__(
        self,
        model_rdms=None,
        target_name=None,
        rsa_method="regression",
        pair_mask=None,
        run_ids=None,
        cond_ids=None,
        n_conditions: int = None,
    ):
        self.model_rdms = model_rdms
        self.target_name = target_name
        self.rsa_method = rsa_method
        self.pair_mask = pair_mask

        self.run_ids = None if run_ids is None else np.asarray(run_ids)
        self.cond_ids = None if cond_ids is None else np.asarray(cond_ids)
        self.n_conditions = n_conditions

    def fit(self, X, y=None):
        return self

    def score(self, X, y=None):
        if X.shape[1] < 3:
            return 0.0

        if self.run_ids is None or self.cond_ids is None or self.n_conditions is None:
            raise ValueError("crossrun_only=True requires run_ids, cond_ids, and n_conditions")
        if X.shape[0] != self.run_ids.shape[0] or X.shape[0] != self.cond_ids.shape[0]:
            raise ValueError("X rows must match run_ids/cond_ids lengths")

        neural_rdm = compute_crossrun_neural_rdm(
            patterns=X,
            run_ids=self.run_ids,
            cond_ids=self.cond_ids,
            n_cond=int(self.n_conditions),
        )

        # RSA scoring
        if self.rsa_method == "regression":
            betas = rsa_multiple_regression(neural_rdm, self.model_rdms, pair_mask=self.pair_mask)
            return float(betas[self.target_name])

        if self.rsa_method in ("spearman", "kendall"):
            model_rdm = self.model_rdms[self.target_name]
            return rsa_single_model_correlation(
                neural_rdm, model_rdm, method=self.rsa_method, pair_mask=self.pair_mask
            )

        raise ValueError(f"Unknown rsa_method: {self.rsa_method}")
