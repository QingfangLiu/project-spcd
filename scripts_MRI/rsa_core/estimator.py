"""
SearchLight-compatible RSA estimators.

This module defines scikit-learn–style estimator classes that
wrap RSA computations so they can be used as scoring functions
inside nilearn.decoding.SearchLight.

Estimators here operate on in-memory voxel patterns and delegate
all RSA logic to lower-level neural RDM and statistics modules.
"""

import numpy as np
from sklearn.base import BaseEstimator

from .neural_rdm import compute_neural_rdm
from .rsa_stats import rsa_multiple_regression, rsa_single_model_correlation
        
class SingleModelRSA(BaseEstimator): 
    """
    SearchLight-compatible estimator for single-model RSA.

    Given voxel patterns within a searchlight sphere, this estimator:
    (1) computes a neural RDM using the specified similarity metric,
    (2) compares it to one or more model RDMs using regression or correlation,
    (3) returns a scalar RSA score for the target model.

    Designed to be used as the `estimator` argument in nilearn.decoding.SearchLight.
    """
    def __init__(self, model_rdms=None, target_name=None,
                 rsa_method="regression", neural_metric="correlation",
                 pair_mask=None):
        self.model_rdms = model_rdms
        self.target_name = target_name
        self.rsa_method = rsa_method
        self.neural_metric = neural_metric
        self.pair_mask = pair_mask

    def fit(self, X, y=None): # y is accepted by ignored
        """
        No-op fit method required by scikit-learn interface.

        Parameters
        ----------
        X : np.ndarray
            Input data (ignored).
        y : np.ndarray or None
            Labels (ignored).

        Returns
        -------
        self
        """
        return self
            
    def score(self, X, y=None):
        """
        Compute RSA score for a single searchlight sphere.

        Parameters
        ----------
        X : np.ndarray (n_conditions, n_voxels)
            Condition-wise voxel patterns within the searchlight sphere.
        y : np.ndarray or None
            Dummy labels required by SearchLight API (ignored).

        Returns
        -------
        score : float
            RSA score (regression beta or correlation coefficient).
        """

        if X.shape[1] < 3:
            return 0.0
        
        neural_rdm = compute_neural_rdm(X, metric=self.neural_metric)

        if self.rsa_method == "regression":
            betas = rsa_multiple_regression(neural_rdm, self.model_rdms, pair_mask=self.pair_mask)
            return float(betas[self.target_name]) # return a scalar for the scoring function

        elif self.rsa_method in ("spearman", "kendall"):
            model_rdm = self.model_rdms[self.target_name]  # single target model
            return rsa_single_model_correlation(neural_rdm, model_rdm, method=self.rsa_method, pair_mask=self.pair_mask)

        else:
            raise ValueError(f"Unknown rsa_method: {self.rsa_method}")