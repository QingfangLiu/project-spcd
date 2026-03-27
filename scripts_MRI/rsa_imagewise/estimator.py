import numpy as np
from sklearn.base import BaseEstimator

from .neural_rdm import compute_correlation_rdm
from .rsa_stats import rsa_multiple_regression


class SingleModelRSA(BaseEstimator):
    def __init__(
        self,
        model_rdms=None,
        target_name=None,
        run_ids=None,
        center_within_run: bool = False,
    ):
        self.model_rdms = model_rdms
        self.target_name = target_name
        self.run_ids = run_ids
        self.center_within_run = center_within_run

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _center_within_run(patterns: np.ndarray, run_ids) -> np.ndarray:
        patterns = np.asarray(patterns, dtype=float)
        run_ids = np.asarray(run_ids)
        centered = patterns.copy()
        for run in np.unique(run_ids):
            idx = np.where(run_ids == run)[0]
            if idx.size == 0:
                continue
            centered[idx] -= centered[idx].mean(axis=0, keepdims=True)
        return centered

    @staticmethod
    def _make_cross_run_pair_mask(run_ids) -> np.ndarray:
        """
        Return a boolean mask for the vectorized lower triangle.
        Keep only pairs where the two samples come from different runs.
        """
        run_ids = np.asarray(run_ids)
        n = len(run_ids)

        pair_mask_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i):
                pair_mask_matrix[i, j] = run_ids[i] != run_ids[j]

        tri = np.tril_indices(n, k=-1)
        return pair_mask_matrix[tri]

    def score(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D patterns array, got shape {X.shape}")
        if X.shape[1] < 3:
            return 0.0

        if self.run_ids is None:
            raise ValueError("RSA requires run_ids")

        run_ids = np.asarray(self.run_ids)
        if X.shape[0] != run_ids.shape[0]:
            raise ValueError("X rows must match run_ids lengths")

        if self.center_within_run:
            X = self._center_within_run(X, run_ids)

        neural_rdm = compute_correlation_rdm(X)
        cross_run_pair_mask = self._make_cross_run_pair_mask(run_ids)

        betas = rsa_multiple_regression(
                neural_rdm,
                self.model_rdms,
                pair_mask=cross_run_pair_mask,
            )
        return float(betas[self.target_name])

        