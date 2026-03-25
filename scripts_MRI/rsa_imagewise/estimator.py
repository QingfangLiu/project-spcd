import numpy as np
from sklearn.base import BaseEstimator

from .neural_rdm import compute_correlation_rdm
from .rsa_stats import rsa_multiple_regression


class SingleModelRSA(BaseEstimator):
    def __init__(
        self,
        model_rdms=None,
        target_name=None,
        odor_mask=None,
        run_ids=None,
        cond_ids=None,
        center_within_run: bool = False,
    ):
        self.model_rdms = model_rdms
        self.target_name = target_name
        self.odor_mask = odor_mask
        self.run_ids = run_ids
        self.cond_ids = cond_ids
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

    def score(self, X, y=None):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D patterns array, got shape {X.shape}")
        if X.shape[1] < 3:
            return 0.0

        if self.run_ids is None or self.cond_ids is None:
            raise ValueError("RSA requires run_ids and cond_ids")

        run_ids = np.asarray(self.run_ids)
        cond_ids = np.asarray(self.cond_ids)
        if X.shape[0] != run_ids.shape[0] or X.shape[0] != cond_ids.shape[0]:
            raise ValueError("X rows must match run_ids/cond_ids lengths")

        if self.center_within_run:
            X = self._center_within_run(X, run_ids)

        neural_rdm = compute_correlation_rdm(X)

        betas = rsa_multiple_regression(
                neural_rdm,
                self.model_rdms,
                mask=self.odor_mask,
            )
        return float(betas[self.target_name])

        