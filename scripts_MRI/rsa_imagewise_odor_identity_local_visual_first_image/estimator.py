"""Searchlight estimator with a searchlight-local independent visual RDM."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from .neural_rdm import compute_correlation_rdm
from .rsa_stats import rsa_multiple_regression, vectorize_lower_triangle
from .visual_rdm import create_local_visual_image_rdm_from_patterns


class LocalVisualSingleModelRSA(BaseEstimator):
    def __init__(
        self,
        fixed_model_rdms=None,
        target_name=None,
        n_probe_samples=None,
        probe_run_ids=None,
        probe_sample_labels=None,
        localizer_run_ids=None,
        localizer_condition_labels=None,
        localizer_condition_order=None,
        center_probe_within_run: bool = False,
        visual_zscore_across_trials: bool = True,
        min_voxels: int = 3,
        eps: float = 1e-8,
    ):
        self.fixed_model_rdms = fixed_model_rdms
        self.target_name = target_name
        self.n_probe_samples = n_probe_samples
        self.probe_run_ids = probe_run_ids
        self.probe_sample_labels = probe_sample_labels
        self.localizer_run_ids = localizer_run_ids
        self.localizer_condition_labels = localizer_condition_labels
        self.localizer_condition_order = localizer_condition_order
        self.center_probe_within_run = center_probe_within_run
        self.visual_zscore_across_trials = visual_zscore_across_trials
        self.min_voxels = min_voxels
        self.eps = eps

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
        run_ids = np.asarray(run_ids)
        n = len(run_ids)
        pair_mask_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i):
                pair_mask_matrix[i, j] = run_ids[i] != run_ids[j]
        return pair_mask_matrix[np.tril_indices(n, k=-1)]

    @staticmethod
    def _row_norms_after_center(patterns: np.ndarray) -> np.ndarray:
        centered = patterns - patterns.mean(axis=1, keepdims=True)
        return np.linalg.norm(centered, axis=1)

    def _nondegenerate_visual_model(self, visual_rdm: np.ndarray, pair_mask: np.ndarray) -> bool:
        vec = vectorize_lower_triangle(visual_rdm, pair_mask=pair_mask)
        if not np.isfinite(vec).all():
            return False
        return bool(np.std(vec) >= self.eps)

    def score(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D patterns array, got shape {X.shape}")

        n_probe = int(self.n_probe_samples)
        if X.shape[0] <= n_probe:
            raise ValueError("Combined searchlight input must include probe and localizer rows.")

        probe_run_ids = np.asarray(self.probe_run_ids)
        localizer_run_ids = np.asarray(self.localizer_run_ids)
        localizer_condition_labels = np.asarray(self.localizer_condition_labels, dtype=object)

        probe_X = X[:n_probe, :]
        localizer_X = X[n_probe:, :]
        if probe_X.shape[0] != probe_run_ids.shape[0]:
            raise ValueError("Probe rows must match probe_run_ids.")
        if localizer_X.shape[0] != localizer_run_ids.shape[0]:
            raise ValueError("Localizer rows must match localizer_run_ids.")
        if localizer_X.shape[0] != localizer_condition_labels.shape[0]:
            raise ValueError("Localizer rows must match localizer_condition_labels.")

        finite_voxels = np.isfinite(probe_X).all(axis=0) & np.isfinite(localizer_X).all(axis=0)
        if int(finite_voxels.sum()) < int(self.min_voxels):
            return np.nan

        probe_X = probe_X[:, finite_voxels]
        localizer_X = localizer_X[:, finite_voxels]

        if self.center_probe_within_run:
            probe_X = self._center_within_run(probe_X, probe_run_ids)

        if np.any(self._row_norms_after_center(probe_X) < self.eps):
            return np.nan

        visual_rdm = create_local_visual_image_rdm_from_patterns(
            localizer_patterns=localizer_X,
            localizer_run_ids=localizer_run_ids,
            localizer_condition_labels=localizer_condition_labels,
            condition_order=self.localizer_condition_order,
            sample_labels=self.probe_sample_labels,
            zscore_across_trials=self.visual_zscore_across_trials,
            eps=self.eps,
        )
        if visual_rdm is None:
            return np.nan

        neural_rdm = compute_correlation_rdm(probe_X)
        if not np.isfinite(neural_rdm).all():
            return np.nan

        pair_mask = self._make_cross_run_pair_mask(probe_run_ids)
        if not self._nondegenerate_visual_model(visual_rdm, pair_mask=pair_mask):
            return np.nan

        model_rdms = dict(self.fixed_model_rdms)
        model_rdms["visual_image"] = visual_rdm
        ordered_model_rdms = {
            "odor_identity": model_rdms["odor_identity"],
            "visual_image": model_rdms["visual_image"],
        }

        betas = rsa_multiple_regression(
            neural_rdm,
            ordered_model_rdms,
            pair_mask=pair_mask,
        )
        return float(betas[self.target_name])
