# estimator.py

import numpy as np
from sklearn.base import BaseEstimator # provides scikit-learn conventions
from .neural_rdm import compute_crossnobis_neural_rdm, compute_crossrun_neural_rdm
from .rsa_stats import rsa_multiple_regression, rsa_single_model_correlation

class SingleModelRSA(BaseEstimator):

	def __init__( # put all hyperparameters here
		self,
		model_rdms=None, # dictionary, pre-calculated model rdms (one or multiple)
		target_name=None, # model name, also keys of the model_rdms dictionary
		neural_metric: str = "crossnobis", # "corr" or "crossnobis"
		rsa_method: str = "spearman",      # "regression", "spearman" or "kendall"
		odor_mask=None,
		run_ids=None,  # dim: (n_samples,)
		cond_ids=None, # dim: (n_samples,)
		n_conditions: int = None, # number of conditions
	):

		self.model_rdms = model_rdms
		self.target_name = target_name

		self.neural_metric = neural_metric
		self.rsa_method = rsa_method
		self.odor_mask = odor_mask

		self.run_ids = None if run_ids is None else np.asarray(run_ids)
		self.cond_ids = None if cond_ids is None else np.asarray(cond_ids)
		self.n_conditions = n_conditions

	def fit(self, X, y=None): # X: (n_samples, n_voxels) created from searchlight loop
		return self # returning self to satisfy the estimator API

	def score(self, X, y=None):
		if X.shape[1] < 3: # if searchlight has fewer than 3 voxels
			return 0.0

		if self.run_ids is None or self.cond_ids is None or self.n_conditions is None:
			raise ValueError("cross run RSA requires run_ids, cond_ids, and n_conditions")

		if X.shape[0] != self.run_ids.shape[0] or X.shape[0] != self.cond_ids.shape[0]:
			raise ValueError("X rows must match run_ids/cond_ids lengths")

		if self.neural_metric == "corr": # do cross-run correlation
			neural_rdm = compute_crossrun_neural_rdm(
				patterns=X,
				run_ids=self.run_ids,
				cond_ids=self.cond_ids,
				)
		elif self.neural_metric == "crossnobis":
			neural_rdm, conds = compute_crossnobis_neural_rdm(
				patterns=X,
				run_ids=self.run_ids,
				cond_ids=self.cond_ids,
				)
		else:
			raise ValueError(f"Unknown neural_metric: {self.neural_metric}")

		# RSA scoring
		if self.rsa_method == "regression":
			betas = rsa_multiple_regression(
				neural_rdm,
				self.model_rdms,
				odor_mask=self.odor_mask,
				)
			return float(betas[self.target_name])

		if self.rsa_method in ("spearman", "kendall"):
			model_rdm = self.model_rdms[self.target_name]
			return rsa_single_model_correlation(
				neural_rdm,
				model_rdm,
				method=self.rsa_method,
				odor_mask=self.odor_mask
				)

		raise ValueError(f"Unknown rsa_method: {self.rsa_method}")
		# here raise error works like an implicit else
		# because the function expects a return in the end












