"""
neural_rdm.py

Neural RDM construction utilities for RSA.

This module provides:
1) compute_crossnobis_neural_rdm:
   Cross-validated distance-like dissimilarities between conditions across independent runs.

   Implementation details:
   - For each (run, condition), compute the mean activation pattern across samples
     (supports either condition-wise betas or multiple samples such as single-trial betas).
   - For each condition pair (i, j), use leave-one-run-out cross-validation:
       d_train = mean_train(i) - mean_train(j)
       d_test  = test_run(i)   - test_run(j)
       score   = dot(d_train, d_test)
     and average score across test runs where both conditions are present.

   Interpretation note:
   - The dot-product form corresponds to a crossnobis (cross-validated Mahalanobis)
     distance only when patterns are noise-normalized / whitened (e.g., using ResMS-based
     whitening or a full noise covariance estimate). Otherwise it is still a valid
     cross-validated dissimilarity measure but its scale depends on voxel variance and
     number of voxels.

2) compute_crossrun_neural_rdm:
   Cross-run Pearson-correlation based dissimilarity (1 - r), computed only across
   different runs to avoid within-run noise inflation.

Both functions return an (n_cond x n_cond) symmetric matrix with a 0 diagonal.
"""

import numpy as np

def compute_crossnobis_neural_rdm(
	patterns: np.ndarray, # (n_samples, n_voxels)
	run_ids: np.ndarray,  # (n_samples,)
	cond_ids: np.ndarray, # (n_samples,)
	):

	patterns = np.asarray(patterns, dtype=float)
	run_ids = np.asarray(run_ids)
	cond_ids = np.asarray(cond_ids)

	n_samples, n_vox = patterns.shape

	runs = np.unique(run_ids)   # unique runs
	conds = np.unique(cond_ids) # unique conditions (sorted)

	n_runs = runs.size
	n_cond = conds.size

	if n_runs < 2:
		raise ValueError("Crossnobis requires >= 2 independent runs.")

	# build means per (run, cond): M[r, c, v]
	# when my data are estimated per condition per run, averaging becomes not necessary
	# but this can be considered as a generic function
	# for example, if I want to do single-trial beta estimates and then use them
	# for crossnobis

	M = np.full((n_runs, n_cond, n_vox), np.nan, dtype=float)

	for r_i, r in enumerate(runs):
		for c_i, c in enumerate(conds):
			mask = (run_ids == r) & (cond_ids == c)
			if np.any(mask):
				M[r_i, c_i] = patterns[mask].mean(axis=0) # a vector

	rdm = np.zeros((n_cond, n_cond), dtype=float)

	for i in range(n_cond):
		for j in range(i + 1, n_cond):
			vals = []
			for t in range(n_runs): # test run index
				xi_t = M[t, i] # condition i pattern
				xj_t = M[t, j] # condition j pattern

				if np.any(np.isnan(xi_t)) or np.any(np.isnan(xj_t)):
					continue

				train_idx = np.arange(n_runs) != t
				xi_tr = M[train_idx, i, :]
				xj_tr = M[train_idx, j, :]

				# averaging across training runs
				mu_i = np.nanmean(xi_tr, axis=0) 
				mu_j = np.nanmean(xj_tr, axis=0)

				if np.any(np.isnan(mu_i)) or np.any(np.isnan(mu_j)):
					continue

				d_train = mu_i - mu_j
				d_test = xi_t - xj_t

				vals.append(float(np.dot(d_train, d_test)))

			rdm[i, j] = np.nan if len(vals) == 0 else float(np.mean(vals))
			rdm[j, i] = rdm[i, j]


	np.fill_diagonal(rdm, 0.0)
	return rdm, conds # return conds because rdm corresponds to a specific ordering of conditions


# this one computes using cross run correlations
# keep 1-r in the end to make this a dissimilarity matrix

def compute_crossrun_neural_rdm(
	patterns: np.ndarray, # (n_samples, n_voxels)
	run_ids: np.ndarray,  # (n_samples, )
	cond_ids: np.ndarray  # (n_samples, )
	):

	patterns = np.asarray(patterns)
	run_ids = np.asarray(run_ids).astype(int)
	cond_ids = np.asarray(cond_ids).astype(int)

	# build a lookup table
	idx = {}
	for k in range(patterns.shape[0]):
		idx[(int(run_ids[k]), int(cond_ids[k]))] = k
	# indexed by (run, cond)

	# get sorted, unique runs
	runs = sorted(set(run_ids.tolist()))
	eps = 1e-12

	conds = np.unique(cond_ids) 	# unique conditions (sorted)
	n_cond = conds.size 			# number of conditions

	# a manual Pearson correlation
	def _corr(a, b):
		a = a.astype(float, copy=False) # convert to float but avoid copying memory if already float
		b = b.astype(float, copy=False)
		a = a - a.mean() # keep this to use on both numerator and denominator
		b = b - b.mean()
		da = np.sqrt(np.sum(a * a)) # a * a faster than a ** 2
		db = np.sqrt(np.sum(b * b))
		if da < eps or db < eps:
			return 0.0 # return zero when one variable has no variance
		return float(np.dot(a, b) / (da * db))

	# initialize the rdm output: (n_cond, n_cond)
	rdm = np.zeros((n_cond, n_cond), dtype=float)

	for i in range(n_cond):
		for j in range(n_cond):
			s = 0.0
			c = 0
			ci = int(conds[i])
			cj = int(conds[j])
			for ra in runs:
				for rb in runs:
					if ra == rb:
						continue   # not do within-run
					ka = idx.get((int(ra), ci), None)
					kb = idx.get((int(rb), cj), None)
					if ka is None or kb is None:
						continue
					s += _corr(patterns[ka], patterns[kb])
					c += 1
			rdm[i, j] = 1 - (s / c) if c > 0 else 0.0

	np.fill_diagonal(rdm, 0.0)
	return rdm

















	
















