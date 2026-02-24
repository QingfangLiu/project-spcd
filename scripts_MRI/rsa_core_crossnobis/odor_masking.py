
"""
build_odor_mask

Constructs boolean mask over lower-triangle RDM pairs,
selecting only odor–odor comparisons.
"""

import numpy as np
def build_odor_mask(condition_order):

	"""
	Construct boolean mask for odor-odor pairs

	returns:
	mask: np.darray(n_pairs,)
		boolean mask indicating which condition pairs are both odor cues
	"""

	# cues that are associated with odors
	odor_cues = {'A11', 'B11', 'A12', 'B12', 'A21', 'B21', 'A22', 'B22'}

	n = len(condition_order)
	i, j = np.tril_indices(n, k=-1) # returns index pairs for the strict lower triangle
	return np.array(
		[(condition_order[ii] in odor_cues) and (condition_order[jj] in odor_cues)
		for ii, jj in zip(i, j)], # for each pair, check if both are odor cues
		dtype=bool
	)