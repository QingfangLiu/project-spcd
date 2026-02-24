"""
Model RDM construction for 12-condition SPCD probe task.

All RDMs are aligned to canonical_condition_order.

Models:
- identity (odor group identity)
- association (paired cues)
- value (odor vs clean)
"""

import numpy as np

canonical_condition_order = ["A11", "A21", "C1", "B11", "B21", "D1",
							 "A12", "A22", "C2", "B12", "B22", "D2"]


def create_identity_model_rdm():

	odor_groups = {
		'odor1': ['A11', 'B11', 'A12', 'B12'],
		'odor2': ['A21', 'B21', 'A22', 'B22'],
		'clean': ['C1', 'D1', 'C2', 'D2']
	}

	# map each condition -> group name
	cond2group = {}

	for gname, conds in odor_groups.items():
		for c in conds:
			cond2group[c] = gname

	n = len(canonical_condition_order)
	rdm = np.ones((n, n), dtype=float) # initialize rdm to all ones

	for i in range(n):
		for j in range(i):
			same_group = (cond2group[canonical_condition_order[i]] == cond2group[canonical_condition_order[j]])
			rdm[i, j] = 0.0 if same_group else 1.0
			rdm[j, i] = rdm[i, j]

	np.fill_diagonal(rdm, 0.0)
	return rdm


def create_association_model_rdm():

	pairs = [
		('A11', 'B11'), 
		('A21', 'B21'), 
		('C1', 'D1'),
		('A12', 'B12'), 
		('A22', 'B22'), 
		('C2', 'D2'),
	]

	n = len(canonical_condition_order)
	rdm = np.ones((n, n), dtype=float)   # initialize rdm to all ones
	np.fill_diagonal(rdm, 0.0)

	for i, cond_i in enumerate(canonical_condition_order):
		for j, cond_j in enumerate(canonical_condition_order):
			if i == j:
				continue
			# paired association
			elif (cond_i, cond_j) in pairs or (cond_j, cond_i) in pairs: # allowing symmetric model
				rdm[i, j] = 0.0
	return rdm


def create_value_model_rdm():

	odor_merged = {"A11", "A21", "B11", "B21", "A12", "A22", "B12", "B22"}
	clean = {"C1", "D1", "C2", "D2"}

	group_map = {}
	for c in odor_merged:
		group_map[c] = "odor"
	for c in clean:
		group_map[c] = "clean"

	n = len(canonical_condition_order)
	rdm = np.ones((n, n), dtype=float) # initialize rdm to all ones
	np.fill_diagonal(rdm, 0.0)

	for i, cond_i in enumerate(canonical_condition_order):
		for j, cond_j in enumerate(canonical_condition_order):
			if i == j:
				continue
			else: 
				rdm[i, j] = 0.0 if (group_map[cond_i] == group_map[cond_j]) else 1.0
	return rdm



