
"""
Model representational dissimilarity matrices (RDMs).

This module defines hypothesis-driven model RDMs used in RSA,
including association, odor identity, and value-based models.

All functions here are pure and deterministic: they operate only
on condition labels and return NumPy arrays without performing
any I/O or interacting with neural data.
"""

import numpy as np

CANONICAL_CONDITION_ORDER = [
    "A11", "A21", "C1", "B11", "B21", "D1",
    "A12", "A22", "C2", "B12", "B22", "D2",
]

# ============================================================================
# Model RDM Creation
# ============================================================================

def create_association_model_rdm():
    """
    Create association model RDM (12x12).
    Value = 1 for within-set paired cues, 0 otherwise.
    
    Paired cues:
    - A11-B11, A21-B21, C1-D1 (Set 1)
    - A12-B12, A22-B22, C2-D2 (Set 2)
    
    Returns
    -------
    rdm : np.ndarray (12, 12)
        Similarity matrix (1 = paired, 0 = not paired)
    """
    conditions = ['A11', 'A21', 'C1', 'B11', 'B21', 'D1',
                  'A12', 'A22', 'C2', 'B12', 'B22', 'D2']
    
    # Define paired relationships
    pairs = [
        ('A11', 'B11'), ('A21', 'B21'), ('C1', 'D1'),
        ('A12', 'B12'), ('A22', 'B22'), ('C2', 'D2')
    ]
    
    n = len(conditions)
    rdm = np.zeros((n, n))
    
    for i, cond_i in enumerate(conditions):
        for j, cond_j in enumerate(conditions):
            if i == j:
                rdm[i, j] = 1  # Diagonal
            elif (cond_i, cond_j) in pairs or (cond_j, cond_i) in pairs: # allowing symmetric model
                rdm[i, j] = 1
    
    return rdm


def create_identity_model_rdm():
    """
    Create odor identity model RDM (12x12).
    Value = 1 for same odor identity, 0 otherwise.
    
    Same odor groups:
    - Odor1: A11, B11, A12, B12
    - Odor2: A21, B21, A22, B22
    - Clean: C1, D1, C2, D2
    
    Returns
    -------
    rdm : np.ndarray (12, 12)
        Similarity matrix (1 = same odor, 0 = different odor)
    """
    conditions = ['A11', 'A21', 'C1', 'B11', 'B21', 'D1',
                  'A12', 'A22', 'C2', 'B12', 'B22', 'D2']
    
    # Define odor groups
    odor_groups = {
        'odor1': ['A11', 'B11', 'A12', 'B12'],
        'odor2': ['A21', 'B21', 'A22', 'B22'],
        'clean': ['C1', 'D1', 'C2', 'D2']
    }
    
    n = len(conditions)
    rdm = np.zeros((n, n))
    
    for i, cond_i in enumerate(conditions):
        for j, cond_j in enumerate(conditions):
            if i == j:
                rdm[i, j] = 1  # Diagonal
            else:
                # Check if both conditions in same odor group
                for group_conds in odor_groups.values():
                    if cond_i in group_conds and cond_j in group_conds:
                        rdm[i, j] = 1
                        break
    
    return rdm


def create_value_model_rdm(diagonal: float = 1.0):
    """
    Create odor value model RDM (12x12) where odor1 and odor2 are treated
    as the SAME odor class (merged), and clean is separate.

    Value = 1 iff BOTH conditions are in the merged odor set (odor1 + odor2).
    Otherwise 0.

    Groups:
    - odor_merged: A11, B11, A12, B12, A21, B21, A22, B22
    - clean:       C1, D1, C2, D2
    """
    conditions = ['A11', 'A21', 'C1', 'B11', 'B21', 'D1',
                  'A12', 'A22', 'C2', 'B12', 'B22', 'D2']

    odor_merged = {'A11', 'B11', 'A12', 'B12', 'A21', 'B21', 'A22', 'B22'}
    clean       = {'C1', 'D1', 'C2', 'D2'}

    group_map = {}
    for c in odor_merged:
        group_map[c] = "odor"
    for c in clean:
        group_map[c] = "clean"

    n = len(conditions)
    rdm = np.zeros((n, n), dtype=float)

    for i, ci in enumerate(conditions):
        for j, cj in enumerate(conditions):
            if i == j:
                rdm[i, j] = diagonal
            else:
                rdm[i, j] = float(group_map[ci] == group_map[cj])

    return rdm


    