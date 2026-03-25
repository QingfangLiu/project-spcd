"""Model RDMs for image-wise RSA."""

from __future__ import annotations

import numpy as np

canonical_condition_order = [
    "A11", "A21", "C1", "B11", "B21", "D1",
    "A12", "A22", "C2", "B12", "B22", "D2",
]

_ODOR_GROUPS = {
    "odor1": {"A11", "B11", "A12", "B12"},
    "odor2": {"A21", "B21", "A22", "B22"},
    "clean": {"C1", "D1", "C2", "D2"},
}


def condition_to_group(condition: str) -> str:
    for group, conds in _ODOR_GROUPS.items():
        if condition in conds:
            return group
    raise KeyError(f"Unknown condition: {condition}")


def create_identity_model_rdm(
    condition_order: list[str] = canonical_condition_order,
) -> np.ndarray:
    """Identity model: 0 within odor-group, 1 across odor-groups."""
    n = len(condition_order)
    rdm = np.ones((n, n), dtype=float)
    np.fill_diagonal(rdm, 0.0)

    group_map = {c: condition_to_group(c) for c in condition_order}
    for i in range(n):
        for j in range(i):
            rdm[i, j] = 0.0 if group_map[condition_order[i]] == group_map[condition_order[j]] else 1.0
            rdm[j, i] = rdm[i, j]
    return rdm


def create_button_rdm(button_labels: list[str]) -> np.ndarray:
    """Button nuisance RDM: 0 if same button label, 1 otherwise."""
    labels = [str(x).strip().lower() for x in button_labels]
    n = len(labels)
    rdm = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i):
            rdm[i, j] = 0.0 if labels[i] == labels[j] else 1.0
            rdm[j, i] = rdm[i, j]
    np.fill_diagonal(rdm, 0.0)
    return rdm
