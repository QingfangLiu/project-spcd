"""Model RDMs for premeal odor-identity no-image image-wise RSA."""

from __future__ import annotations

import numpy as np

_ODOR_GROUPS = {
    "odor1": {"A11", "B11", "A12", "B12"},
    "odor2": {"A21", "B21", "A22", "B22"},
}

_CONDITION_TO_GROUP = {
    cond: group
    for group, conds in _ODOR_GROUPS.items()
    for cond in conds
}


def base_condition(label: str) -> str:
    """Convert image-wise label like 'A11_img1' -> 'A11'."""
    return str(label).rsplit("_img", 1)[0]


def condition_to_group(condition: str) -> str:
    try:
        return _CONDITION_TO_GROUP[condition]
    except KeyError:
        raise KeyError(
            f"Unknown odor-predictive condition: {condition}. "
            "This variant excludes clean-air conditions."
        )


def create_odor_identity_rdm(sample_labels) -> np.ndarray:
    """
    Odor-identity model for odor-predictive image-wise samples.

    The RDM is 0 for samples linked to the same odor identity and 1 for samples
    linked to different odor identities. Clean-air cues are not represented in
    this model and should be removed by the loader before model construction.
    """
    groups = np.array(
        [condition_to_group(base_condition(lbl)) for lbl in sample_labels],
        dtype=object,
    )
    rdm = (groups[:, None] != groups[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm
