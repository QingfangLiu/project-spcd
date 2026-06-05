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


def create_button_presence_rdm(button_labels):
    """
    RDM for whether a button was pressed.

    Groups:
      - 'left'  -> pressed
      - 'right' -> pressed
      - 'none'  -> none
    """
    labels = [str(x).strip().lower() for x in button_labels]
    pressed = np.array([lab in ("left", "right") for lab in labels], dtype=int)

    rdm = (pressed[:, None] != pressed[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def create_button_side_rdm(button_labels):
    """
    RDM for button side with 'none' treated as different from button presses.

    Rules:
      left vs right = 1
      left vs left = 0
      right vs right = 0
      anything involving 'none' = 1
      diagonal = 0
    """
    labels = np.array([str(x).strip().lower() for x in button_labels], dtype=object)
    n = len(labels)
    rdm = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i):
            li = labels[i]
            lj = labels[j]

            if li == "none" or lj == "none":
                val = 1.0
            else:
                val = 0.0 if li == lj else 1.0

            rdm[i, j] = val
            rdm[j, i] = val

    np.fill_diagonal(rdm, 0.0)
    return rdm
