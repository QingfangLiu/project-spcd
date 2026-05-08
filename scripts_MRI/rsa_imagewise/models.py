"""Model RDMs for image-wise RSA."""

from __future__ import annotations

import numpy as np

_ODOR_GROUPS = {
    "odor1": {"A11", "B11", "A12", "B12"},
    "odor2": {"A21", "B21", "A22", "B22"},
    "clean": {"C1", "D1", "C2", "D2"},
}


# Build once from the readable group definition above
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
        raise KeyError(f"Unknown condition: {condition}")


def create_odor_group_rdm(sample_labels) -> np.ndarray:
    """
    Odor-group model for image-wise samples:
    0 within odor-group, 1 across odor-groups.
    """
    groups = np.array(
        [condition_to_group(base_condition(lbl)) for lbl in sample_labels],
        dtype=object,
    )
    rdm = (groups[:, None] != groups[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def create_odor_value_rdm(
    sample_labels,
    ratings_by_group,
    normalize: bool = True,
) -> np.ndarray:
    """
    Odor-value model for image-wise samples, derived from pleasantness ratings.

    Parameters
    ----------
    sample_labels : sequence of str
        Image-wise labels such as 'A11_img1'.
    ratings_by_group : dict
        Mapping with keys {'odor1', 'odor2', 'clean'} and numeric values.
    normalize : bool, default=True
        If True, z-score the three group ratings before expanding to samples.

    Returns
    -------
    np.ndarray
        Pairwise absolute odor-value-distance RDM aligned to sample_labels.
    """
    group_values = _prepare_group_values(ratings_by_group, normalize=normalize)

    sample_values = np.array(
        [
            group_values[condition_to_group(base_condition(lbl))]
            for lbl in sample_labels
        ],
        dtype=float,
    )

    rdm = np.abs(sample_values[:, None] - sample_values[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def create_odor_value_rdm_by_phase(
    sample_labels,
    phase_labels,
    ratings_by_phase,
    normalize: bool = True,
) -> np.ndarray:
    """
    Odor-value model for combined pre/post samples.

    Ratings are normalized within each phase before expanding to samples,
    matching the single-session odor_value behavior.
    """
    if len(sample_labels) != len(phase_labels):
        raise ValueError("sample_labels and phase_labels must have the same length")

    group_values_by_phase = {
        phase: _prepare_group_values(ratings_by_group, normalize=normalize)
        for phase, ratings_by_group in ratings_by_phase.items()
    }

    missing_phases = set(phase_labels) - set(group_values_by_phase.keys())
    if missing_phases:
        raise ValueError(
            "ratings_by_phase is missing required phase keys: "
            f"{sorted(missing_phases)}"
        )

    sample_values = np.array(
        [
            group_values_by_phase[phase][condition_to_group(base_condition(lbl))]
            for lbl, phase in zip(sample_labels, phase_labels)
        ],
        dtype=float,
    )

    rdm = np.abs(sample_values[:, None] - sample_values[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _prepare_group_values(ratings_by_group, normalize: bool = True) -> dict[str, float]:
    required_groups = {"odor1", "odor2", "clean"}
    missing_groups = required_groups - set(ratings_by_group.keys())
    if missing_groups:
        raise ValueError(
            "ratings_by_group is missing required keys: "
            f"{sorted(missing_groups)}"
        )

    group_values = {
        group: float(ratings_by_group[group])
        for group in ("odor1", "odor2", "clean")
    }

    if normalize:
        vals = np.array(list(group_values.values()), dtype=float)
        std = vals.std(ddof=0)
        if std > 0:
            vals = (vals - vals.mean()) / std
        else:
            vals = vals - vals.mean()
        group_values = {
            group: float(val)
            for group, val in zip(("odor1", "odor2", "clean"), vals)
        }

    return group_values


def create_cue_type_rdm(sample_labels) -> np.ndarray:
    """
    Cue-type model:
    0 within cue type, 1 across cue types.

    Preconditioned: A*, C*
    Conditioned: B*, D*
    """
    base_conditions = np.array([base_condition(lbl) for lbl in sample_labels], dtype=object)
    cue_type = np.array(
        [
            "preconditioned" if cond.startswith(("A", "C")) else "conditioned"
            for cond in base_conditions
        ],
        dtype=object,
    )
    rdm = (cue_type[:, None] != cue_type[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def create_image_identity_rdm(sample_labels) -> np.ndarray:
    """
    Cue/image identity model, ignoring within-trial presentation position.

    Examples
    --------
    A11_img1 vs A11_img2 = 0
    A11_img1 vs A21_img1 = 1
    """
    labels = np.array([base_condition(lbl) for lbl in sample_labels], dtype=object)
    rdm = (labels[:, None] != labels[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def create_phase_rdm(phase_labels) -> np.ndarray:
    """
    Phase model:
    0 within phase, 1 across premeal/postmeal phases.
    """
    phases = np.array([str(x).strip().lower() for x in phase_labels], dtype=object)
    rdm = (phases[:, None] != phases[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def create_button_presence_rdm(button_labels):
    """
    RDM for whether a button was pressed.

    Groups:
      - 'left'  -> pressed
      - 'right' -> pressed
      - 'none'  -> none

    So:
      left vs right = 0
      left/right vs none = 1
      none vs none = 0
    """
    labels = [str(x).strip().lower() for x in button_labels]
    pressed = np.array([lab in ("left", "right") for lab in labels], dtype=int)

    rdm = (pressed[:, None] != pressed[None, :]).astype(float)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def create_button_side_rdm(button_labels):
    """
    RDM for button side with 'none' treated as maximally different.

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
