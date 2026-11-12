"""Model RDMs for image-wise RSA."""

from __future__ import annotations

import numpy as np

VALUE_MODELS = ("cue_value", "signed_difference", "absolute_difference")


def validate_analysis_options(image_scope="both", value_model="cue_value"):
    if image_scope not in ("both", "second"):
        raise ValueError("image_scope must be 'both' or 'second'")
    if value_model not in VALUE_MODELS:
        raise ValueError(f"Unknown value_model: {value_model}")
    if image_scope != "second" and value_model != "cue_value":
        raise ValueError("Value-difference models require --image_scope second")


def analysis_subfolder(image_scope="both", value_model="cue_value"):
    """Keep analysis variants in distinct default output directories."""
    validate_analysis_options(image_scope, value_model)
    if image_scope == "both":
        return "probe_imagewise"
    return f"probe_imagewise_second_image/{value_model}"

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

    Ratings are normalized across all phase-by-odor values within subject
    before expanding to samples. This preserves pre/post value changes while
    still removing subject-level rating scale differences.
    """
    if len(sample_labels) != len(phase_labels):
        raise ValueError("sample_labels and phase_labels must have the same length")

    phase_labels = [str(phase) for phase in phase_labels]
    group_values_by_phase = _prepare_phase_values(phase_labels, ratings_by_phase, normalize)
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


def _prepare_phase_values(phase_labels, ratings_by_phase, normalize=True):
    """Normalize outcome ratings once across the requested phases, before trials."""
    phase_order = list(dict.fromkeys(phase_labels))

    missing_phases = set(phase_order) - set(ratings_by_phase.keys())
    if missing_phases:
        raise ValueError(
            "ratings_by_phase is missing required phase keys: "
            f"{sorted(missing_phases)}"
        )

    group_values_by_phase = {
        phase: _prepare_group_values(ratings_by_phase[phase], normalize=False)
        for phase in phase_order
    }

    if normalize:
        groups = ("odor1", "odor2", "clean")
        vals = np.array(
            [
                group_values_by_phase[phase][group]
                for phase in phase_order
                for group in groups
            ],
            dtype=float,
        )
        std = vals.std(ddof=0)
        if std > 0:
            vals = (vals - vals.mean()) / std
        else:
            vals = vals - vals.mean()

        idx = 0
        for phase in phase_order:
            for group in groups:
                group_values_by_phase[phase][group] = float(vals[idx])
                idx += 1

    return group_values_by_phase


def create_second_image_value_rdms(first_labels, second_labels, phase_labels,
                                   ratings_by_phase, normalize=True):
    """Return trial variables and three alternative value-distance RDMs.

    Signed difference is V2 - V1. Absolute difference is abs(V2 - V1).
    Each RDM then uses abs(x_i - x_j) BETWEEN trials. These are separate
    candidate models, not three regressors to enter simultaneously.
    """
    if not (len(first_labels) == len(second_labels) == len(phase_labels)):
        raise ValueError("First/second labels and phases must have equal lengths")
    if not len(first_labels):
        raise ValueError("No paired trials provided")
    phases = [str(p) for p in phase_labels]
    for phase in set(phases):
        if phase not in ratings_by_phase:
            raise ValueError(f"Missing pleasantness ratings for phase: {phase}")
        values = _prepare_group_values(ratings_by_phase[phase], normalize=False)
        if not np.isfinite(list(values.values())).all():
            raise ValueError(f"Nonfinite pleasantness ratings for phase: {phase}")
    ratings = _prepare_phase_values(phases, ratings_by_phase, normalize)
    first = np.array([ratings[p][condition_to_group(base_condition(label))]
                      for label, p in zip(first_labels, phases)])
    second = np.array([ratings[p][condition_to_group(base_condition(label))]
                       for label, p in zip(second_labels, phases)])
    variables = {"first_cue_value": first, "cue_value": second,
                 "signed_difference": second - first,
                 "absolute_difference": np.abs(second - first)}
    rdms = {name: np.abs(variables[name][:, None] - variables[name][None, :])
            for name in VALUE_MODELS}
    return variables, rdms


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
