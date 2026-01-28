
"""
Searchlight RSA orchestration.

This module runs whole-brain searchlight RSA for a single subject
using nilearn's SearchLight framework. It combines data loading,
optional pair masking, RSA estimators, and searchlight execution.

Subset pipeline; slices model RDMs from canonical order to subset order.
"""

import numpy as np
from nilearn.image import concat_imgs, new_img_like

from .io import load_condition_averaged_data
from .masking import build_odor_pair_mask
from .estimator import SingleModelRSA

# ============================================================================
# Canonical condition order (single source of truth)
# ============================================================================
CANONICAL_CONDITION_ORDER = [
    "A11", "A21", "C1", "B11", "B21", "D1",
    "A12", "A22", "C2", "B12", "B22", "D2",
]

CUE_SUBSET_TO_CONDITIONS = {
    "all": CANONICAL_CONDITION_ORDER,
    # conditioned cues: involving B or D
    "conditioned": ["B11", "B21", "D1", "B12", "B22", "D2"],
    # preconditioned cues: involving A or C
    "preconditioned": ["A11", "A21", "C1", "A12", "A22", "C2"],
}


def _validate_and_get_subset_order(cue_subset: str) -> list:
    if cue_subset not in CUE_SUBSET_TO_CONDITIONS:
        raise ValueError(
            f"Unknown cue_subset: {cue_subset}. "
            f"Choose from {list(CUE_SUBSET_TO_CONDITIONS.keys())}."
        )
    return CUE_SUBSET_TO_CONDITIONS[cue_subset]


def _slice_square_by_conditions(square: np.ndarray, from_order: list, to_order: list) -> np.ndarray:
    """
    Slice a square matrix (e.g., model RDM) defined on `from_order`
    down to a smaller square on `to_order`, preserving `to_order` ordering.
    """
    if square.shape[0] != square.shape[1]:
        raise ValueError("Expected a square matrix to slice.")
    if square.shape[0] != len(from_order):
        raise ValueError(
            f"Square size {square.shape} does not match from_order length {len(from_order)}."
        )

    idx_map = {c: i for i, c in enumerate(from_order)}
    try:
        idx = [idx_map[c] for c in to_order]
    except KeyError as e:
        raise KeyError(f"Condition {e} not found in from_order.") from e

    return square[np.ix_(idx, idx)]


def searchlight_rsa_single_subject(sub: str,
                                  base_dir: str,
                                  session_folder: str,
                                  session_tag: str,
                                  mask_file: str,
                                  model_rdms: dict,
                                  radius: float = 8.0,
                                  n_jobs: int = -1,
                                  rsa_method: str = "regression",
                                  neural_metric: str = "correlation",
                                  cue_subset: str = "all",
                                  use_pair_mask: bool = False):
    """
    Run searchlight RSA for one subject.

    Parameters
    ----------
    sub : str
        Subject ID (e.g., 'sub-01')
    base_dir : str
        Base directory containing GLM folder
    session_folder : str
        Session folder name (e.g., 'premeal_trialwise_2mm')
    session_tag : str
        Session tag (e.g., 'premeal')
    mask_file : str
        Path to gray matter mask
    model_rdms : dict
        Dictionary of model RDMs (can handle one or more)
    radius : float
        Searchlight radius in mm (default: 8.0)
    n_jobs : int
        Number of parallel jobs (default: -1 for all cores)
    rsa_method: str
        Method to construct rsa (regression, spearman, or kendall)
    neural_metric: str
        Method to conduct neural similarity: 'correlation' (default) or 'euclidean'
    cue_subset:
      - "all" (12 conditions)
      - "conditioned" (B/D only, 6 conditions)
      - "preconditioned" (A/C only, 6 conditions)

    Returns
    -------
    rsa_maps : dict
        Dictionary mapping model names to 3D rsa maps
    """

    from nilearn.decoding import SearchLight
    from nilearn.image import load_img

    print(f"Loading data for {sub}...")
    condition_imgs, condition_order = load_condition_averaged_data(
        sub, base_dir, session_folder, session_tag
    )

    # --- subset conditions ---
    subset_order = _validate_and_get_subset_order(cue_subset)

    # Keep subset in the *requested* order, but validate it exists in loaded order
    available = set(condition_order)
    missing = [c for c in subset_order if c not in available]
    if missing:
        raise ValueError(f"Missing conditions in loaded data for cue_subset='{cue_subset}': {missing}")

    condition_order = list(subset_order)
    print(f"cue_subset={cue_subset}: using {len(condition_order)} conditions: {condition_order}")

    # --- slice model RDMs to match subset ---
    sliced_model_rdms = {
        name: _slice_square_by_conditions(rdm, from_order=CANONICAL_CONDITION_ORDER, to_order=condition_order)
        for name, rdm in model_rdms.items()
    }

    pair_mask = None
    if use_pair_mask:
        # create a boolean mask only involving the two odors
        odor_set = {'A11','B11','A12','B12','A21','B21','A22','B22'}
        pair_mask = build_odor_pair_mask(condition_order, odor_set)
        print(f"Pair-mask: keeping {pair_mask.sum()} / {pair_mask.size} lower-triangle pairs (odor-odor only).")

    # Stack into 4D image
    img_list = [condition_imgs[cond] for cond in condition_order] # a list of 3D images
    img_4d = concat_imgs(img_list) # stack them along the 4th dimension

    print("Loading mask file...")
    mask_img = load_img(mask_file)

    print(f"Running searchlight RSA (radius={radius}mm, n_jobs={n_jobs})...")

    rsa_maps = {}

    for model_name in sliced_model_rdms.keys():
        print(f"  Computing {model_name} map...")

        estimator = SingleModelRSA(
            model_rdms=sliced_model_rdms,   # <-- use sliced (what if I use "all"?)
            target_name=model_name,
            rsa_method=rsa_method,
            neural_metric=neural_metric,
            pair_mask=pair_mask,
        )

        n_cond = len(condition_order)
        searchlight = SearchLight(
            mask_img=mask_img,
            radius=radius,
            estimator=estimator,
            cv=[(np.arange(n_cond), np.arange(n_cond))],
            n_jobs=n_jobs,
            verbose=1
        )

        # note here we have a "descriptive RSA" asking how well does the model RDM explain
        # the sphere's RDM estimated from all available data.
        # in other words, there's no CV happening here
        
        # Dummy y (not useful, only because SearchLight API design needs it)
        # clf.fit(X, y)
        # clf.score(X, y)

        y_dummy = np.arange(n_cond)
        searchlight.fit(img_4d, y_dummy)

        scores_img = new_img_like(mask_img, searchlight.scores_)
        rsa_maps[model_name] = scores_img

    return rsa_maps