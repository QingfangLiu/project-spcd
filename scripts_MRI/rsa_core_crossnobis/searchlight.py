# searchlight.py

"""
Searchlight RSA orchestration (cue-subset capable).

UPDATED for cuewise GLM:
- loads per-run condition maps
- builds neural RDM using cross-run correlations
- supports input stat_type: beta vs tmap
"""

import numpy as np
from nilearn.image import concat_imgs, new_img_like

from .io import load_cuewise_maps_per_run
from .masking import build_odor_pair_mask
from .estimator import SingleModelRSA

# Canonical order (single source of truth)
CANONICAL_CONDITION_ORDER = [
    "A11", "A21", "C1", "B11", "B21", "D1",
    "A12", "A22", "C2", "B12", "B22", "D2",
]

CUE_SUBSET_TO_CONDITIONS = {
    "all": CANONICAL_CONDITION_ORDER,
    "conditioned": ["B11", "B21", "D1", "B12", "B22", "D2"],
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
    if square.shape[0] != square.shape[1]:
        raise ValueError("Expected a square matrix to slice.")
    if square.shape[0] != len(from_order):
        raise ValueError(
            f"Square size {square.shape} does not match from_order length {len(from_order)}."
        )

    idx_map = {c: i for i, c in enumerate(from_order)}
    idx = [idx_map[c] for c in to_order]
    return square[np.ix_(idx, idx)]


def searchlight_rsa_single_subject(
    sub: str,
    base_dir: str,
    session_folder: str,
    session_tag: str,
    mask_file: str,
    model_rdms: dict,
    radius: float = 8.0,
    n_jobs: int = -1,
    rsa_method: str = "regression",
    neural_metric: str = "corr",
    cue_subset: str = "all",       # "all", "conditioned", "preconditioned"
    use_pair_mask: bool = False,
    mode: str = "both",            # "both", "img1only", "img2only"
    stat_type: str = "beta",       # "beta" or "tmap"
):
    """
    Run searchlight RSA for one subject using cuewise GLM maps.

    - Samples are (run, condition) maps.
    - Neural RDM is computed using cross-run similarities 
    """
    
    from nilearn.decoding import SearchLight
    from nilearn.image import load_img

    print(f"Loading cuewise maps for {sub} (mode={mode}, stat_type={stat_type})...")
    
    run_condition_imgs, full_condition_order = load_cuewise_maps_per_run(
        sub=sub,
        base_dir=base_dir,
        session_folder=session_folder,
        session_tag=session_tag,
        mode=mode,
        runs=runs,
        stat_type=stat_type,
    )

    # subset conditions
    condition_order = _validate_and_get_subset_order(cue_subset)
    print(f"cue_subset={cue_subset}: using {len(condition_order)} conditions: {condition_order}")

    # slice model RDMs to subset
    sliced_model_rdms = {
        name: _slice_square_by_conditions(rdm, from_order=CANONICAL_CONDITION_ORDER, to_order=condition_order)
        for name, rdm in model_rdms.items()
    }

    # pair mask on condition space
    pair_mask = None
    if use_pair_mask:
        odor_set = {'A11','B11','A12','B12','A21','B21','A22','B22'}
        pair_mask = build_odor_pair_mask(condition_order, odor_set)
        print(f"Pair-mask: keeping {pair_mask.sum()} / {pair_mask.size} lower-triangle pairs (odor-odor only).")

    # build stacked samples: for each run, for each condition
    cond_to_idx = {c: i for i, c in enumerate(condition_order)}
    img_list = []
    run_ids = []
    cond_ids = []

    runs = (1, 2, 3)

    used_runs = set()
    for run in runs:
        any_in_run = False
        for cond in condition_order:
            key = (run, cond)
            if key not in run_condition_imgs:
                continue
            img_list.append(run_condition_imgs[key])
            run_ids.append(run)
            cond_ids.append(cond_to_idx[cond])
            any_in_run = True
        if any_in_run:
            used_runs.add(run)

    if len(used_runs) < 2:
        raise ValueError(f"Cross-run RSA requires ≥2 runs, but only found: {used_runs}")

    if len(img_list) == 0:
        raise ValueError(f"No maps found for {sub} (mode={mode}, stat_type={stat_type}, subset={cue_subset}).")

    img_4d = concat_imgs(img_list)

    print("Loading mask file...")
    mask_img = load_img(mask_file)

    print(f"Running searchlight RSA (radius={radius}mm, n_jobs={n_jobs})...")
    rsa_maps = {}

    n_samples = len(img_list)
    cv = [(np.arange(n_samples), np.arange(n_samples))]  # descriptive RSA (no CV)

    for model_name in sliced_model_rdms.keys():
        print(f"  Computing {model_name} map...")

        estimator = SingleModelRSA(
            model_rdms=sliced_model_rdms,
            target_name=model_name,
            rsa_method=rsa_method,
            neural_metric=neural_metric,
            pair_mask=pair_mask,
            run_ids=np.asarray(run_ids),
            cond_ids=np.asarray(cond_ids),
            n_conditions=len(condition_order),
        )

        searchlight = SearchLight(
            mask_img=mask_img,
            radius=radius,
            estimator=estimator,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )

        y_dummy = np.arange(n_samples)
        searchlight.fit(img_4d, y_dummy)
        # img_4d is a 4D nifti with shape (x, y, z, n_samples)
        # here X is created: n_samples * n_voxels_in_sphere
        # then X is passed into estimator's fit and score functions

        scores_img = new_img_like(mask_img, searchlight.scores_)
        rsa_maps[model_name] = scores_img

    return rsa_maps
