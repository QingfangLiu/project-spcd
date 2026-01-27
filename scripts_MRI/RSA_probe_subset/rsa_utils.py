"""
RSA utilities for probe task analysis
Loads trial-wise betas, averages by condition, computes neural RDMs,
and performs searchlight RSA with multiple model RDMs.
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from nilearn.image import load_img, concat_imgs, get_data, new_img_like


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


# ============================================================================
# Data Loading Functions
# ============================================================================

def _subject_run_dir(base_dir: str, sub: str, session_folder: str):
    """Return the subject/session folder that holds CSVs and beta NIfTIs."""
    return os.path.join(base_dir, "GLM", session_folder, sub)


def _csv_path_for_run(sub_dir: str, sub: str, run: int, session: str):
    """Return path to beta_index CSV for a given run."""
    return os.path.join(sub_dir, f"{sub}_{session}_run-{run}_beta_index.csv")


def load_run_betas_rsa(sub: str,
                       run: int,
                       base_dir: str,
                       session_folder: str,
                       session_tag: str):
    """
    Load all trial-wise betas for one run, with z-score normalization.
    """
    sub_dir = _subject_run_dir(base_dir, sub, session_folder)
    csv_path = _csv_path_for_run(sub_dir, sub, run, session_tag)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load metadata
    df = pd.read_csv(csv_path)
    df["beta_path"] = df["beta_file"].apply(lambda b: os.path.join(sub_dir, b))

    # Load all betas for this run
    beta_paths = df["beta_path"].tolist()
    ref_img = None
    stack = []

    for p in beta_paths:
        img = load_img(p)
        if ref_img is None:
            ref_img = img
        stack.append(get_data(img))

    # Stack and z-score across ALL trials in this run
    all_stack = np.stack(stack, axis=0)  # (n_trials, X, Y, Z)
    mean = all_stack.mean(axis=0)
    std = all_stack.std(axis=0)
    std[std == 0] = 1.0  # Avoid division by zero

    # Normalize each trial
    norm_imgs = []
    for data in all_stack:
        z = (data - mean) / std
        norm_imgs.append(new_img_like(ref_img, z))

    # Metadata
    metadata = df.loc[:, ["run", "set", "cue_index", "odor_index"]].copy()
    metadata["beta_path"] = beta_paths

    return norm_imgs, metadata


def load_condition_averaged_data(sub: str,
                                 base_dir: str,
                                 session_folder: str,
                                 session_tag: str,
                                 runs=(1, 2, 3)):
    """
    Load and average trials by condition across runs.

    Returns 12 condition-averaged patterns (one per condition) in CANONICAL_CONDITION_ORDER.
    """
    # (confirmed with stimulus generation R code)
    condition_map = {
        (1, 1): 'A11',  # Set1, A, odor1
        (3, 2): 'A21',  # Set1, A, odor2
        (5, 0): 'C1',   # Set1, A, clean
        (2, 1): 'B11',  # Set1, B, odor1
        (4, 2): 'B21',  # Set1, B, odor2
        (6, 0): 'D1',   # Set1, B, clean
        (7, 1): 'A12',  # Set2, A, odor1
        (9, 2): 'A22',  # Set2, A, odor2
        (11, 0): 'C2',  # Set2, A, clean
        (8, 1): 'B12',  # Set2, B, odor1
        (10, 2): 'B22', # Set2, B, odor2
        (12, 0): 'D2',  # Set2, B, clean
    }

    # Collect trials by condition
    condition_trials = {cond: [] for cond in condition_map.values()}

    for run in runs:
        try:
            imgs, meta = load_run_betas_rsa(
                sub, run, base_dir, session_folder, session_tag
            )
        except FileNotFoundError:
            print(f"Warning: Run {run} not found for {sub}, skipping")
            continue

        for idx, row in meta.iterrows():
            key = (int(row['cue_index']), int(row['odor_index']))
            if key in condition_map:
                cond_name = condition_map[key]
                condition_trials[cond_name].append(imgs[idx])

    # Average trials within each condition
    condition_imgs = {}
    for cond_name, trial_imgs in condition_trials.items():
        if len(trial_imgs) == 0:
            raise ValueError(f"No trials found for condition {cond_name}")

        trial_data = np.stack([get_data(img) for img in trial_imgs], axis=0)
        avg_data = trial_data.mean(axis=0)
        condition_imgs[cond_name] = new_img_like(trial_imgs[0], avg_data)

    # Use canonical order
    condition_order = list(CANONICAL_CONDITION_ORDER)

    return condition_imgs, condition_order


# ============================================================================
# Model RDM Creation (unchanged; still returns 12x12 on canonical order)
# ============================================================================

def create_identity_model_rdm():
    """
    Create odor identity model RDM (12x12).
    Value = 1 for same odor identity, 0 otherwise.
    """
    conditions = list(CANONICAL_CONDITION_ORDER)

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
                rdm[i, j] = 1
            else:
                for group_conds in odor_groups.values():
                    if cond_i in group_conds and cond_j in group_conds:
                        rdm[i, j] = 1
                        break

    return rdm


def create_value_model_rdm(diagonal: float = 1.0):
    """
    Create odor value model RDM (12x12) where odor1 and odor2 are treated
    as the SAME odor class (merged), and clean is separate.
    """
    conditions = list(CANONICAL_CONDITION_ORDER)

    odor_merged = {'A11', 'B11', 'A12', 'B12', 'A21', 'B21', 'A22', 'B22'}
    clean = {'C1', 'D1', 'C2', 'D2'}

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


# ============================================================================
# Neural RDM Computation
# ============================================================================

def compute_neural_rdm(patterns, metric='correlation'):
    """
    Compute neural RDM from condition patterns.
    """
    n_cond = patterns.shape[0]

    if metric == 'correlation':
        rdm = np.corrcoef(patterns)
        rdm = np.nan_to_num(rdm, nan=0.0)
        np.fill_diagonal(rdm, 1.0)
    elif metric == 'euclidean':
        distances = squareform(pdist(patterns, metric='euclidean'))
        max_dist = distances.max()
        if max_dist > 0:
            rdm = 1 - (distances / max_dist)
        else:
            rdm = np.ones((n_cond, n_cond))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return rdm


# ============================================================================
# RSA Multiple Regression
# ============================================================================

def rsa_multiple_regression(neural_rdm, model_rdms, pair_mask=None):
    """
    Fit multiple regression: neural_RDM = β1*model1 + β2*model2 + ... + ε
    Uses lower triangle (excluding diagonal) vectorized values.
    """
    n = neural_rdm.shape[0]
    tril_idx = np.tril_indices(n, k=-1)
    y = neural_rdm[tril_idx]

    X = np.column_stack([model_rdm[tril_idx] for model_rdm in model_rdms.values()])

    if pair_mask is not None:
        y = y[pair_mask]
        X = X[pair_mask, :]

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)

    betas = {name: beta for name, beta in zip(model_rdms.keys(), reg.coef_)}

    return betas


# ============================================================================
# RSA single model correlation
# ============================================================================

def rsa_single_model_correlation(neural_rdm, model_rdm, method: str = "spearman", pair_mask=None):
    """
    Correlate neural similarity matrix with ONE model similarity matrix
    using lower triangle (excluding diagonal).
    """
    n = neural_rdm.shape[0]
    tril_idx = np.tril_indices(n, k=-1)

    y = neural_rdm[tril_idx]
    x = model_rdm[tril_idx]

    if pair_mask is not None:
        y = y[pair_mask]
        x = x[pair_mask]

    if method == "spearman":
        r, _ = spearmanr(x, y)
    elif method == "kendall":
        r, _ = kendalltau(x, y)
    else:
        raise ValueError(f"Unknown RSA correlation method: {method}")

    return 0.0 if np.isnan(r) else float(r)


# ============================================================================
# Searchlight RSA
# ============================================================================

from sklearn.base import BaseEstimator


class SingleModelRSA(BaseEstimator):
    def __init__(self, model_rdms=None, target_name=None,
                 rsa_method="regression", neural_metric="correlation",
                 pair_mask=None):
        self.model_rdms = model_rdms
        self.target_name = target_name
        self.rsa_method = rsa_method
        self.neural_metric = neural_metric
        self.pair_mask = pair_mask

    def fit(self, X, y=None):
        return self

    def score(self, X, y=None):
        if X.shape[1] < 3:
            return 0.0

        neural_rdm = compute_neural_rdm(X, metric=self.neural_metric)

        if self.rsa_method == "regression":
            betas = rsa_multiple_regression(neural_rdm, self.model_rdms, pair_mask=self.pair_mask)
            return float(betas[self.target_name])

        elif self.rsa_method in ("spearman", "kendall"):
            model_rdm = self.model_rdms[self.target_name]
            return rsa_single_model_correlation(neural_rdm, model_rdm, method=self.rsa_method, pair_mask=self.pair_mask)

        else:
            raise ValueError(f"Unknown rsa_method: {self.rsa_method}")


def build_odor_pair_mask(condition_order, odor_set):
    n = len(condition_order)
    i, j = np.tril_indices(n, k=-1)
    return np.array(
        [(condition_order[ii] in odor_set) and (condition_order[jj] in odor_set)
         for ii, jj in zip(i, j)],
        dtype=bool
    )


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
                                  use_pair_mask: bool = False,
                                  cue_subset: str = "all"):
    """
    Run searchlight RSA for one subject.

    cue_subset:
      - "all" (12 conditions)
      - "conditioned" (B/D only, 6 conditions)
      - "preconditioned" (A/C only, 6 conditions)
    """
    from nilearn.decoding import SearchLight
    from nilearn.image import load_img

    print(f"Loading data for {sub}...")
    condition_imgs, condition_order = load_condition_averaged_data(
        sub, base_dir, session_folder, session_tag
    )

    # --- subset conditions (this is the new part) ---
    subset_order = _validate_and_get_subset_order(cue_subset)

    # Keep subset in the *requested* order, but validate it exists in loaded order
    available = set(condition_order)
    missing = [c for c in subset_order if c not in available]
    if missing:
        raise ValueError(f"Missing conditions in loaded data for cue_subset='{cue_subset}': {missing}")

    condition_order = list(subset_order)
    print(f"cue_subset={cue_subset}: using {len(condition_order)} conditions: {condition_order}")

    # --- slice model RDMs to match subset (new) ---
    sliced_model_rdms = {
        name: _slice_square_by_conditions(rdm, from_order=CANONICAL_CONDITION_ORDER, to_order=condition_order)
        for name, rdm in model_rdms.items()
    }

    pair_mask = None
    if use_pair_mask:
        # Note: if you subset first, pair-mask is usually unnecessary.
        odor_set = {'A11','B11','A12','B12','A21','B21','A22','B22'}
        pair_mask = build_odor_pair_mask(condition_order, odor_set)
        print(f"Pair-mask: keeping {pair_mask.sum()} / {pair_mask.size} lower-triangle pairs (odor-odor only).")

    # Stack into 4D image
    img_list = [condition_imgs[cond] for cond in condition_order]
    img_4d = concat_imgs(img_list)

    print("Loading mask file...")
    mask_img = load_img(mask_file)

    print(f"Running searchlight RSA (radius={radius}mm, n_jobs={n_jobs})...")

    rsa_maps = {}

    for model_name in sliced_model_rdms.keys():
        print(f"  Computing {model_name} map...")

        estimator = SingleModelRSA(
            model_rdms=sliced_model_rdms,   # <-- use sliced
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

        y_dummy = np.arange(n_cond)
        searchlight.fit(img_4d, y_dummy)

        scores_img = new_img_like(mask_img, searchlight.scores_)
        rsa_maps[model_name] = scores_img

    return rsa_maps


def save_rsa_maps(sub: str,
                  rsa_maps: dict,
                  output_dir: str,
                  session_tag: str,
                  radius: float = 8.0,
                  rsa_method: str = "regression",
                  neural_metric: str = "correlation",
                  use_pair_mask: bool = False,
                  cue_subset: str = "all"):
    """
    Save RSA score maps for one subject.
    """
    os.makedirs(output_dir, exist_ok=True)

    if rsa_method == "regression":
        stat_label = "beta"
    elif rsa_method == "spearman":
        stat_label = "rho"
    elif rsa_method == "kendall":
        stat_label = "tau"
    else:
        raise ValueError(f"Unknown rsa_method: {rsa_method}")

    neural_metric_label = {
        "correlation": "corr",
        "euclidean": "euc"
    }.get(neural_metric, neural_metric)

    mask_label = "mask-odor" if use_pair_mask else None
    subset_label = None if cue_subset == "all" else f"subset-{cue_subset}"

    for model_name, rsa_map in rsa_maps.items():

        name_parts = [
            f"{sub}_session-{session_tag}",
            model_name,
            rsa_method,
            stat_label,
            f"ns-{neural_metric_label}",
            f"radius-{int(radius)}mm",
        ]

        if subset_label is not None:
            name_parts.append(subset_label)

        if mask_label is not None:
            name_parts.append(mask_label)

        output_file = os.path.join(
            output_dir,
            "_".join(name_parts) + ".nii.gz"
        )
        rsa_map.to_filename(output_file)
        print(f"Saved: {output_file}")
