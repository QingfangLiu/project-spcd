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
    
    Parameters
    ----------
    sub : str
        Subject ID (e.g., 'sub-01')
    run : int
        Run number (1, 2, or 3)
    base_dir : str
        Base directory containing GLM folder
    session_folder : str
        Folder name (e.g., 'premeal_trialwise_2mm')
    session_tag : str
        Session tag for CSV naming (e.g., 'premeal')
    
    Returns
    -------
    imgs : list of NIfTI images (z-scored)
    metadata : pd.DataFrame with columns ['run', 'set', 'cue_index', 'odor_index', 'beta_path']
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
    
    Returns 12 condition-averaged patterns (one per condition):
    A11, A21, C1, B11, B21, D1, A12, A22, C2, B12, B22, D2
    
    Parameters
    ----------
    sub : str
        Subject ID
    base_dir : str
        Base directory
    session_folder : str
        Session folder name (e.g., 'premeal_trialwise_2mm')
    session_tag : str
        Session tag (e.g., 'premeal')
    runs : tuple
        Run numbers to include
    
    Returns
    -------
    condition_imgs : dict
        Dictionary mapping condition name (e.g., 'A11') to 3D NIfTI image
    condition_order : list
        Ordered list of condition names for indexing
    """
    # Condition mapping based on cue_index and odor_index
    # A = odd cue_index, B = even cue_index
    # Set 1: cue_index 1-6, Set 2: cue_index 7-12
    # Odor: 0=clean, 1=odor1, 2=odor2

    # (confirmed with stimulus generation R code)
    
    condition_map = { # very useful to map "meta" data to condition info
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
                # 4 images per condition (if not missing data)
                # 12 images across 3 runs (images have been normalized within each run)
    
    # Average trials within each condition
    condition_imgs = {}
    for cond_name, trial_imgs in condition_trials.items():
        if len(trial_imgs) == 0:
            raise ValueError(f"No trials found for condition {cond_name}")
        
        # Stack and average
        trial_data = np.stack([get_data(img) for img in trial_imgs], axis=0)
        avg_data = trial_data.mean(axis=0)
        condition_imgs[cond_name] = new_img_like(trial_imgs[0], avg_data)
    
    # Define condition order for consistent indexing
    condition_order = ['A11', 'A21', 'C1', 'B11', 'B21', 'D1',
                       'A12', 'A22', 'C2', 'B12', 'B22', 'D2']
    
    return condition_imgs, condition_order


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


# ============================================================================
# Neural RDM Computation
# ============================================================================

def compute_neural_rdm(patterns, metric='correlation'):
    """
    Compute neural RDM from condition patterns.
    
    Parameters
    ----------
    patterns : np.ndarray (n_conditions, n_voxels)
        Neural patterns for each condition
    metric : str
        'correlation' (default) or 'euclidean'
    
    Returns
    -------
    rdm : np.ndarray (n_conditions, n_conditions)
        Neural similarity matrix (correlation values if metric='correlation')
    """
    n_cond = patterns.shape[0]
    
    if metric == 'correlation': # in [-1, 1]
        # Compute correlation-based similarity (guardrail added)
        rdm = np.corrcoef(patterns) # correlation between all pairs of rows
        rdm = np.nan_to_num(rdm, nan=0.0) # replace NANs with 0
        np.fill_diagonal(rdm, 1.0) # force diagonal to be 1.0
    elif metric == 'euclidean': # in [0, 1]
        # Compute pairwise distances and convert to similarity
        distances = squareform(pdist(patterns, metric='euclidean'))
        # Convert to similarity (invert and normalize)
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

def rsa_multiple_regression(neural_rdm, model_rdms):
    """
    Fit multiple regression: neural_RDM = β1*model1 + β2*model2 + ... + ε
    
    Uses lower triangle (excluding diagonal) vectorized values.
    
    Parameters
    ----------
    neural_rdm : np.ndarray (n, n)
        Neural similarity matrix
    model_rdms : dict
        Dictionary of model RDMs, e.g., {'association': rdm1, 'identity': rdm2}
    
    Returns
    -------
    betas : dict
        Regression coefficients for each model
    """
    n = neural_rdm.shape[0]
    
    # Vectorize lower triangle (excluding diagonal)
    tril_idx = np.tril_indices(n, k=-1) # k=-1: exclude the diagonal
    y = neural_rdm[tril_idx]
    
    # Stack model RDM vectors
    X = np.column_stack([model_rdm[tril_idx] for model_rdm in model_rdms.values()])
    
    # Fit linear regression
    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)
    
    # Get betas
    betas = {name: beta for name, beta in zip(model_rdms.keys(), reg.coef_)}
    
    return betas


# ============================================================================
# RSA single model correlation
# ============================================================================

def rsa_single_model_correlation(neural_rdm, model_rdm, method: str = "spearman"):
    """
    Correlate neural similarity matrix with ONE model similarity matrix
    using lower triangle (excluding diagonal).
    
    method: "spearman" or "kendall" (kendall = tau-b in scipy)
    """
    n = neural_rdm.shape[0]
    tril_idx = np.tril_indices(n, k=-1)

    y = neural_rdm[tril_idx]
    x = model_rdm[tril_idx]

    if method == "spearman":
        r, _ = spearmanr(x, y)
    elif method == "kendall":
        r, _ = kendalltau(x, y)  # scipy's kendalltau is tau-b (tie-corrected)
    else:
        raise ValueError(f"Unknown RSA correlation method: {method}")

    # be safe in searchlight: return 0.0 if nan
    return 0.0 if np.isnan(r) else float(r)


# ============================================================================
# Searchlight RSA
# ============================================================================

# Create a custom scoring function for this model
from sklearn.base import BaseEstimator
# use BaseEstimator to provide get_params() and set_params() methods
        
class SingleModelRSA(BaseEstimator): 
    def __init__(self, model_rdms=None, target_name=None,
                 rsa_method="regression", neural_metric="correlation"):
        self.model_rdms = model_rdms
        self.target_name = target_name
        self.rsa_method = rsa_method
        self.neural_metric = neural_metric

    def fit(self, X, y=None): # y is accepted by ignored
        return self
            
    def score(self, X, y=None): # y is accepted by ignored
        if X.shape[1] < 3:
            return 0.0
        
        neural_rdm = compute_neural_rdm(X, metric=self.neural_metric)

        if self.rsa_method == "regression":
            betas = rsa_multiple_regression(neural_rdm, self.model_rdms)
            return float(betas[self.target_name]) # return a scalar for the scoring function

        elif self.rsa_method in ("spearman", "kendall"):
            model_rdm = self.model_rdms[self.target_name]  # single target model
            return rsa_single_model_correlation(neural_rdm, model_rdm, method=self.rsa_method)

        else:
            raise ValueError(f"Unknown rsa_method: {self.rsa_method}")


def searchlight_rsa_single_subject(sub: str,
                                    base_dir: str,
                                    session_folder: str,
                                    session_tag: str,
                                    mask_file: str,
                                    model_rdms: dict,
                                    radius: float = 8.0,
                                    n_jobs: int = -1,
                                    rsa_method: str = "regression",
                                    neural_metric: str = "correlation"):
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
    
    Returns
    -------
    rsa_maps : dict
        Dictionary mapping model names to 3D rsa maps
    """
    from nilearn.decoding import SearchLight
    from nilearn.image import load_img
    
    # Load condition-averaged data
    print(f"Loading data for {sub}...")
    condition_imgs, condition_order = load_condition_averaged_data(
        sub, base_dir, session_folder, session_tag
    )
    
    # Stack into 4D image
    img_list = [condition_imgs[cond] for cond in condition_order] # a list of 12 3D images
    img_4d = concat_imgs(img_list) # stack them along the 4th dimension (79*95*79*12)

    print("Loading mask file...")    
    mask_img = load_img(mask_file)
    
    # Run searchlight
    print(f"Running searchlight RSA (radius={radius}mm, n_jobs={n_jobs})...")
    
    # We'll run one searchlight per model to get separate maps
    rsa_maps = {}
    
    for i, model_name in enumerate(model_rdms.keys()):
        print(f"  Computing {model_name} map...")

        estimator = SingleModelRSA( # return beta for each model i
            model_rdms=model_rdms,
            target_name=model_name,
            rsa_method=rsa_method,
            neural_metric=neural_metric,
        )

        searchlight = SearchLight(
            mask_img=mask_img,
            radius=radius,
            estimator=estimator,
            cv=[(np.arange(len(condition_order)), np.arange(len(condition_order)))],
            n_jobs=n_jobs,
            verbose=1
        )
        # note here we have a "descriptive RSA" asking how well does the model RDM explain
        # the sphere's RDM estimated from all available data.
        # in other words, there's no CV happening here
        
        # Dummy y (not useful, only because SearchLight API design needs it)
        # clf.fit(X, y)
        # clf.score(X, y)
        y_dummy = np.arange(len(condition_order)) # an array from 0 to 11
        
        searchlight.fit(img_4d, y_dummy) # a 4D image was given to searchlight
        scores_img = new_img_like(mask_img, searchlight.scores_) # wrap scores into nifti
        rsa_maps[model_name] = scores_img
    
    return rsa_maps


def save_rsa_maps(sub: str,
                  rsa_maps: dict,
                  output_dir: str,
                  session_tag: str,
                  radius: float = 8.0,
                  rsa_method: str = "regression",
                  neural_metric: str = "correlation"):
    """
    Save RSA score maps for one subject.
    
    Parameters
    ----------
    sub : str
        Subject ID
    rsa_maps : dict
        Dictionary of rsa maps (NIfTI images)
    output_dir : str
        Output directory
    session_tag : str
        Session tag (e.g., 'premeal')
    radius : float
        Searchlight radius for filename
    rsa_method : str
        "regression", "spearman", or "kendall"
    neural_metric: str
        Method to conduct neural similarity: 
        'correlation' (default) or 'euclidean'
    """
    os.makedirs(output_dir, exist_ok=True)

    # map rsa method to statistic label
    if rsa_method == "regression":
        stat_label = "beta"
    elif rsa_method == "spearman":
        stat_label = "rho"
    elif rsa_method == "kendall":
        stat_label = "tau"
    else:
        raise ValueError(f"Unknown rsa_method: {rsa_method}")

    # use a mapping dictionary
    neural_metric_label = {
        "correlation": "corr",
        "euclidean": "euc"
    }.get(neural_metric, neural_metric)

    for model_name, rsa_map in rsa_maps.items():
        output_file = os.path.join(
            output_dir,
            f"{sub}_session-{session_tag}_{model_name}_{rsa_method}_{stat_label}_ns-{neural_metric_label}_radius-{int(radius)}mm.nii.gz"
        )
        rsa_map.to_filename(output_file)
        print(f"Saved: {output_file}")
