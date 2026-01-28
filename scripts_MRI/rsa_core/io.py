
"""
Data loading utilities for RSA analyses.

This module handles disk I/O for trial-wise beta images and
metadata, including loading, normalization, and aggregation of
neural data into condition-averaged patterns.

It encapsulates all assumptions about directory structure,
file naming, and session/run organization.
"""

import os
import numpy as np
import pandas as pd
from nilearn.image import load_img, get_data, new_img_like


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


