
"""
This function loads from image-wise beta estimates
and then averages to per run per cue

"""

import os
import numpy as np
import pandas as pd
from nilearn.image import load_img, get_fdata, new_img_like
from collections import defaultdict

condition_map = { # map "meta" data to condition info
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


def _load_run_betas(
    sub: str,
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
    sub_dir = os.path.join(base_dir, "GLM", session_folder, sub)
    csv_path = os.path.join(sub_dir, f"{sub}_{session_tag}_run-{run}_beta_index.csv")
    
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
        img = load_img(p) # returns in-memory niimg-like object
        if ref_img is None:
            ref_img = img
        stack.append(get_fdata(img)) # convert data to float point values
    
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


def build_run_condition_patterns(
		sub: str,
		base_dir: str,
		session_folder: str,
		session_tag: str,
	):
								
    """
    Parameters
    ----------
    sub : str
        Subject ID (e.g., 'sub-01')
    base_dir : str
        Base directory containing GLM folder
    session_folder : str
        Folder name (e.g., 'premeal_trialwise_2mm')
    session_tag : str
        Session tag for CSV naming (e.g., 'premeal')

    Returns:
    ----------
      patterns: np.array (n_samples, n_vox)
      run_ids: np.array (n_samples,)
      cond_ids: np.array (n_samples,)  # integer indices w.r.t condition_order
      condition_order: list of names (len 12)
    """
    
    # canonical condition order
    condition_order = ['A11', 'A21', 'C1', 'B11', 'B21', 'D1',
                       'A12', 'A22', 'C2', 'B12', 'B22', 'D2']
    cond_index = {name: idx for idx, name in enumerate(condition_order)}

    # gather per (run,condition) trial images
    run_cond_imgs = defaultdict(list)  # key (run, cond_name) -> list of 3D arrays
    # defaultdict creates empty list that allows "append" without error

    runs = [1, 2, 3] # how to allow missing for some subjects?

    for run in runs:
        imgs, meta = _load_run_betas(sub, run, base_dir, session_folder, session_tag)
        for idx, row in meta.iterrows():
            key = (int(row['cue_index']), int(row['odor_index']))
            if key in condition_map:
                cond_name = condition_map[key]
                arr3 = get_data(imgs[idx])   # 3D array
                run_cond_imgs[(run, cond_name)].append(arr3)

    # build per-(run,cond) means and flatten
    patterns_list = []
    run_ids = []
    cond_ids = []
    for run in runs:
        for cond_name in condition_order:
            imgs_for_cell = run_cond_imgs.get((run, cond_name), [])
            if len(imgs_for_cell) == 0:
                # if a (run,cond) cell is missing, skip
                print(f"Warning: missing run {run} cond {cond_name} => skipping this cell")
                continue
            arr = np.stack(imgs_for_cell, axis=0)  # (n_trials_in_cell, X,Y,Z)
            mean_arr = arr.mean(axis=0)
            flattened = mean_arr.ravel() # transform a multi-dimensional array into 1D flattened array (using default C-order, correct?)
            patterns_list.append(flattened)
            run_ids.append(run)
            cond_ids.append(cond_index[cond_name])

    patterns = np.vstack(patterns_list)   # (n_cells, n_vox)
    run_ids = np.array(run_ids, dtype=int)
    cond_ids = np.array(cond_ids, dtype=int)

    return patterns, run_ids, cond_ids, condition_order


    