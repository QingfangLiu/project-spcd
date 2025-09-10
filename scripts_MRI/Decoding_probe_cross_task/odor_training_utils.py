
# odor_training_utils.py (Revised for within-dataset normalization)
# ---------------------------------------------------------------
# This module builds the Day-1 odor training set for cross-task decoding.
#   * load_decoding_data_odor(..., normalize=True) now performs voxelwise
#     z-scoring *within the odor dataset itself* (across its own trials).

import os
import numpy as np
import pandas as pd
from nilearn.image import load_img, concat_imgs, get_data, new_img_like


def _odor_events_path(base_dir: str, sub: str, run: int):
    """Return <base_dir>/nii/<sub>/func_PAID/odor_scan/<sub>_task-odor_run-<run>_events.tsv"""
    return os.path.join(
        base_dir, "nii", sub, "func_PAID", "odor_scan",
        f"{sub}_task-odor_run-{run}_events.tsv"
    )


def _odor_beta_path(base_dir: str, sub: str, run: int, trial_num: int):
    """Return <base_dir>/GLM/odor_scan_trialwise_2mm/<sub>/beta_run<run>_trial<trial_num:02d>.nii"""
    return os.path.join(
        base_dir, "GLM", "odor_scan_trialwise_2mm", sub,
        f"beta_run{run}_trial{trial_num:02d}.nii"
    )


def _zscore_4d_by_voxel(imgs_4d, eps: float = 1e-8):
    """Z-score each voxel across trials *within this dataset*.

    Parameters
    ----------
    imgs_4d : nibabel.Nifti1Image
        4D image with shape (n_trials, X, Y, Z).
    eps : float
        Small floor to avoid divide-by-zero; any std < eps is set to 1.

    Returns
    -------
    nibabel.Nifti1Image
        4D image of the same shape with voxelwise z-scored data.
    """
    X = get_data(imgs_4d)  # (n_trials, X, Y, Z)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)  # ddof=1 for sample std
    sd[sd < eps] = 1.0
    Z = (X - mu) / sd
    return new_img_like(imgs_4d, Z.astype(np.float32))


def load_decoding_data_odor(sub: str,
                            base_dir: str,
                            runs=(1, 2),
                            n_classes: int = 2,
                            normalize: bool = True):
    """Build odor training set (images + labels) from Day-1 odor scan.

    Parameters
    ----------
    sub : str
        Subject id (e.g., 'sub-01').
    base_dir : str
        Root directory of the project.
    runs : tuple[int, ...]
        Runs to include.
    n_classes : int
        2 => keep {'odor_1','odor_2'}; 3 => keep {'clean_air','odor_1','odor_2'}.
    normalize : bool
        If True (default), perform voxelwise z-scoring *within the odor dataset*.

    Returns
    -------
    all_imgs : nibabel.Nifti1Image
        4D image stack of trialwise betas.
    y : np.ndarray
        Labels aligned to the 4D image's first dimension.
    """
    imgs_all, y_all = [], []
    for run in runs:
        tsv = _odor_events_path(base_dir, sub, run)
        if not os.path.exists(tsv):
            raise FileNotFoundError(f"Events TSV not found: {tsv}")
        ev = pd.read_csv(tsv, sep='\t')

        if n_classes == 2:
            ev = ev[ev['trial_type'].isin(['odor_1', 'odor_2'])].copy()
            ev['y'] = ev['trial_type'].map({'odor_1': 0, 'odor_2': 1})
        elif n_classes == 3:
            ev = ev[ev['trial_type'].isin(['clean_air', 'odor_1', 'odor_2'])].copy()
            ev['y'] = ev['trial_type'].map({'clean_air': 0, 'odor_1': 1, 'odor_2': 2})
        else:
            raise ValueError('n_classes must be 2 or 3')

        # One beta per row, sequentially numbered within the run
        for i, row in ev.reset_index(drop=True).iterrows():
            trial_num = i + 1
            beta_path = _odor_beta_path(base_dir, sub, run, trial_num)
            if not os.path.exists(beta_path):
                raise FileNotFoundError(f"Beta not found: {beta_path}")
            imgs_all.append(load_img(beta_path))
            y_all.append(int(row['y']))

    if len(imgs_all) == 0:
        raise RuntimeError(f"No odor data found for {sub} with runs={runs}")

    all_imgs = concat_imgs(imgs_all)
    y = np.array(y_all, dtype=int)

    # Within-dataset normalization (recommended for cross-task decoding)
    if normalize:
        all_imgs = _zscore_4d_by_voxel(all_imgs)

    return all_imgs, y
