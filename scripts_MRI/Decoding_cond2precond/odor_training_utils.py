
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
        4D image with shape (X, Y, Z, n_trials).
    eps : float
        Small floor to avoid divide-by-zero; any std < eps is set to 1.

    Returns
    -------
    nibabel.Nifti1Image
        4D image of the same shape with voxelwise z-scored data.
    """
    X = get_data(imgs_4d)  # (X, Y, Z, n_trials)
    
    mu = X.mean(axis=-1, keepdims=True)
    sd = X.std(axis=-1, ddof=1, keepdims=True)
    sd[sd < eps] = 1.0

    Z = (X - mu) / sd
    return new_img_like(imgs_4d, Z.astype(np.float32))


def load_decoding_data_odor(sub: str,
                            base_dir: str,
                            runs=(1, 2),
                            n_classes: int = 2,
                            normalize: bool = True):
    """
    Build odor training set with optional per-run voxelwise z-scoring.

    Returns
    -------
    all_imgs : nibabel.Nifti1Image  # trials concatenated across runs
    y : np.ndarray[int]             # labels aligned to trials
    """
    imgs_all_4d = []   # list of 4D stacks, one per run
    y_all = []

    for run in runs:
        tsv = _odor_events_path(base_dir, sub, run)
        if not os.path.exists(tsv):
            raise FileNotFoundError(f"Events TSV not found: {tsv}")
        ev = pd.read_csv(tsv, sep="\t")

        if n_classes == 2:
            ev = ev[ev["trial_type"].isin(["odor_1", "odor_2"])].copy()
            ev["y"] = ev["trial_type"].map({"odor_1": 0, "odor_2": 1})
        elif n_classes == 3:
            ev = ev[ev["trial_type"].isin(["clean_air", "odor_1", "odor_2"])].copy()
            ev["y"] = ev["trial_type"].map({"clean_air": 0, "odor_1": 1, "odor_2": 2})
        else:
            raise ValueError("n_classes must be 2 or 3")

        # load this run's betas in order
        run_imgs = []
        run_labels = []
        for i, row in ev.reset_index(drop=True).iterrows():
            trial_num = i + 1
            beta_path = _odor_beta_path(base_dir, sub, run, trial_num)
            if not os.path.exists(beta_path):
                raise FileNotFoundError(f"Beta not found: {beta_path}")
            run_imgs.append(load_img(beta_path))
            run_labels.append(int(row["y"]))

        if len(run_imgs) == 0:
            continue

        run_4d = concat_imgs(run_imgs)          # (X, Y, Z, n_trials_in_run)
        if normalize:
            run_4d = _zscore_4d_by_voxel(run_4d) # <-- per-run z-scoring

        imgs_all_4d.append(run_4d)
        y_all.extend(run_labels)

    if len(imgs_all_4d) == 0:
        raise RuntimeError(f"No odor data found for {sub} with runs={runs}")

    all_imgs = concat_imgs(imgs_all_4d)         # concat runs after per-run z-score
    y = np.array(y_all, dtype=int)
    return all_imgs, y

