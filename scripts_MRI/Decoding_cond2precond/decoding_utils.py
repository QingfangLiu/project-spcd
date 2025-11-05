# decoding_utils.py
#
# Utilities for Goal 1: Pre-conditioned → Conditioned cue reactivation
#
# TRAIN:  Day 1 conditioned cues (B1–B6), from character/place localizer
# TEST:   Day 2 pre-meal probe, A cues only (A1–A6)
# LABELS: 6-way B-identity classes (B1..B6 -> 0..5; A mapped via A->B)
#
# This module reuses:
#   - odor_training_utils._zscore_4d_by_voxel   (runwise voxel z-scoring)
#   - probe_decoding_utils._runwise_setwise_zscore
#   - probe_decoding_utils._subject_run_dir, _csv_path_for_run
#
# Folder conventions (base_dir = e.g. /data/liuq13/SPCD_mini):
#   - Day 1 localizer events TSVs:
#       base_dir/nii/sub-XX/func_PAID/{character|place}/
#           sub-XX_task-{character|place}_run-<run>_events.tsv
#   - Day 1 localizer trialwise GLM:
#       base_dir/GLM/{character|place}_trialwise_2mm/sub-XX/
#           beta_run<run>_trial<trial_num:02d>.nii
#   - Day 2 probe trialwise GLM + CSVs:
#       base_dir/GLM/premeal_trialwise_2mm/sub-XX/
#           sub-XX_premeal_run-<run>_beta_index.csv
#
#   - Subject-level design:
#       base_dir/task_info/subject_info.csv
#           columns: ['subject', 'Day 0', 'Day 1', 'Day 2', 'feeding', 'pairing']
#             pairing == 'c' -> B cues are characters
#             pairing == 'p' -> B cues are places

import os
import numpy as np
import pandas as pd
from nilearn.image import load_img, concat_imgs

from odor_training_utils import _zscore_4d_by_voxel
from probe_decoding_utils import (
    _subject_run_dir,
    _csv_path_for_run,
    _runwise_setwise_zscore,
)

# ---------------------------------------------------------------------
# Helper: mapping from pairing code to localizer folder names
# ---------------------------------------------------------------------

def _subject_info_path(base_dir: str) -> str:
    return os.path.join(base_dir, "task_info", "subject_info.csv")


def _get_pairing_for_subject(base_dir: str, sub: str) -> str:
    """Return pairing code ('c' or 'p') for a subject from subject_info.csv."""
    path = _subject_info_path(base_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"subject_info.csv not found: {path}")
    df = pd.read_csv(path)
    row = df.loc[df["subject"] == sub]
    if row.empty:
        raise ValueError(f"Subject {sub} not found in subject_info.csv")
    pairing = row.iloc[0]["pairing"]
    if pairing not in ("c", "p"):
        raise ValueError(f"Unexpected pairing code '{pairing}' for {sub}")
    return pairing


def _localizer_info_for_subject(base_dir: str, sub: str):
    """
    From subject_info.pairing, determine which Day-1 localizer to use.
    Returns:
        stim_type  : 'character' or 'place'
        glm_folder : 'character_trialwise_2mm' or 'place_trialwise_2mm'
    """
    pairing = _get_pairing_for_subject(base_dir, sub)
    if pairing == "c":
        stim_type = "character"
    elif pairing == "p":
        stim_type = "place"
    else:
        raise ValueError(f"Unsupported pairing code '{pairing}' for {sub}")
    glm_folder = f"{stim_type}_trialwise_2mm"
    return stim_type, glm_folder


def _localizer_events_path(base_dir: str, sub: str, stim_type: str, run: int) -> str:
    """
    e.g.
    base_dir/nii/sub-01/func_PAID/character/sub-01_task-character_run-1_events.tsv
    """
    return os.path.join(
        base_dir,
        "nii",
        sub,
        "func_PAID",
        stim_type,
        f"{sub}_task-{stim_type}_run-{run}_events.tsv",
    )


def _localizer_beta_path(base_dir: str, glm_folder: str, sub: str,
                         run: int, trial_num: int) -> str:
    """
    e.g.
    base_dir/GLM/character_trialwise_2mm/sub-01/beta_run1_trial01.nii
    """
    return os.path.join(
        base_dir,
        "GLM",
        glm_folder,
        sub,
        f"beta_run{run}_trial{trial_num:02d}.nii",
    )

# ---------------------------------------------------------------------
# Mapping between cue indices and B-identity classes
# ---------------------------------------------------------------------

# B cues correspond to even-numbered cue indices: 2,4,6,8,10,12
#   2 -> B1_1   (class 0)
#   4 -> B2_1   (class 1)
#   6 -> B3_1   (class 2)
#   8 -> B1_2   (class 3)
#  10 -> B2_2   (class 4)
#  12 -> B3_2   (class 5)
_B_INDEX_TO_CLASS = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5}

def _b_index_from_img_label(trial_type: str):
    """
    Convert 'img_k' trial_type to an integer k. Returns None for malformed strings.
    """
    if not isinstance(trial_type, str):
        return None
    if not trial_type.startswith("img_"):
        return None
    try:
        k = int(trial_type.split("_")[1])
    except Exception:
        return None
    return k


def _b_class_from_b_index(b_idx: int) -> int:
    """Map B cue index (2,4,6,8,10,12) to class 0..5."""
    if b_idx not in _B_INDEX_TO_CLASS:
        raise ValueError(f"Invalid B index {b_idx}; expected one of {sorted(_B_INDEX_TO_CLASS)}")
    return _B_INDEX_TO_CLASS[b_idx]


def _b_class_from_a_cue_index(a_idx: int) -> int:
    """
    Map A cue index (1,3,5,7,9,11) to the corresponding B-identity class.

    A->B mapping:
      1 -> 2
      3 -> 4
      5 -> 6
      7 -> 8
      9 -> 10
     11 -> 12
    """
    b_idx = a_idx + 1
    return _b_class_from_b_index(b_idx)

# ---------------------------------------------------------------------
# Loader 1: Day-1 conditioned B cues (training data)
# ---------------------------------------------------------------------

def load_decoding_data_conditioned_B(
    sub: str,
    base_dir: str,
    runs=(1, 2, 3, 4),
    normalize: bool = True,
):
    """
    Build training set from Day-1 localizer B cues (conditioned).
    For each subject:
      - Use subject_info.pairing to select character vs place localizer.
      - Load events TSV per run.
      - Keep only B cues (even-numbered img_2,img_4,...,img_12).
      - Map each B cue to 6-way class (0..5) via _B_INDEX_TO_CLASS.
      - Load corresponding beta_run<trial> images.

    Parameters
    ----------
    sub : str
        Subject ID, e.g. 'sub-01'.
    base_dir : str
        Top-level directory (same as other decoding scripts).
    runs : iterable of int
        Day-1 localizer runs to include.
    normalize : bool
        If True, runwise voxelwise z-scoring across selected trials (per run).

    Returns
    -------
    all_imgs : nibabel.Nifti1Image
        4D stack of all B-cue betas across runs.
    y : np.ndarray[int]
        Labels (0..5) aligned to trials.
    """
    stim_type, glm_folder = _localizer_info_for_subject(base_dir, sub)

    imgs_all_4d = []
    y_all = []

    for run in runs:
        tsv = _localizer_events_path(base_dir, sub, stim_type, run)
        if not os.path.exists(tsv):
            # allow missing runs (e.g., only 3 runs for some subjects)
            continue

        ev = pd.read_csv(tsv, sep="\t")

        # trial_num = 1..nTrials, following original order in TSV
        ev = ev.reset_index(drop=True)
        ev["trial_num"] = np.arange(1, len(ev) + 1)

        # Parse img index and keep B cues only (even indices that map to B classes)
        ev["img_index"] = ev["trial_type"].apply(_b_index_from_img_label)
        mask_b = ev["img_index"].isin(_B_INDEX_TO_CLASS.keys())
        ev_B = ev.loc[mask_b].copy()

        if ev_B.empty:
            continue

        run_imgs = []
        run_labels = []

        for _, row in ev_B.iterrows():
            b_idx = int(row["img_index"])
            cl = _b_class_from_b_index(b_idx)
            trial_num = int(row["trial_num"])
            beta_path = _localizer_beta_path(base_dir, glm_folder, sub, run, trial_num)
            if not os.path.exists(beta_path):
                raise FileNotFoundError(f"Beta not found: {beta_path}")
            run_imgs.append(load_img(beta_path))
            run_labels.append(cl)

        if len(run_imgs) == 0:
            continue

        run_4d = concat_imgs(run_imgs)
        if normalize:
            run_4d = _zscore_4d_by_voxel(run_4d)

        imgs_all_4d.append(run_4d)
        y_all.extend(run_labels)

    if len(imgs_all_4d) == 0:
        raise RuntimeError(f"No Day-1 B-cue data found for {sub} with runs={runs}")

    all_imgs = concat_imgs(imgs_all_4d)
    y = np.array(y_all, dtype=int)
    return all_imgs, y

# ---------------------------------------------------------------------
# Loader 2: Day-2 pre-conditioned A cues (test data)
# ---------------------------------------------------------------------

def load_decoding_data_preconditioned_A(
    sub: str,
    base_dir: str,
    runs=(1, 2, 3),
    normalize: bool = True,
    session_folder: str = "premeal_trialwise_2mm",
    session_tag: str = "premeal",
):
    """
    Build test set from Day-2 probe A cues (pre-conditioned).

    - Reads trialwise GLM CSVs as in probe_decoding_utils.
    - Keeps only A cues (odd cue_index).
    - For each A cue (cue_index in {1,3,5,7,9,11}):
        label = B-identity class 0..5 (via cue_index -> B index -> class).
    - Normalization: per-run, per-set voxelwise z-scoring using all trials
      in that run/set (via _runwise_setwise_zscore), consistent with your
      existing probe decoding utilities.

    Returns
    -------
    all_imgs : nibabel.Nifti1Image
        4D stack of A-cue betas.
    y : np.ndarray[int]
        6-way B-identity labels for each A trial (0..5).
    """
    imgs_all = []
    y_all = []

    for run in runs:
        sub_dir = _subject_run_dir(base_dir, sub, session_folder=session_folder)
        csv_path = _csv_path_for_run(sub_dir, sub, run, session=session_tag)
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        # add full beta path
        df["beta_path"] = df["beta_file"].apply(lambda b: os.path.join(sub_dir, b))

        # Keep only A cues (pre-conditioned; odd cue_index)
        df_A = df.loc[df["cue_index"] % 2 == 1].copy()
        if df_A.empty:
            continue

        # Build 6-way B-identity labels from A cue indices
        df_A["y"] = df_A["cue_index"].apply(_b_class_from_a_cue_index)

        # Normalization: set-wise within this run using ALL trials in df
        if normalize:
            imgs = _runwise_setwise_zscore(df_run_all=df, df_selected=df_A)
        else:
            imgs = [load_img(p) for p in df_A["beta_path"].tolist()]

        imgs_all.extend(imgs)
        y_all.extend(df_A["y"].to_list())

    if len(imgs_all) == 0:
        raise RuntimeError(f"No Day-2 A-cue data found for {sub} with runs={runs}")

    all_imgs = concat_imgs(imgs_all)
    y = np.array(y_all, dtype=int)
    return all_imgs, y
