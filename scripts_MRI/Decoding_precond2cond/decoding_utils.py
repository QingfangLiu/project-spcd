import os
import numpy as np
import pandas as pd
from nilearn.image import load_img, concat_imgs, get_data, new_img_like

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


def _localizer_info_for_subject(base_dir: str, sub: str):
    """
    From subject_info.pairing, determine which Day-1 localizer to use.
    Returns:
        stim_type  : 'character' or 'place'
        glm_folder : 'character_trialwise_2mm' or 'place_trialwise_2mm'
    """
    pairing = _get_pairing_for_subject(base_dir, sub)
    if pairing == "p":
        stim_type = "character" # find the swapped stim type
    elif pairing == "c":
        stim_type = "place" # find the swapped stim type
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
# Loader: Day-1 pre-conditioned A cues from localizer (TRAIN for Goal 4)
# ---------------------------------------------------------------------

def load_decoding_data_preconditioned_A_localizer(
    sub: str,
    base_dir: str,
    runs=(1, 2, 3, 4),
    normalize: bool = True,
):
    """
    Build training set from Day-1 localizer A cues (pre-conditioned).

    For each subject:
      - Use subject_info.pairing to select character vs place localizer.
      - Load events TSV per run.
      - Keep only A cues (odd-numbered img_1,img_3,...,img_11).
      - Map each A cue to 6-way B-identity class (0..5) via A->B mapping:
            1->2, 3->4, 5->6, 7->8, 9->10, 11->12 -> class
      - Load corresponding beta_run<trial> images from the localizer GLM.

    Parameters
    ----------
    sub : str
        Subject ID, e.g. 'sub-01'.
    base_dir : str
        Base directory (e.g., '/Users/liuq13/SPCD_data').
    runs : tuple
        Localizer runs to include (default: (1,2,3,4)).
    normalize : bool
        If True, apply runwise voxelwise z-scoring within the localizer
        dataset using _zscore_4d_by_voxel.

    Returns
    -------
    imgs : 4D NIfTI image
        Concatenated A-cue betas across runs.
    y : np.ndarray
        1D array of class labels (0..5) per trial.
    """
    stim_type, glm_folder = _localizer_info_for_subject(base_dir, sub)

    imgs_all_4d = []
    y_all = []

    for run in runs:
        tsv = _localizer_events_path(base_dir, sub, stim_type, run)
        if not os.path.exists(tsv):
            # allow missing runs (e.g., fewer than 4 runs for some subjects)
            continue

        ev = pd.read_csv(tsv, sep="\t")

        # trial_num = 1..nTrials, following original order in TSV
        ev = ev.reset_index(drop=True)
        ev["trial_num"] = np.arange(1, len(ev) + 1)

        # Parse img index from 'img_k' and keep A cues only (odd indices)
        ev["img_index"] = ev["trial_type"].apply(_b_index_from_img_label)
        mask_a = ev["img_index"].notnull() & (ev["img_index"] % 2 == 1)
        ev_A = ev.loc[mask_a].copy()

        if ev_A.empty:
            continue

        run_imgs = []
        run_labels = []

        for _, row in ev_A.iterrows():
            a_idx = int(row["img_index"])
            cl = _b_class_from_a_cue_index(a_idx)  # 0..5
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
        raise RuntimeError(f"No Day-1 A-cue data found for {sub} with runs={runs}")

    all_imgs = concat_imgs(imgs_all_4d)
    y = np.array(y_all, dtype=int)
    return all_imgs, y


# ---------------------------------------------------------------------
# Loader: Day-2 odor-prediction B cues (TEST for Goal 4)
# ---------------------------------------------------------------------

def load_decoding_data_conditioned_B_odorpred(
    sub: str,
    base_dir: str,
    runs=(1, 2, 3),
    odorpred_glm_folder: str = "odorpred_cueEARLY_trialwise_2mm",
    normalize: bool = True,
):
    """
    Build test set from Day-2 odor prediction B cues (conditioned).

    - Uses trialwise GLM in:
          base_dir/GLM/<odorpred_glm_folder>/<sub>/
    - For each run, reads:
          <sub>_odorpred_run-<run>_beta_index.csv
    - Keeps rows with trial_type == 'cue' and cue_idx in {2,4,6,8,10,12}.
    - Labels each B cue with 6-way B-identity class (0..5).

    Special-case:
      - For sub-35, run 3 is excluded (partial scan).

    Normalization:
      - Per-run voxelwise z-scoring using _zscore_4d_by_voxel, applied
        within the odor-prediction dataset itself (no leakage from other tasks).
    """
    # Exclude run 3 for sub-35 (partial scan)
    if sub == "sub-35":
        runs = tuple(r for r in runs if r != 3)

    sub_dir = os.path.join(base_dir, "GLM", odorpred_glm_folder, sub)
    if not os.path.isdir(sub_dir):
        raise FileNotFoundError(f"Subject GLM folder not found: {sub_dir}")

    imgs_all_4d = []
    y_all = []

    for run in runs:
        csv_path = os.path.join(sub_dir, f"{sub}_odorpred_run-{run}_beta_index.csv")
        if not os.path.exists(csv_path):
            # allow missing runs (but warn)
            print(f"⚠️ CSV not found for {sub}, run {run}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Build full beta paths
        df["beta_path"] = df["beta_file"].apply(
            lambda fn: os.path.join(sub_dir, fn)
        )

        # Keep B cues: trial_type == 'cue' and even-index cue_idx
        mask_b = (df["trial_type"] == "cue") & df["cue_idx"].isin(
            _B_INDEX_TO_CLASS.keys()
        )
        df_B = df.loc[mask_b].copy()

        if df_B.empty:
            continue

        run_imgs = []
        run_labels = []

        for _, row in df_B.iterrows():
            b_idx = int(row["cue_idx"])
            cl = _b_class_from_b_index(b_idx)
            beta_path = row["beta_path"]
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
        raise RuntimeError(
            f"No Day-2 odor-prediction B-cue data found for {sub} with runs={runs}"
        )

    all_imgs = concat_imgs(imgs_all_4d)
    y = np.array(y_all, dtype=int)
    return all_imgs, y

