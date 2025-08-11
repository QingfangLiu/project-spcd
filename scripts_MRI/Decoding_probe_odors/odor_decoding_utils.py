
# === Additions for cue-based cross-set decoding ===
import os
import pandas as pd
import numpy as np
from nilearn.image import load_img, concat_imgs, get_data, new_img_like

def _subject_run_dir(base_dir: str, sub: str, session_folder: str = "premeal_trialwise_2mm"):
    """Return the subject/session folder that holds CSVs and beta NIfTIs."""
    return os.path.join(base_dir, "GLM", session_folder, sub)

def _csv_path_for_run(sub_dir: str, sub: str, run: int, session: str = "premeal"):
    """CSV name is like sub-01_premeal_run-1_beta_index.csv, in the subject dir."""
    return os.path.join(sub_dir, f"{sub}_{session}_run-{run}_beta_index.csv")

def _cue_type_from_index(cue_index: int) -> str:
    """Odd = A, Even = B."""
    return "A" if (cue_index % 2 == 1) else "B"

def _filter_by_cue_type(df: pd.DataFrame, cue_type: str) -> pd.DataFrame:
    """Keep only rows whose cue_index matches the cue type (A=odd, B=even)."""
    if cue_type not in ("A", "B"):
        raise ValueError("cue_type must be 'A' or 'B'")
    is_A = (df["cue_index"] % 2 == 1)
    return df[is_A] if cue_type == "A" else df[~is_A]

def _labels_from_odor_index(df: pd.DataFrame, n_classes: int):
    """
    Build labels from odor_index.
      - 3-class: 0=clean_air, 1=odor_1, 2=odor_2  (use values as-is)
      - 2-class: keep {1,2}; map 1->0 (odor_1), 2->1 (odor_2)
    Returns: filtered_df, labels (np.array)
    """
    if n_classes == 3:
        keep = df["odor_index"].isin([0, 1, 2])
        df2 = df.loc[keep].copy()
        y = df2["odor_index"].to_numpy()  # 0,1,2
        return df2, y

    elif n_classes == 2:
        keep = df["odor_index"].isin([1, 2])
        df2 = df.loc[keep].copy()
        # odor_1->0, odor_2->1
        df2["y"] = df2["odor_index"].map({1: 0, 2: 1})
        return df2, df2["y"].to_numpy()

    else:
        raise ValueError("n_classes must be 2 or 3")

def _runwise_zscore_across_trials(all_img_paths_for_run: list, img_paths_to_normalize: list):
    """
    Z-score per voxel across ALL trials of the run (A/B/air/etc.).
    Then apply those run-wise mean/std to the selected trials.
    Returns: list of new NIfTI images (same order as img_paths_to_normalize)
    """
    # Load all trials to compute run-level mean/std
    all_data = []
    ref_img = None
    for p in all_img_paths_for_run:
        img = load_img(p)
        if ref_img is None:
            ref_img = img
        all_data.append(get_data(img))

    # Stack: shape (n_trials_all, X, Y, Z)
    all_stack = np.stack(all_data, axis=0)
    mean = all_stack.mean(axis=0)
    std = all_stack.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero

    # Normalize only the selected trials
    norm_imgs = []
    for p in img_paths_to_normalize:
        img = load_img(p)
        data = get_data(img)
        z = (data - mean) / std
        norm_imgs.append(new_img_like(img, z))

    return norm_imgs

def load_betas_cue(sub: str,
                   run: int,
                   base_dir: str,
                   session_folder: str = "premeal_trialwise_2mm",
                   session_tag: str = "premeal",
                   cue_type: str = "A",
                   n_classes: int = 2,
                   normalize: bool = True):
    """
    Load beta images + labels for ONE run, filtered by cue type and class setting.

    - Resolves CSV and beta paths from: <base_dir>/<session_folder>/<sub>/
    - Filters by cue type via cue_index parity (A=odd, B=even).
    - Labels from odor_index; 2-class keeps {1,2} (odor_1 vs odor_2), 3-class keeps {0,1,2}.
    - If normalize=True: run-wise z-score across ALL trials of the run (voxel-wise).

    Returns
    -------
    imgs : list of NIfTI images (selected trials for this run)
    y    : ndarray of labels
    meta : pd.DataFrame with columns ['run','set','cue_index','odor_index','beta_path']
    """
    sub_dir = _subject_run_dir(base_dir, sub, session_folder=session_folder)
    csv_path = _csv_path_for_run(sub_dir, sub, run, session=session_tag)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Resolve beta file paths relative to the subject dir
    df["beta_path"] = df["beta_file"].apply(lambda b: os.path.join(sub_dir, b))

    # 1) Filter by cue type (A=odd, B=even)
    df_ct = _filter_by_cue_type(df, cue_type)

    # 2) Make labels from odor_index (and possibly drop clean_air if 2-class)
    df_lab, y = _labels_from_odor_index(df_ct, n_classes=n_classes)

    # 3) Select beta paths for these rows (preserving order)
    beta_paths_sel = df_lab["beta_path"].tolist()

    # 4) Run-wise normalization across ALL trials in this run (A/B/air/etc.)
    if normalize:
        all_beta_paths_this_run = df["beta_path"].tolist()
        imgs = _runwise_zscore_across_trials(all_beta_paths_this_run, beta_paths_sel)
    else:
        imgs = [load_img(p) for p in beta_paths_sel]

    # Meta for CV split construction
    meta = df_lab.loc[:, ["run", "set", "cue_index", "odor_index"]].copy()
    meta["beta_path"] = beta_paths_sel

    return imgs, y, meta

def load_decoding_data_cue(sub: str,
                           base_dir: str,
                           runs=(1, 2, 3),
                           cue_type: str = "A",
                           n_classes: int = 2,
                           normalize: bool = True,
                           session_folder: str = "premeal_trialwise_2mm",
                           session_tag: str = "premeal"):
    """
    Load/stack all runs for cue-based decoding.

    Returns
    -------
    all_imgs : 4D NIfTI
    all_y    : labels (ndarray)
    run_ids  : ndarray of run id per trial
    set_ids  : ndarray of set id per trial (1 or 2)
    """
    imgs_all, y_all, run_ids, set_ids = [], [], [], []

    for r in runs:
        try:
            imgs, y, meta = load_betas_cue(
                sub, r, base_dir,
                session_folder=session_folder,
                session_tag=session_tag,
                cue_type=cue_type,
                n_classes=n_classes,
                normalize=normalize
            )
        except FileNotFoundError:
            # Gracefully skip missing run
            continue

        imgs_all.extend(imgs)
        y_all.extend(y.tolist())
        run_ids.extend(meta["run"].to_list())
        set_ids.extend(meta["set"].to_list())

    if len(imgs_all) == 0:
        raise RuntimeError(f"No data found for {sub} with runs={runs}, cue_type={cue_type}")

    all_imgs = concat_imgs(imgs_all)
    all_y = np.array(y_all)
    run_ids = np.array(run_ids)
    set_ids = np.array(set_ids)

    return all_imgs, all_y, run_ids, set_ids
