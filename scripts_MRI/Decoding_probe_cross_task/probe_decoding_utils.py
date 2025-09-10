
# === Additions for cue-based cross-set decoding (Revised: set-wise normalization, no leakage) ===
import os
import pandas as pd
import numpy as np
from nilearn.image import load_img, concat_imgs, get_data, new_img_like

def _subject_run_dir(base_dir: str, sub: str, session_folder: str = "premeal_trialwise_2mm"):
    # Return the subject/session folder that holds CSVs and beta NIfTIs.
    return os.path.join(base_dir, "GLM", session_folder, sub)

def _csv_path_for_run(sub_dir: str, sub: str, run: int, session: str = "premeal"):
    # CSV name is like sub-01_premeal_run-1_beta_index.csv, in the subject dir.
    return os.path.join(sub_dir, f"{sub}_{session}_run-{run}_beta_index.csv")

def _cue_type_from_index(cue_index: int) -> str:
    # Odd = A, Even = B.
    return "A" if (cue_index % 2 == 1) else "B"

def _filter_by_cue_type(df: pd.DataFrame, cue_type: str) -> pd.DataFrame:
    # Keep only rows whose cue_index matches the cue type (A=odd, B=even).
    if cue_type not in ("A", "B"):
        raise ValueError("cue_type must be 'A' or 'B'")
    is_A = (df["cue_index"] % 2 == 1)
    return df[is_A] if cue_type == "A" else df[~is_A]

def _labels_from_odor_index(df: pd.DataFrame, n_classes: int):
    # Build labels from odor_index.
    #   - 3-class: 0=clean_air, 1=odor_1, 2=odor_2 (use values as-is)
    #   - 2-class: keep {1,2}; map 1->0 (odor_1), 2->1 (odor_2)
    # Returns: filtered_df, labels (np.array)
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

def _runwise_setwise_zscore(df_run_all: pd.DataFrame, df_selected: pd.DataFrame):
    # Leakage-safe normalization: voxelwise z-score computed WITHIN each set (1 vs 2)
    # using all trials from that set in the run; then applied to the selected rows for that set.
    #
    # Parameters
    # ----------
    # df_run_all : DataFrame
    #   All rows for this run (any cue_type / class), with columns including ['set','beta_path'].
    # df_selected : DataFrame
    #   The subset to be returned (already filtered by cue_type and class). Must include ['set','beta_path'].
    #
    # Returns
    # -------
    # norm_imgs : list of NIfTI images aligned to df_selected.index order.
    norm_imgs = [None] * len(df_selected)
    sel_positions = {idx: pos for pos, idx in enumerate(df_selected.index)}
    for s in sorted(df_selected["set"].unique()):
        df_stats = df_run_all.loc[df_run_all["set"] == s]
        if df_stats.empty:
            continue
        ref_img = None
        stack = []
        for p in df_stats["beta_path"].tolist():
            img = load_img(p)
            if ref_img is None:
                ref_img = img
            stack.append(get_data(img))
        all_stack = np.stack(stack, axis=0)  # (n_trials_s, X, Y, Z)
        mean = all_stack.mean(axis=0)
        std = all_stack.std(axis=0)
        std[std == 0] = 1.0
        df_sel_s = df_selected.loc[df_selected["set"] == s]
        for idx, row in df_sel_s.iterrows():
            img = load_img(row["beta_path"])
            data = get_data(img)
            z = (data - mean) / std
            norm_imgs[sel_positions[idx]] = new_img_like(img, z)
    for i, im in enumerate(norm_imgs):
        if im is None:
            img = load_img(df_selected.iloc[i]["beta_path"])
            norm_imgs[i] = img  # unnormalized fallback
    return norm_imgs

def load_betas_cue(sub: str,
                   run: int,
                   base_dir: str,
                   session_folder: str = "premeal_trialwise_2mm",
                   session_tag: str = "premeal",
                   cue_type: str = "A",
                   n_classes: int = 2,
                   normalize: bool = True):
    # Load beta images + labels for ONE run, filtered by cue type and class setting.
    # - Resolves CSV and beta paths from: <base_dir>/<session_folder>/<sub>/
    # - Filters by cue type via cue_index parity (A=odd, B=even).
    # - Labels from odor_index; 2-class keeps {1,2} (odor_1 vs odor_2), 3-class keeps {0,1,2}.
    # - If normalize=True: **set-wise** run normalization (voxel-wise) to avoid cross-set leakage.
    #
    # Returns
    # -------
    # imgs : list of NIfTI images (selected trials for this run)
    # y    : ndarray of labels
    # meta : pd.DataFrame with columns ['run','set','cue_index','odor_index','beta_path']
    sub_dir = _subject_run_dir(base_dir, sub, session_folder=session_folder)
    csv_path = _csv_path_for_run(sub_dir, sub, run, session=session_tag)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["beta_path"] = df["beta_file"].apply(lambda b: os.path.join(sub_dir, b))
    df_ct = _filter_by_cue_type(df, cue_type)
    df_lab, y = _labels_from_odor_index(df_ct, n_classes=n_classes)
    beta_paths_sel = df_lab["beta_path"].tolist()
    if normalize:
        imgs = _runwise_setwise_zscore(df_run_all=df, df_selected=df_lab)
    else:
        imgs = [load_img(p) for p in beta_paths_sel]
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
    # Load/stack all runs for cue-based decoding.
    #
    # Returns
    # -------
    # all_imgs : 4D NIfTI
    # all_y    : labels (ndarray)
    # run_ids  : ndarray of run id per trial
    # set_ids  : ndarray of set id per trial (1 or 2)
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
