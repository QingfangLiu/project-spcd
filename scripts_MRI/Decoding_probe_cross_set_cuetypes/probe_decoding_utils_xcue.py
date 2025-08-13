
# probe_decoding_utils_xcue.py
# Utilities for cross-cue + cross-set decoding (Approach 2), with leakage-safe
# normalization computed WITHIN-RUN and WITHIN (set, cue_type) cells.

import os
import pandas as pd
import numpy as np
from nilearn.image import load_img, concat_imgs, get_data, new_img_like

def _subject_run_dir(base_dir: str, sub: str, session_folder: str = "premeal_trialwise_2mm"):
    return os.path.join(base_dir, "GLM", session_folder, sub)

def _csv_path_for_run(sub_dir: str, sub: str, run: int, session: str = "premeal"):
    return os.path.join(sub_dir, f"{sub}_{session}_run-{run}_beta_index.csv")

def _labels_from_odor_index(df: pd.DataFrame, n_classes: int):
    if n_classes == 3:
        keep = df["odor_index"].isin([0, 1, 2])
        df2 = df.loc[keep].copy()
        y = df2["odor_index"].to_numpy()  # 0,1,2
        return df2, y
    elif n_classes == 2:
        keep = df["odor_index"].isin([1, 2])
        df2 = df.loc[keep].copy()
        df2["y"] = df2["odor_index"].map({1: 0, 2: 1})
        return df2, df2["y"].to_numpy()
    else:
        raise ValueError("n_classes must be 2 or 3")

def _cue_type_series_from_index(series: pd.Series):
    return series.apply(lambda k: "A" if (int(k) % 2 == 1) else "B")

def _runwise_setcue_zscore(df_run_all: pd.DataFrame, df_selected: pd.DataFrame):
    """
    Leakage-safe normalization computed WITHIN each RUN and each (set, cue_type) cell.

    For each (set s, cue_type c):
      - compute voxelwise mean/std from ALL trials in this run with set==s and cue_type==c
      - apply to the selected rows of the same (s, c)
    Any selected rows whose (s, c) cell has no stats will be returned unnormalized (fallback).
    """
    if not {"set", "cue_type", "beta_path"}.issubset(df_run_all.columns):
        raise ValueError("df_run_all must include columns: set, cue_type, beta_path")
    if not {"set", "cue_type", "beta_path"}.issubset(df_selected.columns):
        raise ValueError("df_selected must include columns: set, cue_type, beta_path")

    norm_imgs = [None] * len(df_selected)
    sel_positions = {idx: pos for pos, idx in enumerate(df_selected.index)}

    # iterate over all (set, cue) present in the selection
    for s in sorted(df_selected["set"].unique()):
        for c in sorted(df_selected["cue_type"].unique()):
            df_stats = df_run_all.loc[(df_run_all["set"] == s) & (df_run_all["cue_type"] == c)]
            if df_stats.empty:
                continue

            # stack images to compute stats
            ref_img = None
            stack = []
            for p in df_stats["beta_path"].tolist():
                img = load_img(p)
                if ref_img is None:
                    ref_img = img
                stack.append(get_data(img))
            all_stack = np.stack(stack, axis=0)
            mean = all_stack.mean(axis=0)
            std = all_stack.std(axis=0)
            std[std == 0] = 1.0

            df_sel_sc = df_selected.loc[(df_selected["set"] == s) & (df_selected["cue_type"] == c)]
            for idx, row in df_sel_sc.iterrows():
                img = load_img(row["beta_path"])
                z = (get_data(img) - mean) / std
                norm_imgs[sel_positions[idx]] = new_img_like(img, z)

    # Fallback for any rows not normalized due to missing stats
    for i, im in enumerate(norm_imgs):
        if im is None:
            norm_imgs[i] = load_img(df_selected.iloc[i]["beta_path"])
    return norm_imgs

def load_betas_all_cues(sub: str,
                        run: int,
                        base_dir: str,
                        session_folder: str = "premeal_trialwise_2mm",
                        session_tag: str = "premeal",
                        n_classes: int = 2,
                        normalize: bool = True):
    """Load one run keeping BOTH cue types (A/B)."""
    sub_dir = _subject_run_dir(base_dir, sub, session_folder=session_folder)
    csv_path = _csv_path_for_run(sub_dir, sub, run, session=session_tag)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["beta_path"] = df["beta_file"].apply(lambda b: os.path.join(sub_dir, b))
    # Ensure cue_type exists in BOTH the full-run df and the selected df
    df["cue_type"] = _cue_type_series_from_index(df["cue_index"])

    # labels from odor_index
    df_lab, y = _labels_from_odor_index(df, n_classes=n_classes)

    # add cue_type to selection (already present from df)
    df_lab = df_lab.copy()
    # df_lab['cue_type'] already inherited

    beta_paths_sel = df_lab["beta_path"].tolist()

    if normalize:
        imgs = _runwise_setcue_zscore(df_run_all=df, df_selected=df_lab)
    else:
        imgs = [load_img(p) for p in beta_paths_sel]

    meta = df_lab.loc[:, ["run", "set", "cue_index", "odor_index", "cue_type"]].copy()
    meta["beta_path"] = beta_paths_sel
    return imgs, y, meta

def load_decoding_data_both_cues(sub: str,
                                 base_dir: str,
                                 runs=(1, 2, 3),
                                 n_classes: int = 2,
                                 normalize: bool = True,
                                 session_folder: str = "premeal_trialwise_2mm",
                                 session_tag: str = "premeal"):
    """Stack all runs while keeping BOTH cue types."""
    imgs_all, y_all, run_ids, set_ids, cue_types = [], [], [], [], []
    for r in runs:
        try:
            imgs, y, meta = load_betas_all_cues(
                sub, r, base_dir,
                session_folder=session_folder,
                session_tag=session_tag,
                n_classes=n_classes,
                normalize=normalize
            )
        except FileNotFoundError:
            continue

        imgs_all.extend(imgs)
        y_all.extend(y.tolist())
        run_ids.extend(meta["run"].to_list())
        set_ids.extend(meta["set"].to_list())
        cue_types.extend(meta["cue_type"].to_list())

    if len(imgs_all) == 0:
        raise RuntimeError(f"No data found for {sub} with runs={runs}")

    all_imgs = concat_imgs(imgs_all)
    return all_imgs, np.array(y_all), np.array(run_ids), np.array(set_ids), np.array(cue_types)
