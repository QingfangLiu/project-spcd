"""Probe-beta loading for first-image-only cross-task decoding."""

import os

import numpy as np
import pandas as pd
from nilearn.image import concat_imgs, load_img

from probe_decoding_utils import (
    _csv_path_for_run,
    _filter_by_cue_type,
    _labels_from_odor_index,
    _runwise_setwise_zscore,
    _subject_run_dir,
)


def _select_first_image_rows(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """Validate image metadata and retain only the first image from each trial."""
    if "img_index" not in df.columns:
        raise ValueError(f"Probe beta index is missing img_index: {csv_path}")

    img_index = pd.to_numeric(df["img_index"], errors="coerce")
    if img_index.isna().any():
        raise ValueError(f"Probe beta index has invalid img_index values: {csv_path}")

    invalid = sorted(set(img_index.loc[~img_index.isin([1, 2])].tolist()))
    if invalid:
        raise ValueError(
            f"Probe beta index has unexpected img_index values {invalid}: {csv_path}"
        )

    first = df.loc[img_index == 1].copy()
    if first.empty:
        raise ValueError(f"Probe beta index has no first-image rows: {csv_path}")
    first["img_index"] = 1

    if "trial_type" in first.columns:
        unexpected = sorted(set(first["trial_type"].astype(str)) - {"cue_img1"})
        if unexpected:
            raise ValueError(
                "First-image rows have unexpected trial_type values "
                f"{unexpected}: {csv_path}"
            )

    return first


def load_betas_cue_first_image(
    sub: str,
    run: int,
    base_dir: str,
    session_folder: str = "premeal_trialwise_2mm",
    session_tag: str = "premeal",
    cue_type: str = "A",
    n_classes: int = 2,
    normalize: bool = True,
):
    """Load one run of probe betas after restricting all operations to img_index=1."""
    sub_dir = _subject_run_dir(base_dir, sub, session_folder=session_folder)
    csv_path = _csv_path_for_run(sub_dir, sub, run, session=session_tag)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"beta_file", "run", "set", "cue_index", "odor_index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Probe beta index is missing columns {sorted(missing)}: {csv_path}"
        )

    df["beta_path"] = df["beta_file"].apply(lambda name: os.path.join(sub_dir, name))
    df_first = _select_first_image_rows(df, csv_path)
    df_cue = _filter_by_cue_type(df_first, cue_type)
    df_selected, y = _labels_from_odor_index(df_cue, n_classes=n_classes)
    if df_selected.empty:
        raise ValueError(
            f"No first-image samples remain for {sub} run {run}, "
            f"cue_type={cue_type}, n_classes={n_classes}"
        )

    beta_paths = df_selected["beta_path"].tolist()
    if normalize:
        # The normalization pool is first-image-only, including nonselected cues/classes.
        imgs = _runwise_setwise_zscore(
            df_run_all=df_first,
            df_selected=df_selected,
        )
    else:
        imgs = [load_img(path) for path in beta_paths]

    meta_columns = ["run", "set", "cue_index", "odor_index", "img_index"]
    for optional in ("trial", "trial_type"):
        if optional in df_selected.columns:
            meta_columns.append(optional)
    meta = df_selected.loc[:, meta_columns].copy()
    meta["beta_path"] = beta_paths
    return imgs, y, meta


def load_decoding_data_cue_first_image(
    sub: str,
    base_dir: str,
    runs=(1, 2, 3),
    cue_type: str = "A",
    n_classes: int = 2,
    normalize: bool = True,
    session_folder: str = "premeal_trialwise_2mm",
    session_tag: str = "premeal",
):
    """Load and concatenate first-image-only probe betas across runs."""
    imgs_all, y_all, run_ids, set_ids = [], [], [], []
    for run in runs:
        try:
            imgs, y, meta = load_betas_cue_first_image(
                sub,
                run,
                base_dir,
                session_folder=session_folder,
                session_tag=session_tag,
                cue_type=cue_type,
                n_classes=n_classes,
                normalize=normalize,
            )
        except FileNotFoundError:
            continue
        imgs_all.extend(imgs)
        y_all.extend(y.tolist())
        run_ids.extend(meta["run"].tolist())
        set_ids.extend(meta["set"].tolist())

    if not imgs_all:
        raise RuntimeError(
            f"No first-image data found for {sub} with runs={runs}, "
            f"cue_type={cue_type}"
        )

    return (
        concat_imgs(imgs_all),
        np.asarray(y_all),
        np.asarray(run_ids),
        np.asarray(set_ids),
    )
