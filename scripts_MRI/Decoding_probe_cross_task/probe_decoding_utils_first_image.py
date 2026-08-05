"""Probe-beta loading for binary first-image cross-task decoding."""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from nilearn.image import concat_imgs, load_img

from probe_decoding_utils import (
    _csv_path_for_run,
    _runwise_setwise_zscore,
    _subject_run_dir,
)


CUE_SUBSET_INDICES = {
    "conditioned_odor_only": {2, 4, 8, 10},
    "preconditioned_odor_only": {1, 3, 7, 9},
}
CUE_SUBSETS = tuple(CUE_SUBSET_INDICES)
SUB14 = "sub-14"


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


def _select_cue_subset_rows(
    df: pd.DataFrame,
    cue_subset: str,
    csv_path: str,
) -> pd.DataFrame:
    """Retain the established conditioned or preconditioned odor cues."""
    if cue_subset not in CUE_SUBSET_INDICES:
        raise ValueError(
            f"cue_subset must be one of {sorted(CUE_SUBSET_INDICES)}"
        )

    cue_index = pd.to_numeric(df["cue_index"], errors="coerce")
    if cue_index.isna().any():
        raise ValueError(f"Probe beta index has invalid cue_index values: {csv_path}")

    selected = df.loc[cue_index.isin(CUE_SUBSET_INDICES[cue_subset])].copy()
    if selected.empty:
        raise ValueError(f"No {cue_subset} rows found: {csv_path}")
    selected["cue_index"] = cue_index.loc[selected.index].astype(int)
    selected["cue_subset"] = cue_subset
    return selected


def _apply_sub14_original_labels(
    df: pd.DataFrame,
    sub: str,
    base_dir: str,
    session_tag: str,
    original_events_dir: Optional[str],
) -> pd.DataFrame:
    """Use the ROI analysis convention for sub-14 physical-odor labels."""
    labeled = df.copy()
    if sub != SUB14:
        labeled["label_source"] = "beta_index_csv"
        return labeled

    required = {"run", "trial", "img_index"}
    missing = required - set(labeled.columns)
    if missing:
        raise ValueError(
            "sub-14 beta index cannot be matched to original events; missing "
            f"columns {sorted(missing)}"
        )

    if original_events_dir is None:
        original_events_dir = os.path.join(
            base_dir,
            "nii",
            sub,
            "func_PAID",
            session_tag,
        )

    event_rows = []
    for run in sorted(labeled["run"].astype(int).unique()):
        event_path = os.path.join(
            original_events_dir,
            f"{sub}_task-{session_tag}_run-{run}_events_orig.tsv",
        )
        if not os.path.exists(event_path):
            raise FileNotFoundError(
                f"Original sub-14 {session_tag} events not found: {event_path}"
            )
        events = pd.read_csv(event_path, sep="\t")
        event_required = {"trial_type", "run", "trial", "img_index", "odor_index"}
        event_missing = event_required - set(events.columns)
        if event_missing:
            raise ValueError(
                f"Original sub-14 TSV is missing columns {sorted(event_missing)}: "
                f"{event_path}"
            )
        events = events.loc[
            events["trial_type"].isin(["cue_img1", "cue_img2"]),
            ["run", "trial", "img_index", "odor_index"],
        ].copy()
        event_rows.append(events)

    originals = pd.concat(event_rows, ignore_index=True)
    key = ["run", "trial", "img_index"]
    if originals.duplicated(key).any():
        raise ValueError(
            "Original sub-14 events contain duplicate (run, trial, img_index) keys"
        )
    originals = originals.rename(columns={"odor_index": "_original_odor_index"})

    sample_keys = labeled.loc[:, key].copy()
    sample_keys["_sample_order"] = np.arange(len(sample_keys))
    matched = sample_keys.merge(
        originals,
        on=key,
        how="left",
        validate="many_to_one",
    ).sort_values("_sample_order")
    if matched["_original_odor_index"].isna().any():
        missing_rows = matched.loc[matched["_original_odor_index"].isna(), key]
        raise ValueError(
            "Could not match sub-14 probe rows to original events:\n"
            + missing_rows.to_string(index=False)
        )

    labeled["odor_index"] = matched["_original_odor_index"].astype(int).to_numpy()
    labeled["label_source"] = "original_premeal_events"
    return labeled


def _binary_odor_labels(
    df: pd.DataFrame,
    sub: str,
    run: int,
    cue_subset: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Map odor_index 1/2 to binary labels 0/1 and reject clean air."""
    odor_index = pd.to_numeric(df["odor_index"], errors="coerce")
    if odor_index.isna().any() or not odor_index.isin([1, 2]).all():
        bad = sorted(set(odor_index.dropna().tolist()) - {1, 2})
        raise ValueError(
            f"Non-odor labels remain for {sub} run {run}, "
            f"cue_subset={cue_subset}: {bad}"
        )

    labeled = df.copy()
    labeled["odor_index"] = odor_index.astype(int)
    labeled["y"] = labeled["odor_index"].map({1: 0, 2: 1})
    return labeled, labeled["y"].to_numpy(dtype=int)


def load_betas_cue_first_image(
    sub: str,
    run: int,
    base_dir: str,
    cue_subset: str,
    session_folder: str = "premeal_trialwise_2mm",
    session_tag: str = "premeal",
    normalize: bool = True,
    sub14_original_events_dir: Optional[str] = None,
):
    """Load one binary probe run for one first-image odor cue subset."""
    sub_dir = _subject_run_dir(base_dir, sub, session_folder=session_folder)
    csv_path = _csv_path_for_run(sub_dir, sub, run, session=session_tag)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {
        "beta_file",
        "run",
        "set",
        "cue_index",
        "odor_index",
        "img_index",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Probe beta index is missing columns {sorted(missing)}: {csv_path}"
        )

    df["beta_path"] = df["beta_file"].apply(
        lambda name: os.path.join(sub_dir, name)
    )
    df_first = _select_first_image_rows(df, csv_path)
    df_subset = _select_cue_subset_rows(df_first, cue_subset, csv_path)
    df_subset = _apply_sub14_original_labels(
        df_subset,
        sub=sub,
        base_dir=base_dir,
        session_tag=session_tag,
        original_events_dir=sub14_original_events_dir,
    )
    df_selected, y = _binary_odor_labels(df_subset, sub, run, cue_subset)

    beta_paths = df_selected["beta_path"].tolist()
    if normalize:
        imgs = _runwise_setwise_zscore(
            df_run_all=df_selected,
            df_selected=df_selected,
        )
    else:
        imgs = [load_img(path) for path in beta_paths]

    meta_columns = [
        "run",
        "set",
        "cue_index",
        "odor_index",
        "img_index",
        "cue_subset",
        "label_source",
    ]
    for optional in ("trial", "trial_type"):
        if optional in df_selected.columns:
            meta_columns.append(optional)
    meta = df_selected.loc[:, meta_columns].copy()
    meta["beta_path"] = beta_paths
    return imgs, y, meta


def load_decoding_data_cue_first_image(
    sub: str,
    base_dir: str,
    cue_subset: str,
    runs=(1, 2, 3),
    normalize: bool = True,
    session_folder: str = "premeal_trialwise_2mm",
    session_tag: str = "premeal",
    sub14_original_events_dir: Optional[str] = None,
):
    """Load and concatenate one binary first-image cue subset across runs."""
    imgs_all, y_all, run_ids, set_ids = [], [], [], []
    for run in runs:
        sub_dir = _subject_run_dir(base_dir, sub, session_folder=session_folder)
        csv_path = _csv_path_for_run(sub_dir, sub, run, session=session_tag)
        if not os.path.exists(csv_path):
            continue
        imgs, y, meta = load_betas_cue_first_image(
            sub,
            run,
            base_dir,
            cue_subset=cue_subset,
            session_folder=session_folder,
            session_tag=session_tag,
            normalize=normalize,
            sub14_original_events_dir=sub14_original_events_dir,
        )
        imgs_all.extend(imgs)
        y_all.extend(y.tolist())
        run_ids.extend(meta["run"].tolist())
        set_ids.extend(meta["set"].tolist())

    if not imgs_all:
        raise RuntimeError(
            f"No first-image data found for {sub}, cue_subset={cue_subset}, "
            f"runs={runs}"
        )

    all_y = np.asarray(y_all, dtype=int)
    if set(all_y) != {0, 1}:
        raise ValueError(
            f"Probe data do not contain both odor classes for {sub}, "
            f"cue_subset={cue_subset}"
        )
    return (
        concat_imgs(imgs_all),
        all_y,
        np.asarray(run_ids),
        np.asarray(set_ids),
    )
