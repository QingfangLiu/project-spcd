# io_imagewise.py

import os
from typing import Sequence

import numpy as np
import pandas as pd
from nilearn.image import load_img

# Canonical 12-condition order
BASE_CONDITION_ORDER = [
    "A11", "A21", "C1", "B11", "B21", "D1",
    "A12", "A22", "C2", "B12", "B22", "D2",
]

# Map cue_index/odor_index to base condition
condition_map = {
    (1, 1): "A11",
    (3, 2): "A21",
    (5, 0): "C1",
    (2, 1): "B11",
    (4, 2): "B21",
    (6, 0): "D1",
    (7, 1): "A12",
    (9, 2): "A22",
    (11, 0): "C2",
    (8, 1): "B12",
    (10, 2): "B22",
    (12, 0): "D2",
}

BETA_INDEX_TEMPLATE = "{sub}_{session_tag}_run-{run}_beta_index.csv"
EVENTS_TEMPLATE = "{sub}_task-{session_tag}_run-{run}_events.tsv"

def _load_run_betas(
    sub: str,
    run: int,
    base_dir: str,
    session_folder: str,
    session_tag: str,
):
    """
    Load one run's image-wise beta maps.

    Returns
    -------
    imgs : list of NIfTI images
    df : pd.DataFrame
        Rows from the beta index CSV, with beta_path added.
    """
    sub_dir = os.path.join(base_dir, "GLM", session_folder, sub)
    csv_path = os.path.join(
        sub_dir,
        BETA_INDEX_TEMPLATE.format(sub=sub, session_tag=session_tag, run=run),
    )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    if "beta_file" not in df.columns:
        raise ValueError(f"Missing required column 'beta_file' in {csv_path}")

    df["beta_path"] = df["beta_file"].apply(lambda b: os.path.join(sub_dir, b))

    imgs = [load_img(p) for p in df["beta_path"].tolist()]

    return imgs, df


def _load_trial_button_map(base_dir: str, sub: str, session_tag: str, run: int):
    """
    Read events.tsv and build a mapping:
        trial -> response_hand

    Missing responses are kept as 'none'.
    """
    events_dir = os.path.join(base_dir, "nii", sub, "func_PAID", session_tag)
    events_path = os.path.join(
        events_dir,
        EVENTS_TEMPLATE.format(sub=sub, session_tag=session_tag, run=run),
    )

    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events TSV not found: {events_path}")

    ev = pd.read_csv(events_path, sep="\t").copy()

    required = {"trial", "trial_type", "response_hand"}
    missing = required - set(ev.columns)
    if missing:
        raise ValueError(f"Events TSV missing columns {sorted(missing)}: {events_path}")

    # One button row per trial
    button_rows = ev.loc[ev["trial_type"] == "button", ["trial", "response_hand"]].copy()

    trial_to_button = {}
    for _, row in button_rows.iterrows():
        trial = int(row["trial"])
        resp = row["response_hand"]
        resp = str(resp).strip().lower()
        trial_to_button[trial] = resp # some trials may not have a resp column, will later be assigned a 'none'

    return trial_to_button


def build_imagewise_patterns(
    sub: str,
    base_dir: str,
    session_folder: str,
    session_tag: str,
    button_scope: str = "second_image_only",
    cue_subset: str = "all",
):
    """
    Build image-wise RSA inputs.

    Parameters
    ----------
    button_scope : {"both_images", "second_image_only"}
        - both_images: attach trial button label to both image presentations
        - second_image_only: attach button label only to img_index == 2,
          and assign 'none' to img_index == 1

    Returns
    -------
    run_ids : np.ndarray, shape (n_samples,)
    button_labels : np.ndarray, shape (n_samples,)
        Categorical labels like 'left', 'right', 'none'
    sample_labels : np.ndarray, shape (n_samples,)
        One label per actual sample, e.g. 'A11_img1'
    imgs : list
        NIfTI images in the same order as patterns rows
    """
    if button_scope not in ("both_images", "second_image_only"):
        raise ValueError("button_scope must be 'both_images' or 'second_image_only'")

    if cue_subset not in ("all", "conditioned_only", "preconditioned_only"):
        raise ValueError("cue_subset must be one of: all, conditioned_only, preconditioned_only")

    # all 24 possible labels (len: 24)
    sample_labels_order = [f"{c}_img1" for c in BASE_CONDITION_ORDER] + [
        f"{c}_img2" for c in BASE_CONDITION_ORDER
    ] 
    # mapping from label to 0,...,23
    sample_index = {name: idx for idx, name in enumerate(sample_labels_order)}

    img_list = []
    run_ids = []
    button_labels = []
    sample_labels = []

    runs = [1, 2, 3]

    for run in runs:
        try:
            imgs, meta = _load_run_betas(
                sub=sub,
                run=run,
                base_dir=base_dir,
                session_folder=session_folder,
                session_tag=session_tag,
            )

        except Exception as e:
            print(f"[SKIP] sub={sub} run={run} (beta/CSV issue): {e}")
            continue

        required = {"trial", "img_index", "cue_index", "odor_index"}
        missing = required - set(meta.columns)
        
        if missing:
            print(f"[SKIP] sub={sub} run={run} (missing columns): {missing}")
            continue

        try:
            trial_to_button = _load_trial_button_map(
                base_dir=base_dir,
                sub=sub,
                session_tag=session_tag,
                run=run,
            )

        except Exception as e:
            print(f"[SKIP] sub={sub} run={run} (events issue): {e}")
            continue

        for idx, row in meta.iterrows():
            cue_key = (int(row["cue_index"]), int(row["odor_index"]))
            if cue_key not in condition_map:
                continue

            base_cond = condition_map[cue_key]

            if cue_subset == "conditioned_only" and not base_cond.startswith(("B", "D")):
                continue
            if cue_subset == "preconditioned_only" and not base_cond.startswith(("A", "C")):
                continue
            img_index = int(row["img_index"])
            if img_index not in (1, 2):
                raise ValueError(
                    f"Unexpected img_index={img_index} in run {run}, row {idx}. "
                    "Expected 1 or 2."
                )

            sample_label = f"{base_cond}_img{img_index}"
            if sample_label not in sample_index:
                raise ValueError(f"Unknown sample label: {sample_label}")

            trial = int(row["trial"])
            trial_button = trial_to_button.get(trial, "none")
            if pd.isna(trial_button) or str(trial_button).strip() == "":
                trial_button = "none"
            trial_button = str(trial_button).strip().lower()

            if button_scope == "second_image_only" and img_index == 1:
                button_label = "none"
            else:
                button_label = trial_button

            img_list.append(imgs[idx])
            run_ids.append(run)
            button_labels.append(button_label)
            sample_labels.append(sample_label)

    if len(img_list) == 0:
        raise ValueError("No image-wise samples were found.")

    run_ids = np.asarray(run_ids, dtype=int)
    button_labels = np.asarray(button_labels, dtype=object)
    sample_labels = np.asarray(sample_labels, dtype=object)

    return run_ids, button_labels, sample_labels, img_list


def build_imagewise_patterns_for_sessions(
    sub: str,
    base_dir: str,
    session_tags: Sequence[str],
    button_scope: str = "second_image_only",
    cue_subset: str = "all",
):
    """
    Build image-wise RSA inputs for one or more probe sessions.

    When multiple sessions are requested, run IDs are offset by session so
    premeal run 1 and postmeal run 1 are treated as different runs.
    """
    if isinstance(session_tags, str):
        session_tags = (session_tags,)

    valid_sessions = {"premeal", "postmeal"}
    invalid_sessions = sorted(set(session_tags) - valid_sessions)
    if invalid_sessions:
        raise ValueError(f"Unknown session tag(s): {invalid_sessions}")

    all_run_ids = []
    all_button_labels = []
    all_sample_labels = []
    all_phase_labels = []
    all_img_list = []

    for phase_idx, session_tag in enumerate(session_tags):
        session_folder = f"{session_tag}_trialwise_2mm"
        run_ids, button_labels, sample_labels, img_list = build_imagewise_patterns(
            sub=sub,
            base_dir=base_dir,
            session_folder=session_folder,
            session_tag=session_tag,
            button_scope=button_scope,
            cue_subset=cue_subset,
        )

        run_offset = phase_idx * 100
        all_run_ids.append(run_ids + run_offset)
        all_button_labels.append(button_labels)
        all_sample_labels.append(sample_labels)
        all_phase_labels.append(
            np.full(sample_labels.shape, session_tag, dtype=object)
        )
        all_img_list.extend(img_list)

    if not all_img_list:
        raise ValueError("No image-wise samples were found.")

    return (
        np.concatenate(all_run_ids),
        np.concatenate(all_button_labels),
        np.concatenate(all_sample_labels),
        np.concatenate(all_phase_labels),
        all_img_list,
    )
