# io_imagewise.py

import os

import numpy as np
import pandas as pd
from nilearn.image import load_img, new_img_like

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
    Load one run's image-wise beta maps and z-score them within the run.

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

    beta_paths = df["beta_path"].tolist()
    ref_img = None
    stack = []

    for p in beta_paths:
        img = load_img(p)
        if ref_img is None:
            ref_img = img
        stack.append(img.get_fdata())

    all_stack = np.stack(stack, axis=0)  # (n_samples, X, Y, Z)
    mean = all_stack.mean(axis=0)
    std = all_stack.std(axis=0)
    std[std == 0] = 1.0

    norm_imgs = []
    for data in all_stack:
        z = (data - mean) / std
        norm_imgs.append(new_img_like(ref_img, z))

    return norm_imgs, df


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
        if pd.isna(resp) or str(resp).strip() == "" or str(resp).strip().lower() == "n/a":
            resp = "none"
        else:
            resp = str(resp).strip().lower()
        trial_to_button[trial] = resp

    return trial_to_button


def build_imagewise_patterns(
    sub: str,
    base_dir: str,
    session_folder: str,
    session_tag: str,
    button_scope: str = "both_images",
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
    patterns : np.ndarray, shape (n_samples, n_vox)
    run_ids : np.ndarray, shape (n_samples,)
    cond_ids : np.ndarray, shape (n_samples,)
        Integer indices into canonical sample label order.
    button_labels : np.ndarray, shape (n_samples,)
        Categorical labels like 'left', 'right', 'none'
    sample_labels : np.ndarray, shape (n_samples,)
        One label per actual sample, e.g. 'A11_img1'
    imgs : list
        NIfTI images in the same order as patterns rows
    """
    if button_scope not in ("both_images", "second_image_only"):
        raise ValueError("button_scope must be 'both_images' or 'second_image_only'")

    sample_labels_order = [f"{c}_img1" for c in BASE_CONDITION_ORDER] + [
        f"{c}_img2" for c in BASE_CONDITION_ORDER
    ]
    sample_index = {name: idx for idx, name in enumerate(sample_labels_order)}

    patterns_list = []
    img_list = []
    run_ids = []
    cond_ids = []
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

            arr3 = imgs[idx].get_fdata()
            patterns_list.append(arr3.ravel())
            img_list.append(imgs[idx])
            run_ids.append(run)
            cond_ids.append(sample_index[sample_label])
            button_labels.append(button_label)
            sample_labels.append(sample_label)

    if len(patterns_list) == 0:
        raise ValueError("No image-wise samples were found.")

    patterns = np.vstack(patterns_list)
    run_ids = np.asarray(run_ids, dtype=int)
    cond_ids = np.asarray(cond_ids, dtype=int)
    button_labels = np.asarray(button_labels, dtype=object)
    sample_labels = np.asarray(sample_labels, dtype=object)

    return patterns, run_ids, cond_ids, button_labels, sample_labels, img_list
    