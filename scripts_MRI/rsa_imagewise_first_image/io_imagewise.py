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


def build_imagewise_patterns(
    sub: str,
    base_dir: str,
    session_folder: str,
    session_tag: str,
    cue_subset: str = "all",
):
    """
    Build first-image-only RSA inputs.

    This variant keeps only img_index == 1 from each trial and does not load or
    model button responses.

    Returns
    -------
    run_ids : np.ndarray, shape (n_samples,)
    sample_labels : np.ndarray, shape (n_samples,)
        One label per actual sample, e.g. 'A11_img1'
    imgs : list
        NIfTI images in the same order as sample_labels.
    """
    if cue_subset not in ("all", "conditioned_only", "preconditioned_only"):
        raise ValueError("cue_subset must be one of: all, conditioned_only, preconditioned_only")

    sample_labels_order = [f"{c}_img1" for c in BASE_CONDITION_ORDER]
    sample_index = {name: idx for idx, name in enumerate(sample_labels_order)}

    img_list = []
    run_ids = []
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

        required = {"img_index", "cue_index", "odor_index"}
        missing = required - set(meta.columns)

        if missing:
            print(f"[SKIP] sub={sub} run={run} (missing columns): {missing}")
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
            if img_index != 1:
                continue

            sample_label = f"{base_cond}_img1"
            if sample_label not in sample_index:
                raise ValueError(f"Unknown sample label: {sample_label}")

            img_list.append(imgs[idx])
            run_ids.append(run)
            sample_labels.append(sample_label)

    if len(img_list) == 0:
        raise ValueError("No first-image samples were found.")

    run_ids = np.asarray(run_ids, dtype=int)
    sample_labels = np.asarray(sample_labels, dtype=object)

    return run_ids, sample_labels, img_list


def build_imagewise_patterns_for_sessions(
    sub: str,
    base_dir: str,
    session_tags: Sequence[str],
    cue_subset: str = "all",
):
    """
    Build first-image-only RSA inputs for one or more probe sessions.

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
    all_sample_labels = []
    all_phase_labels = []
    all_img_list = []

    for phase_idx, session_tag in enumerate(session_tags):
        session_folder = f"{session_tag}_trialwise_2mm"
        run_ids, sample_labels, img_list = build_imagewise_patterns(
            sub=sub,
            base_dir=base_dir,
            session_folder=session_folder,
            session_tag=session_tag,
            cue_subset=cue_subset,
        )

        run_offset = phase_idx * 100
        all_run_ids.append(run_ids + run_offset)
        all_sample_labels.append(sample_labels)
        all_phase_labels.append(
            np.full(sample_labels.shape, session_tag, dtype=object)
        )
        all_img_list.extend(img_list)

    if not all_img_list:
        raise ValueError("No first-image samples were found.")

    return (
        np.concatenate(all_run_ids),
        np.concatenate(all_sample_labels),
        np.concatenate(all_phase_labels),
        all_img_list,
    )
