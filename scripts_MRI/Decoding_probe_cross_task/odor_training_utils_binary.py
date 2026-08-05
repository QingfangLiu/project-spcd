"""Binary passive-odor training data with beta numbering preserved."""

import os

import numpy as np
import pandas as pd
from nilearn.image import concat_imgs, load_img

from odor_training_utils import (
    _odor_beta_path,
    _odor_events_path,
    _zscore_4d_by_voxel,
)


def _binary_odor_event_rows(
    events: pd.DataFrame,
    event_path: str,
) -> pd.DataFrame:
    """Select odor trials after assigning beta numbers in the full event order."""
    if "trial_type" not in events.columns:
        raise ValueError(f"Odor events TSV has no trial_type column: {event_path}")

    numbered = events.copy()
    numbered["beta_trial_number"] = np.arange(1, len(numbered) + 1)
    selected = numbered.loc[
        numbered["trial_type"].isin(["odor_1", "odor_2"])
    ].copy()
    if selected.empty:
        raise ValueError(f"No odor_1/odor_2 events found: {event_path}")
    selected["y"] = selected["trial_type"].map({"odor_1": 0, "odor_2": 1})
    return selected


def load_decoding_data_odor_binary(
    sub: str,
    base_dir: str,
    runs=(1, 2),
    normalize: bool = True,
):
    """Load binary passive-odor betas with correct full-design beta indices."""
    run_images = []
    labels = []

    for run in runs:
        event_path = _odor_events_path(base_dir, sub, run)
        if not os.path.exists(event_path):
            raise FileNotFoundError(f"Events TSV not found: {event_path}")
        events = pd.read_csv(event_path, sep="\t")
        selected = _binary_odor_event_rows(events, event_path)

        images = []
        run_labels = []
        for _, row in selected.iterrows():
            beta_trial = int(row["beta_trial_number"])
            beta_path = _odor_beta_path(base_dir, sub, run, beta_trial)
            if not os.path.exists(beta_path):
                gz_path = beta_path + ".gz"
                if os.path.exists(gz_path):
                    beta_path = gz_path
                else:
                    raise FileNotFoundError(f"Beta not found: {beta_path}[.gz]")
            images.append(load_img(beta_path))
            run_labels.append(int(row["y"]))

        images_4d = concat_imgs(images)
        if normalize:
            images_4d = _zscore_4d_by_voxel(images_4d)
        run_images.append(images_4d)
        labels.extend(run_labels)

    if not run_images:
        raise RuntimeError(f"No binary odor data found for {sub}, runs={runs}")
    y = np.asarray(labels, dtype=int)
    if set(y) != {0, 1}:
        raise ValueError(f"Passive odor data do not contain both classes: {sub}")
    return concat_imgs(run_images), y
