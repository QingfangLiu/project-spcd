"""Pleasantness-rating helpers for image-wise RSA."""

from __future__ import annotations

import os
import pandas as pd


def _default_subject_summary_path(sub: str) -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(project_root, "beh_processed_subs", sub, "pleasantness_summary.csv")


def load_subject_pleasantness_ratings(
    sub: str,
    session_tag: str,
    ratings_file: str | None = None,
):
    """
    Load subject-specific pleasantness ratings for a probe session.

    By default this reads the processed behavior file:
      beh_processed_subs/<sub>/pleasantness_summary.csv

    Expected default format:
      - one row per odor category
      - columns: odor_num, pre_mean, post_mean, change

    Coding follows the processed behavioral analysis:
      - odor_num 0 = clean
      - odor_num 1 = non-devalued odor
      - odor_num 2 = devalued odor

    In the standardized probe/odor coding used elsewhere in this project:
      - odor1 = non-devalued odor
      - odor2 = devalued odor
      - clean = clean air

    If ``ratings_file`` is provided, it may instead point to a wide CSV with
    columns:
      - sub
      - pre_odor1, pre_odor2, pre_clean
      - post_odor1, post_odor2, post_clean

    Parameters
    ----------
    sub : str
        Subject identifier, e.g. 'sub-01'.
    session_tag : {'premeal', 'postmeal'}
        Probe session used to choose pre- vs post-meal ratings.
    ratings_file : str or None
        Optional explicit path to the ratings CSV. If omitted, uses a
        placeholder path inside the project.

    Returns
    -------
    dict or None
        {'odor1': float, 'odor2': float, 'clean': float} if ratings are found,
        otherwise None so the pipeline can still run without this model.
    """
    if session_tag not in ("premeal", "postmeal"):
        raise ValueError("session_tag must be 'premeal' or 'postmeal'")

    if ratings_file is None:
        ratings_file = _default_subject_summary_path(sub)

    if not os.path.exists(ratings_file):
        print(
            "[INFO] Pleasantness ratings file not found; skipping pleasantness "
            f"model: {ratings_file}"
        )
        return None

    df = pd.read_csv(ratings_file).copy()

    # Wide override format: one row per subject
    if "sub" in df.columns:
        row = df.loc[df["sub"].astype(str) == str(sub)]
        if row.empty:
            print(
                "[INFO] No pleasantness ratings found for subject; skipping "
                f"pleasantness model: {sub}"
            )
            return None

        prefix = "pre" if session_tag == "premeal" else "post"
        required_cols = [
            f"{prefix}_odor1",
            f"{prefix}_odor2",
            f"{prefix}_clean",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                "Pleasantness ratings file is missing required columns: "
                f"{missing_cols}"
            )

        row0 = row.iloc[0]
        return {
            "odor1": float(row0[f"{prefix}_odor1"]),
            "odor2": float(row0[f"{prefix}_odor2"]),
            "clean": float(row0[f"{prefix}_clean"]),
        }

    # Default long format: one file per subject with odor_num coding
    required_cols = {"odor_num", "pre_mean", "post_mean"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            "Pleasantness summary file is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    session_col = "pre_mean" if session_tag == "premeal" else "post_mean"
    value_map = dict(zip(df["odor_num"].astype(int), df[session_col].astype(float)))
    missing_codes = {0, 1, 2} - set(value_map.keys())
    if missing_codes:
        raise ValueError(
            "Pleasantness summary file is missing required odor_num values: "
            f"{sorted(missing_codes)}"
        )

    return {
        "odor1": float(value_map[1]),
        "odor2": float(value_map[2]),
        "clean": float(value_map[0]),
    }
