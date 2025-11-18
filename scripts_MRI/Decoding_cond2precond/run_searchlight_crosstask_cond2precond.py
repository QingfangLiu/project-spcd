# run_searchlight_crosstask_cond2precond.py
#
# Goal 1: Pre-conditioned → Conditioned cue reactivation
#
# TRAIN: Day-1 localizer B cues (conditioned; character/place depends on subject)
# TEST : Day-2 pre-meal probe A cues (pre-conditioned)
# LABELS: 6-way B-identity (B1..B6 -> 0..5)
#
# Interpretation:
#   Above-chance decoding of A cues (using B-trained classifier) indicates
#   that pre-conditioned cues evoke patterns resembling their associated
#   conditioned cues.

import os
import argparse
import numpy as np

from nilearn.decoding import SearchLight
from nilearn.image import load_img, concat_imgs, new_img_like
from sklearn.svm import SVC

from decoding_utils import (
    load_decoding_data_conditioned_B,
    load_decoding_data_preconditioned_A,
)

def run_one_subject(
    sub: str,
    base_dir: str,
    mask_img_path: str,
    radius_mm: float = 8.0,
    n_jobs: int = -1,
    session_folder: str = "premeal_trialwise_2mm",
    session_tag: str = "premeal",
):
    out_dir = os.path.join(base_dir, "searchlight_maps", "cue_reactivation")
    os.makedirs(out_dir, exist_ok=True)

    tag_stem = (
        f"{sub}_session-{session_tag}_mode-cond2precond_"
        f"classes-6_radius-{int(radius_mm)}mm"
    )
    out_path = os.path.join(out_dir, f"{tag_stem}.nii.gz")
    if os.path.exists(out_path):
        print(f"⏭️ Skipping {sub} (already exists): {out_path}")
        return

    # -----------------------------
    # TRAIN: Day-1 conditioned B cues
    # -----------------------------
    train_imgs, y_train = load_decoding_data_conditioned_B(
        sub=sub,
        base_dir=base_dir,
        runs=(1, 2, 3, 4),
        normalize=True,
    )

    # -----------------------------
    # TEST: Day-2 pre-meal A cues
    # -----------------------------
    test_imgs, y_test = load_decoding_data_preconditioned_A(
        sub=sub,
        base_dir=base_dir,
        runs=(1, 2, 3),
        normalize=True,
        session_folder=session_folder,
        session_tag=session_tag,
    )

    # Concatenate TRAIN + TEST into single dataset for nilearn SearchLight
    X = concat_imgs([train_imgs, test_imgs]) # train_imgs and test_imgs are both 4D images: n_trials,x,y,z
    y = np.concatenate([y_train, y_test]) # 1D arrays of labels per trial

    # Add diagnostic prints in your script:
    print(f"Train shape: {train_imgs.shape}")
    print(f"Test shape: {test_imgs.shape}")
    print(f"Train labels: {np.unique(y_train, return_counts=True)}")
    print(f"Test labels: {np.unique(y_test, return_counts=True)}")
    print(f"Train label range: {y_train.min()}-{y_train.max()}")
    print(f"Test label range: {y_test.min()}-{y_test.max()}")

    n_train = len(y_train) # number of train trials
    n_total = len(y) # number of all trials
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_total)
    splits = [(train_idx, test_idx)] # used by SearchLight, train classifier on train_idx and test on test_idx

    # -----------------------------
    # Searchlight decoding
    # -----------------------------
    mask_img = load_img(mask_img_path)
    sl = SearchLight(
        mask_img=mask_img,
        radius=radius_mm,
        estimator=SVC(kernel="linear", C=1.0),
        n_jobs=n_jobs,
        cv=splits,           # a train-test transfer
        scoring="accuracy",
        verbose=1,
    )

    print(
        f"\nSubject {sub} | cond→precond | "
        f"classes=6 | radius={radius_mm}mm | "
        f"TRAIN=B(localizer) | TEST=A(probe,{session_tag})"
    )

    # Check label distribution
    print(f"Label distribution train: {np.bincount(y_train)}")
    print(f"Label distribution test: {np.bincount(y_test)}")

    # Check if test labels actually match B-identities
    print(f"Unique train labels: {np.unique(y_train)}")
    print(f"Unique test labels: {np.unique(y_test)}")

    sl.fit(X, y)

    out_img = new_img_like(mask_img, sl.scores_.astype(np.float32))
    out_img.to_filename(out_path)
    print(f"✅ Saved map: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, help="e.g., sub-01")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--mask_img", required=True)
    parser.add_argument("--radius_mm", type=float, default=8.0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--session_folder", default="premeal_trialwise_2mm")
    parser.add_argument("--session_tag", default="premeal")

    args = parser.parse_args()
    run_one_subject(
        sub=args.sub,
        base_dir=args.base_dir,
        mask_img_path=args.mask_img,
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        session_folder=args.session_folder,
        session_tag=args.session_tag,
    )
