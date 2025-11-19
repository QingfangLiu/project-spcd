# run_searchlight_crosstask_precond2cond_odorpred.py
#
# Goal 4: Conditioned cue → Pre-conditioned cue reactivation (odor prediction)
#
# TRAIN: Day-1 localizer A cues (pre-conditioned; character/place depends on subject)
# TEST : Day-2 odor-prediction B cues (conditioned)
# LABELS: 6-way B-identity (B1..B6 -> 0..5), with A cues mapped via A->B.
#
# Interpretation:
#   Above-chance decoding of B cues (using A-trained classifier) indicates
#   that conditioned cues during odor prediction evoke patterns resembling
#   their associated pre-conditioned cues.

import os
import argparse
import numpy as np

from nilearn.decoding import SearchLight
from nilearn.image import load_img, concat_imgs, new_img_like
from sklearn.svm import SVC

from decoding_utils import (
    load_decoding_data_preconditioned_A_localizer,
    load_decoding_data_conditioned_B_odorpred,
)


def run_one_subject(
    sub: str,
    base_dir: str,
    mask_img_path: str,
    radius_mm: float = 8.0,
    n_jobs: int = -1,
    odorpred_glm_folder: str = "odorpred_cueEARLY_trialwise_2mm",
):
    out_dir = os.path.join(base_dir, "searchlight_maps", "odorpred_crosstask")
    os.makedirs(out_dir, exist_ok=True)

    glm_tag = odorpred_glm_folder.replace("odorpred_", "")
    tag_stem = (
        f"{sub}_session-odorpred_mode-precond2cond_{glm_tag}_"
        f"classes-6_radius-{int(radius_mm)}mm"
    )
    out_path = os.path.join(out_dir, f"{tag_stem}.nii.gz")
    if os.path.exists(out_path):
        print(f"⏭️ Skipping {sub} (already exists): {out_path}")
        return

    # -----------------------------
    # TRAIN: Day-1 pre-conditioned A cues (localizer)
    # -----------------------------
    train_imgs, y_train = load_decoding_data_preconditioned_A_localizer(
        sub=sub,
        base_dir=base_dir,
        runs=(1, 2, 3, 4),
        normalize=True,
    )

    # -----------------------------
    # TEST: Day-2 odor-prediction B cues
    # -----------------------------
    test_imgs, y_test = load_decoding_data_conditioned_B_odorpred(
        sub=sub,
        base_dir=base_dir,
        runs=(1, 2, 3),
        odorpred_glm_folder=odorpred_glm_folder,
        normalize=True,
    )

    # Concatenate TRAIN + TEST into single dataset for nilearn SearchLight
    X = concat_imgs([train_imgs, test_imgs])  # 4D: n_trials,x,y,z
    y = np.concatenate([y_train, y_test])     # 1D labels per trial

    # Diagnostics
    print(f"Train shape: {train_imgs.shape}")
    print(f"Test shape:  {test_imgs.shape}")
    print(f"Train labels: {np.unique(y_train, return_counts=True)}")
    print(f"Test labels:  {np.unique(y_test, return_counts=True)}")
    print(f"Train label range: {y_train.min()}-{y_train.max()}")
    print(f"Test label range:  {y_test.min()}-{y_test.max()}")

    n_train = len(y_train)
    n_total = len(y)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_total)
    splits = [(train_idx, test_idx)]  # train on train_idx, test on test_idx

    # -----------------------------
    # Searchlight decoding
    # -----------------------------
    mask_img = load_img(mask_img_path)
    sl = SearchLight(
        mask_img=mask_img,
        radius=radius_mm,
        estimator=SVC(kernel="linear", C=1.0),
        n_jobs=n_jobs,
        cv=splits,
        scoring="accuracy",
        verbose=1,
    )

    print(
        f"\nSubject {sub} | precond→cond (odorpred) | "
        f"train n={len(y_train)}, test n={len(y_test)}"
    )

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
    parser.add_argument(
        "--odorpred_glm_folder",
        default="odorpred_cueEARLY_trialwise_2mm",
        help="Which odor-prediction GLM to use "
             "(e.g., odorpred_cueEARLY_trialwise_2mm or odorpred_cueEXT5s_trialwise_2mm)",
    )

    args = parser.parse_args()
    run_one_subject(
        sub=args.sub,
        base_dir=args.base_dir,
        mask_img_path=args.mask_img,
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        odorpred_glm_folder=args.odorpred_glm_folder,
    )
