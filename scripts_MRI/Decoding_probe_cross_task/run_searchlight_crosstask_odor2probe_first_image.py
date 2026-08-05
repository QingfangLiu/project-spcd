#!/usr/bin/env python3
"""Binary odor-to-probe searchlight for one first-image cue subset."""

import argparse
import os

import numpy as np
from nilearn.decoding import SearchLight
from nilearn.image import concat_imgs, load_img, new_img_like
from sklearn.svm import SVC

from odor_training_utils_binary import load_decoding_data_odor_binary
from probe_decoding_utils_first_image import (
    CUE_SUBSETS,
    load_decoding_data_cue_first_image,
)


def run_one_subject(
    sub,
    base_dir,
    mask_img_path,
    cue_subset,
    radius_mm=8.0,
    n_jobs=4,
    session_folder="premeal_trialwise_2mm",
    session_tag="premeal",
    sub14_original_events_dir=None,
):
    """Train odor1-vs-odor2 and test one first-image odor cue subset."""
    out_dir = os.path.join(base_dir, "searchlight_maps", "probe_crosstask")
    os.makedirs(out_dir, exist_ok=True)
    tag_stem = (
        f"{sub}_session-{session_tag}_mode-crosstask_subset-{cue_subset}_"
        f"classes-2_images-first_radius-{int(radius_mm)}mm"
    )
    out_path = os.path.join(out_dir, f"{tag_stem}.nii.gz")
    if os.path.exists(out_path):
        print(f"Skipping {sub} (exists): {out_path}")
        return

    odor_imgs, y_train = load_decoding_data_odor_binary(
        sub=sub,
        base_dir=base_dir,
        runs=(1, 2),
    )
    probe_imgs, y_test, _, _ = load_decoding_data_cue_first_image(
        sub=sub,
        base_dir=base_dir,
        cue_subset=cue_subset,
        runs=(1, 2, 3),
        normalize=True,
        session_folder=session_folder,
        session_tag=session_tag,
        sub14_original_events_dir=sub14_original_events_dir,
    )

    images = concat_imgs([odor_imgs, probe_imgs])
    labels = np.concatenate([y_train, y_test])
    n_train = len(y_train)
    split = [(np.arange(n_train), np.arange(n_train, len(labels)))]

    mask_img = load_img(mask_img_path)
    searchlight = SearchLight(
        mask_img=mask_img,
        radius=radius_mm,
        estimator=SVC(kernel="linear", C=1.0),
        n_jobs=n_jobs,
        cv=split,
        scoring="accuracy",
        verbose=1,
    )
    print(
        f"\nSubject {sub} | subset={cue_subset} | classes=2 | "
        f"radius={radius_mm}mm | train n={len(y_train)} | test n={len(y_test)} | "
        f"TEST=probe({session_tag}, first image only)"
    )
    searchlight.fit(images, labels)
    out_img = new_img_like(mask_img, searchlight.scores_.astype(np.float32))
    out_img.to_filename(out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run binary odor-to-probe decoding for conditioned or "
            "preconditioned first-image cues."
        )
    )
    parser.add_argument("--sub", required=True, help="e.g., sub-01")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--mask_img", required=True)
    parser.add_argument("--cue_subset", choices=CUE_SUBSETS, required=True)
    parser.add_argument("--radius_mm", type=float, default=8.0)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Parallel workers; match this to the CPUs allocated by the scheduler.",
    )
    parser.add_argument("--session_folder", default="premeal_trialwise_2mm")
    parser.add_argument("--session_tag", default="premeal")
    parser.add_argument(
        "--sub14_original_events_dir",
        help=(
            "Directory containing sub-14 *_events_orig.tsv files. Default: "
            "<base_dir>/nii/sub-14/func_PAID/<session_tag>."
        ),
    )
    args = parser.parse_args()

    run_one_subject(
        sub=args.sub,
        base_dir=args.base_dir,
        mask_img_path=args.mask_img,
        cue_subset=args.cue_subset,
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        session_folder=args.session_folder,
        session_tag=args.session_tag,
        sub14_original_events_dir=args.sub14_original_events_dir,
    )


if __name__ == "__main__":
    main()
