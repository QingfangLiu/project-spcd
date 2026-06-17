from __future__ import annotations

import os
import numpy as np
from typing import Optional, Sequence

from nilearn.image import concat_imgs, new_img_like, load_img

from .io_imagewise import build_imagewise_patterns_for_sessions
from .estimator import SingleModelRSA
from .models import (
    create_odor_identity_rdm,
)
from .save_maps import save_single_rsa_map

MODEL_NAMES = (
    "odor_identity_no_image",
)


def searchlight_rsa_single_subject(
    sub: str,
    base_dir: str,
    session_tag: str | Sequence[str],
    radius: float,
    n_jobs: int,
    cue_subset: str = "conditioned_odor_only",
    center_within_run: bool = False,
    output_dir: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
):
    from nilearn.decoding import SearchLight

    session_tags = (session_tag,) if isinstance(session_tag, str) else tuple(session_tag)
    if session_tags != ("premeal",):
        raise ValueError("This odor-identity variant is premeal-only.")

    run_ids, sample_labels, session_labels, img_list = (
        build_imagewise_patterns_for_sessions(
            sub=sub,
            base_dir=base_dir,
            session_tags=session_tags,
            cue_subset=cue_subset,
        )
    )

    n_samples = len(img_list)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for searchlight RSA.")

    if (
        len(run_ids) != n_samples
        or len(sample_labels) != n_samples
        or len(session_labels) != n_samples
    ):
        raise ValueError("Loader output lengths do not match number of image samples.")

    model_rdms = {
        "odor_identity_no_image": create_odor_identity_rdm(sample_labels),
    }
    model_rdms = {name: model_rdms[name] for name in MODEL_NAMES}

    if targets is None:
        targets = list(MODEL_NAMES)
    else:
        invalid_targets = sorted(set(targets) - set(MODEL_NAMES))
        if invalid_targets:
            raise ValueError(f"Unknown target regressor(s): {invalid_targets}")

    img_4d = concat_imgs(img_list)

    mask_file = os.path.join(base_dir, "masks", "gm_0.1.nii")
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    mask_img = load_img(mask_file)

    rsa_maps = {}
    y_dummy = np.arange(n_samples)

    for target_name in targets:
        print(f"Running searchlight for target regressor: {target_name}")

        estimator = SingleModelRSA(
            model_rdms=model_rdms,
            target_name=target_name,
            run_ids=run_ids,
            center_within_run=center_within_run,
        )

        searchlight = SearchLight(
            mask_img=mask_img,
            radius=radius,
            estimator=estimator,
            cv=[(np.arange(n_samples), np.arange(n_samples))],
            n_jobs=n_jobs,
            verbose=1,
        )

        searchlight.fit(img_4d, y_dummy)
        rsa_map = new_img_like(mask_img, searchlight.scores_)
        rsa_maps[target_name] = rsa_map

        if output_dir is not None:
            save_single_rsa_map(
                sub=sub,
                model_name=target_name,
                rsa_map=rsa_map,
                output_dir=output_dir,
                cue_subset=cue_subset,
            )

    return rsa_maps
