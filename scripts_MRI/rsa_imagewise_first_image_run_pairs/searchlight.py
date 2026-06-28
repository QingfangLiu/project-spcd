from __future__ import annotations

import os
import numpy as np
from typing import Optional, Sequence

from nilearn.image import concat_imgs, new_img_like, load_img
from .io_imagewise import build_imagewise_patterns_for_sessions
from .estimator import RUN_PAIR_TO_RUNS, SingleModelRSA
from .models import (
    create_odor_value_rdm,
    create_cue_type_rdm,
    create_image_identity_rdm,
)
from .save_maps import save_single_rsa_map

MODEL_NAMES = (
    "odor_value",
    "cue_type",
    "image_identity",
)


def searchlight_rsa_single_subject(
    sub: str,
    base_dir: str,
    session_tag: str | Sequence[str],
    radius: float,
    n_jobs: int,
    cue_subset: str = "all",
    center_within_run: bool = False,
    output_dir: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
    ratings_by_group: Optional[dict] = None,
    ratings_by_phase: Optional[dict] = None,
    run_pair: str = "run1_vs_run2",
):
    from nilearn.decoding import SearchLight

    session_tags = (session_tag,) if isinstance(session_tag, str) else tuple(session_tag)
    if len(session_tags) != 1:
        raise ValueError("Run-pair RSA supports one session at a time: premeal or postmeal.")
    if run_pair not in RUN_PAIR_TO_RUNS:
        raise ValueError(f"Unknown run_pair={run_pair!r}. Expected one of: {sorted(RUN_PAIR_TO_RUNS)}")

    run_ids, sample_labels, phase_labels, img_list = (
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
        or len(phase_labels) != n_samples
    ):
        raise ValueError("Loader output lengths do not match number of image samples.")

    cue_type_rdm = create_cue_type_rdm(sample_labels)
    image_identity_rdm = create_image_identity_rdm(sample_labels)

    model_rdms = {
        "cue_type": cue_type_rdm,
        "image_identity": image_identity_rdm,
    }

    if ratings_by_phase is not None:
        phase = str(phase_labels[0])
        odor_value_rdm = create_odor_value_rdm(
            sample_labels=sample_labels,
            ratings_by_group=ratings_by_phase[phase],
        )
        model_rdms["odor_value"] = odor_value_rdm
    elif ratings_by_group is not None:
        odor_value_rdm = create_odor_value_rdm(
            sample_labels=sample_labels,
            ratings_by_group=ratings_by_group,
        )
        model_rdms["odor_value"] = odor_value_rdm

    if cue_subset in ("conditioned_only", "preconditioned_only"):
        # In single-cue-type subsets, cue_type has no variance and is dropped.
        model_rdms.pop("cue_type", None)

    model_rdms = {name: model_rdms[name] for name in MODEL_NAMES if name in model_rdms}
    available_model_names = tuple(model_rdms.keys())

    if targets is None:
        targets = list(available_model_names)
    else:
        invalid_targets = sorted(set(targets) - set(available_model_names))
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
        print(f"Running searchlight for target regressor: {target_name} | {run_pair}")

        estimator = SingleModelRSA(
            model_rdms=model_rdms,
            target_name=target_name,
            run_ids=run_ids,
            center_within_run=center_within_run,
            run_pair=run_pair,
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
    
