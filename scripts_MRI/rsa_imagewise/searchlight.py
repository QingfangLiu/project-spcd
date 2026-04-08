import os
import numpy as np
from typing import Optional, Sequence

from nilearn.image import concat_imgs, new_img_like, load_img
from .io_imagewise import build_imagewise_patterns
from .estimator import SingleModelRSA
from .models import (
    create_odor_group_rdm,
    create_conditioning_rdm,
    create_image_identity_rdm,
    create_button_presence_rdm,
    create_button_side_rdm,
)
from .save_maps import save_single_rsa_map

MODEL_NAMES = (
    "odor_group",
    "conditioning",
    "image_identity",
    "button_presence",
    "button_side",
)


def searchlight_rsa_single_subject(
    sub: str,
    base_dir: str,
    session_folder: str,
    session_tag: str,
    radius: float,
    n_jobs: int,
    button_scope: str = "both_images",
    center_within_run: bool = False,
    output_dir: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
):
    from nilearn.decoding import SearchLight

    run_ids, button_labels, sample_labels, img_list = build_imagewise_patterns(
        sub=sub,
        base_dir=base_dir,
        session_folder=session_folder,
        session_tag=session_tag,
        button_scope=button_scope,
    )

    n_samples = len(img_list)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for searchlight RSA.")

    if (
        len(run_ids) != n_samples
        or len(button_labels) != n_samples
        or len(sample_labels) != n_samples
    ):
        raise ValueError("Loader output lengths do not match number of image samples.")

    odor_group_rdm = create_odor_group_rdm(sample_labels)
    conditioning_rdm = create_conditioning_rdm(sample_labels)
    image_identity_rdm = create_image_identity_rdm(sample_labels)
    button_presence_rdm = create_button_presence_rdm(button_labels)
    button_side_rdm = create_button_side_rdm(button_labels)

    model_rdms = {
        "odor_group": odor_group_rdm,
        "conditioning": conditioning_rdm,
        "image_identity": image_identity_rdm,
        "button_presence": button_presence_rdm,
        "button_side": button_side_rdm,
    }

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
                radius=radius,
                button_scope=button_scope,
            )

    return rsa_maps
    
