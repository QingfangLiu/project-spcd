from __future__ import annotations

import os
import json
import numpy as np
from typing import Optional, Sequence

from nilearn.image import concat_imgs, new_img_like, load_img
from .io_imagewise import build_imagewise_patterns_for_sessions
from .estimator import SingleModelRSA
from .models import (
    create_odor_value_rdm,
    create_odor_value_rdm_by_phase,
    create_cue_type_rdm,
    create_image_identity_rdm,
    create_phase_rdm,
    create_button_presence_rdm,
    create_button_side_rdm,
    create_second_image_value_rdms,
    validate_analysis_options,
    condition_to_group,
    base_condition,
)
from .save_maps import save_single_rsa_map

MODEL_NAMES = (
    "odor_value",
    "cue_type",
    "image_identity",
    "button_presence",
    "button_side",
)


def searchlight_rsa_single_subject(
    sub: str,
    base_dir: str,
    session_tag: str | Sequence[str],
    radius: float,
    n_jobs: int,
    button_scope: str = "second_image_only",
    cue_subset: str = "all",
    center_within_run: bool = False,
    output_dir: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
    ratings_by_group: Optional[dict] = None,
    ratings_by_phase: Optional[dict] = None,
    image_scope: str = "both",
    value_model: str = "cue_value",
):
    from nilearn.decoding import SearchLight

    session_tags = (session_tag,) if isinstance(session_tag, str) else tuple(session_tag)
    validate_analysis_options(image_scope, value_model)
    loaded = (
        build_imagewise_patterns_for_sessions(
            sub=sub,
            base_dir=base_dir,
            session_tags=session_tags,
            button_scope=button_scope,
            cue_subset=cue_subset,
            image_scope=image_scope,
            return_trial_metadata=image_scope == "second",
        )
    )
    run_ids, button_labels, sample_labels, phase_labels, img_list = loaded[:5]
    trial_metadata = loaded[5] if image_scope == "second" else None

    n_samples = len(img_list)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for searchlight RSA.")

    if (
        len(run_ids) != n_samples
        or len(button_labels) != n_samples
        or len(sample_labels) != n_samples
        or len(phase_labels) != n_samples
    ):
        raise ValueError("Loader output lengths do not match number of image samples.")

    is_combined_phase = len(set(phase_labels)) > 1
    cue_type_rdm = create_cue_type_rdm(sample_labels)
    image_identity_rdm = create_image_identity_rdm(sample_labels)
    button_presence_rdm = create_button_presence_rdm(button_labels)
    button_side_rdm = create_button_side_rdm(button_labels)

    model_rdms = {
        "cue_type": cue_type_rdm,
        "image_identity": image_identity_rdm,
        "button_presence": button_presence_rdm,
        "button_side": button_side_rdm,
    }

    if image_scope == "second":
        if ratings_by_phase is None and ratings_by_group is not None and len(session_tags) == 1:
            ratings_by_phase = {session_tags[0]: ratings_by_group}
        if ratings_by_phase is None:
            raise ValueError("Second-image value analyses require complete pleasantness ratings")
        trial_variables, value_rdms = create_second_image_value_rdms(
            trial_metadata.first_label, sample_labels, phase_labels, ratings_by_phase)
        model_rdms['odor_value'] = value_rdms[value_model]
    elif ratings_by_phase is not None:
        if is_combined_phase:
            odor_value_rdm = create_odor_value_rdm_by_phase(
                sample_labels=sample_labels,
                phase_labels=phase_labels,
                ratings_by_phase=ratings_by_phase,
            )
        else:
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

    if is_combined_phase:
        phase_rdm = create_phase_rdm(phase_labels)
        print(
            "Phase RDM computed for combined sessions "
            f"(shape={phase_rdm.shape}); excluded from RSA regression."
        )

    if cue_subset in ("conditioned_only", "preconditioned_only"):
        # In single-cue-type subsets, cue_type has no variance and is dropped.
        model_rdms.pop("cue_type", None)

    model_rdms = {name: model_rdms[name] for name in MODEL_NAMES if name in model_rdms}
    metadata_path = None
    if image_scope == "second":
        from .diagnostics import check_second_image_design, write_second_image_diagnostics

        diagnostic_rdms = dict(value_rdms)
        diagnostic_rdms.update({n: r for n, r in model_rdms.items() if n != 'odor_value'})
        diagnostic_rdms['first_image_identity'] = create_image_identity_rdm(trial_metadata.first_label)
        pair_labels = np.array(['|'.join(sorted((
            condition_to_group(base_condition(first)),
            condition_to_group(base_condition(second)),
        ))) for first, second in zip(trial_metadata.first_label, sample_labels)])
        diagnostic_rdms['outcome_pair_identity'] = (pair_labels[:, None] != pair_labels[None, :]).astype(float)
        model_rdms, pair_mask, design_info = check_second_image_design(model_rdms, run_ids)

    available_model_names = tuple(model_rdms.keys())

    if targets is None:
        targets = ['odor_value'] if image_scope == 'second' else list(available_model_names)
    else:
        invalid_targets = sorted(set(targets) - set(available_model_names))
        if invalid_targets:
            raise ValueError(f"Unknown target regressor(s): {invalid_targets}")

    if image_scope == 'second' and output_dir is not None:
        metadata_path = write_second_image_diagnostics(
            output_dir, sub, cue_subset, value_model, center_within_run,
            trial_metadata, trial_variables, diagnostic_rdms, pair_mask,
            design_info, targets)

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

    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text())
        metadata['status'] = 'complete'
        metadata['saved_maps'] = [f'{sub}_model-{name}_subset-{cue_subset}.nii.gz'
                                  for name in rsa_maps]
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return rsa_maps
    
