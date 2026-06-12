from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from nilearn.image import concat_imgs, load_img, new_img_like

from .estimator import LocalVisualSingleModelRSA
from .io_imagewise import build_imagewise_patterns_for_sessions
from .models import (
    create_button_presence_rdm,
    create_button_side_rdm,
    create_odor_identity_rdm,
)
from .save_maps import save_single_rsa_map
from .visual_rdm import build_localizer_images_for_local_visual_rsa

MODEL_NAMES = (
    "odor_identity",
    "visual_image",
    "button_presence",
    "button_side",
)


def _write_localizer_diagnostics(
    diagnostics_dir: Path,
    sub: str,
    cue_subset: str,
    localizer_rows: list[dict],
    summary: dict,
) -> None:
    sub_dir = diagnostics_dir / sub
    sub_dir.mkdir(parents=True, exist_ok=True)
    prefix = cue_subset

    pd.DataFrame(localizer_rows).to_csv(
        sub_dir / f"{prefix}_local_visual_localizer_samples.csv",
        index=False,
        na_rep="NaN",
    )
    pd.DataFrame([summary]).to_csv(
        sub_dir / f"{prefix}_local_visual_input_summary.csv",
        index=False,
        na_rep="NaN",
    )


def _write_score_qc(
    diagnostics_dir: Path,
    sub: str,
    cue_subset: str,
    target_name: str,
    scores: np.ndarray,
    mask_data: np.ndarray,
) -> None:
    sub_dir = diagnostics_dir / sub
    sub_dir.mkdir(parents=True, exist_ok=True)

    in_mask_scores = np.asarray(scores, dtype=float)[mask_data]
    finite_scores = in_mask_scores[np.isfinite(in_mask_scores)]
    row = {
        "sub": sub,
        "cue_subset": cue_subset,
        "target": target_name,
        "n_searchlights_in_mask": int(in_mask_scores.size),
        "n_searchlights_valid": int(finite_scores.size),
        "n_searchlights_nan": int(np.isnan(in_mask_scores).sum()),
        "valid_fraction": float(finite_scores.size / in_mask_scores.size)
        if in_mask_scores.size
        else np.nan,
        "score_mean": float(np.mean(finite_scores)) if finite_scores.size else np.nan,
        "score_sd": float(np.std(finite_scores, ddof=1))
        if finite_scores.size > 1
        else np.nan,
        "score_min": float(np.min(finite_scores)) if finite_scores.size else np.nan,
        "score_max": float(np.max(finite_scores)) if finite_scores.size else np.nan,
    }

    out_path = sub_dir / f"{cue_subset}_local_visual_searchlight_qc_model-{target_name}.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def searchlight_rsa_single_subject(
    sub: str,
    base_dir: str,
    session_tag: str | Sequence[str],
    radius: float,
    n_jobs: int,
    button_scope: str = "second_image_only",
    cue_subset: str = "conditioned_odor_only",
    center_within_run: bool = False,
    output_dir: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
    visual_zscore_across_trials: bool = True,
    save_visual_diagnostics: bool = False,
):
    from nilearn.decoding import SearchLight

    session_tags = (session_tag,) if isinstance(session_tag, str) else tuple(session_tag)
    if session_tags != ("premeal",):
        raise ValueError("This local-visual odor-identity variant is premeal-only.")

    run_ids, button_labels, sample_labels, session_labels, probe_img_list = (
        build_imagewise_patterns_for_sessions(
            sub=sub,
            base_dir=base_dir,
            session_tags=session_tags,
            button_scope=button_scope,
            cue_subset=cue_subset,
        )
    )

    n_probe_samples = len(probe_img_list)
    if n_probe_samples < 2:
        raise ValueError("Need at least 2 probe samples for searchlight RSA.")

    if (
        len(run_ids) != n_probe_samples
        or len(button_labels) != n_probe_samples
        or len(sample_labels) != n_probe_samples
        or len(session_labels) != n_probe_samples
    ):
        raise ValueError("Loader output lengths do not match number of probe images.")

    (
        localizer_img_list,
        localizer_run_ids,
        localizer_condition_labels,
        localizer_condition_order,
        localizer_rows,
        localizer_summary,
    ) = build_localizer_images_for_local_visual_rsa(
        sub=sub,
        base_dir=base_dir,
        sample_labels=sample_labels,
        cue_subset=cue_subset,
    )

    diagnostics_dir = None
    if save_visual_diagnostics and output_dir is not None:
        diagnostics_dir = Path(output_dir) / "diagnostics"
        _write_localizer_diagnostics(
            diagnostics_dir=diagnostics_dir,
            sub=sub,
            cue_subset=cue_subset,
            localizer_rows=localizer_rows,
            summary=localizer_summary,
        )

    fixed_model_rdms = {
        "odor_identity": create_odor_identity_rdm(sample_labels),
        "button_presence": create_button_presence_rdm(button_labels),
        "button_side": create_button_side_rdm(button_labels),
    }

    if targets is None:
        targets = list(MODEL_NAMES)
    else:
        invalid_targets = sorted(set(targets) - set(MODEL_NAMES))
        if invalid_targets:
            raise ValueError(f"Unknown target regressor(s): {invalid_targets}")

    combined_img_list = list(probe_img_list) + list(localizer_img_list)
    img_4d = concat_imgs(combined_img_list)
    n_total = len(combined_img_list)

    mask_file = os.path.join(base_dir, "masks", "gm_0.1.nii")
    if not os.path.exists(mask_file):
        raise FileNotFoundError(f"Mask file not found: {mask_file}")

    mask_img = load_img(mask_file)
    mask_data = np.asanyarray(mask_img.dataobj) > 0

    rsa_maps = {}
    y_dummy = np.arange(n_total)
    cv = [(np.arange(n_total), np.arange(n_total))]

    for target_name in targets:
        print(f"Running local-visual searchlight for target regressor: {target_name}")

        estimator = LocalVisualSingleModelRSA(
            fixed_model_rdms=fixed_model_rdms,
            target_name=target_name,
            n_probe_samples=n_probe_samples,
            probe_run_ids=run_ids,
            probe_sample_labels=sample_labels,
            localizer_run_ids=localizer_run_ids,
            localizer_condition_labels=localizer_condition_labels,
            localizer_condition_order=localizer_condition_order,
            center_probe_within_run=center_within_run,
            visual_zscore_across_trials=visual_zscore_across_trials,
        )

        searchlight = SearchLight(
            mask_img=mask_img,
            radius=radius,
            estimator=estimator,
            cv=cv,
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
        if diagnostics_dir is not None:
            _write_score_qc(
                diagnostics_dir=diagnostics_dir,
                sub=sub,
                cue_subset=cue_subset,
                target_name=target_name,
                scores=searchlight.scores_,
                mask_data=mask_data,
            )

    return rsa_maps
