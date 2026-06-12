#!/usr/bin/env python3
"""
Premeal image-wise RSA for odor identity with independent visual-image control.

Usage:
    python -m rsa_imagewise_odor_identity_visual_image.run_rsa \
        --sub sub-01 \
        --base_dir /Users/liuq13/SPCD_data \
        --cue_subset conditioned_odor_only \
        --target odor_identity
"""

import argparse
import os
import sys

from .visual_rdm import DEFAULT_VISUAL_MASK

MODEL_NAMES = (
    "odor_identity",
    "visual_image",
    "button_presence",
    "button_side",
)

CUE_SUBSETS = (
    "conditioned_odor_only",
    "preconditioned_odor_only",
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run premeal image-wise searchlight RSA for odor identity, replacing "
            "the binary image-identity control with a subject-specific visual "
            "RDM from independent image-scan betas."
        )
    )
    parser.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument(
        "--session",
        type=str,
        choices=["premeal"],
        default="premeal",
        help="Only premeal is supported in this variant",
    )
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing GLM and nii folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--radius", type=float, default=8.0, help="Searchlight radius in mm")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument(
        "--button_scope",
        type=str,
        choices=["both_images", "second_image_only"],
        default="second_image_only",
        help="Apply trial button label to both image presentations or only the second image",
    )
    parser.add_argument(
        "--cue_subset",
        type=str,
        choices=CUE_SUBSETS,
        required=True,
        help="Use conditioned or preconditioned odor-predictive cues only",
    )
    parser.add_argument(
        "--center_within_run",
        action="store_true",
        help="Center patterns within each run before correlation RSA",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=MODEL_NAMES,
        default="odor_identity",
        help="Run one target regressor map while still controlling for all model RDMs",
    )
    parser.add_argument(
        "--visual_mask",
        type=str,
        default=str(DEFAULT_VISUAL_MASK),
        help="Visual mask path, absolute or relative to base_dir",
    )
    parser.add_argument(
        "--no_visual_zscore_across_trials",
        action="store_true",
        help="Disable voxelwise within-run z-scoring across all independent localizer trials",
    )
    parser.add_argument(
        "--save_visual_diagnostics",
        action="store_true",
        help="Save per-subject/subset visual RDM CSVs under <output_dir>/diagnostics",
    )

    args = parser.parse_args()

    session_tags = (args.session,)
    session_folders = [f"{session_tag}_trialwise_2mm" for session_tag in session_tags]

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.base_dir,
            "rsa_maps",
            "probe_imagewise_odor_identity_visual_image",
            args.session,
        )

    print("=" * 80)
    print("Premeal Image-wise Odor-Identity RSA Searchlight with Visual-Image Control")
    print("=" * 80)
    print(f"Subject:            {args.sub}")
    print(f"Session:            {args.session}")
    print(f"Base dir:           {args.base_dir}")
    print(f"Session folder(s):  {', '.join(session_folders)}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Radius:             {args.radius} mm")
    print(f"N jobs:             {args.n_jobs}")
    print(f"Button scope:       {args.button_scope}")
    print(f"Cue subset:         {args.cue_subset}")
    print(f"Center within run:  {args.center_within_run}")
    print(f"Target:             {args.target}")
    print(f"Visual mask:        {args.visual_mask}")
    print(f"Visual z-score:     {not args.no_visual_zscore_across_trials}")
    print(f"Visual diagnostics: {args.save_visual_diagnostics}")
    print("Odor value model:   disabled")
    print("=" * 80)
    if args.button_scope != "second_image_only":
        print(
            "[INFO] Output filenames do not include button_scope; use a separate "
            "--output_dir if comparing button-scope variants."
        )

    try:
        from .searchlight import searchlight_rsa_single_subject

        searchlight_rsa_single_subject(
            sub=args.sub,
            base_dir=args.base_dir,
            session_tag=session_tags,
            radius=args.radius,
            n_jobs=args.n_jobs,
            button_scope=args.button_scope,
            cue_subset=args.cue_subset,
            center_within_run=args.center_within_run,
            output_dir=args.output_dir,
            targets=[args.target],
            visual_mask=args.visual_mask,
            visual_zscore_across_trials=not args.no_visual_zscore_across_trials,
            save_visual_diagnostics=args.save_visual_diagnostics,
        )
    except Exception as e:
        print(f"\nERROR during searchlight: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Odor-identity visual-image RSA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
