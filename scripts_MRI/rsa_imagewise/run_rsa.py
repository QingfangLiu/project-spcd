#!/usr/bin/env python3
"""
Image-wise RSA with cue/image models + button nuisance.

Usage:
    python -m rsa_imagewise.run_rsa \
        --sub sub-01 \
        --session premeal \
        --base_dir /Users/liuq13/SPCD_data \
        --button_scope both_images
"""

import argparse
import os
import sys

from .searchlight import searchlight_rsa_single_subject, MODEL_NAMES
from .ratings import load_subject_pleasantness_ratings


def main():
    parser = argparse.ArgumentParser(
        description="Run image-wise searchlight RSA (cue/value/image models + button nuisance)"
    )
    parser.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument("--session", type=str, choices=["premeal", "postmeal"], required=True, help="Session to analyze")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing GLM and nii folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--ratings_file",
        type=str,
        default=None,
        help="Optional CSV with subject pre/post pleasantness ratings",
    )
    parser.add_argument("--radius", type=float, default=8.0, help="Searchlight radius in mm")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument(
        "--button_scope",
        type=str,
        choices=["both_images", "second_image_only"],
        default="both_images",
        help="Apply trial button label to both image presentations or only the second image",
    )
    parser.add_argument(
        "--cue_subset",
        type=str,
        choices=["all", "conditioned_only", "preconditioned_only"],
        default="all",
        help="Restrict RSA samples to all cues, conditioned cues (B/D), or preconditioned cues (A/C)",
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
        default=None,
        help="Run only one target regressor map while still controlling for all model RDMs",
    )

    args = parser.parse_args()

    session_folder = f"{args.session}_trialwise_2mm"

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.base_dir,
            "rsa_maps",
            "probe_imagewise",
            args.session,
        )

    print("=" * 80)
    print("Image-wise RSA Searchlight Analysis")
    print("=" * 80)
    print(f"Subject:            {args.sub}")
    print(f"Session:            {args.session}")
    print(f"Base dir:           {args.base_dir}")
    print(f"Session folder:     {session_folder}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Radius:             {args.radius} mm")
    print(f"N jobs:             {args.n_jobs}")
    print(f"Button scope:       {args.button_scope}")
    print(f"Cue subset:         {args.cue_subset}")
    print(f"Center within run:   {args.center_within_run}")
    print(f"Target:             {args.target or 'all'}")
    print("=" * 80)

    ratings_by_group = load_subject_pleasantness_ratings(
        sub=args.sub,
        session_tag=args.session,
        ratings_file=args.ratings_file,
    )
    print(
        "Pleasantness model: ",
        "enabled" if ratings_by_group is not None else "disabled",
        sep="",
    )

    if args.target == "pleasantness" and ratings_by_group is None:
        raise ValueError(
            "Target 'pleasantness' requires subject pleasantness ratings. "
            "Provide --ratings_file or add the placeholder ratings CSV."
        )

    try:
        searchlight_rsa_single_subject(
            sub=args.sub,
            base_dir=args.base_dir,
            session_folder=session_folder,
            session_tag=args.session,
            radius=args.radius,
            n_jobs=args.n_jobs,
            button_scope=args.button_scope,
            cue_subset=args.cue_subset,
            center_within_run=args.center_within_run,
            output_dir=args.output_dir,
            targets=[args.target] if args.target is not None else None,
            ratings_by_group=ratings_by_group,
        )
    except Exception as e:
        print(f"\nERROR during searchlight: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 80)
    print("RSA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
