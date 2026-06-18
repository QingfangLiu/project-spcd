#!/usr/bin/env python3
"""
First-image-only RSA with cue/value/image models.

Usage:
    python -m rsa_imagewise_first_image.run_rsa \
        --sub sub-01 \
        --session both \
        --base_dir /Users/liuq13/SPCD_data
"""

import argparse
import os
import sys

from .ratings import load_subject_pleasantness_ratings

MODEL_NAMES = (
    "odor_value",
    "cue_type",
    "image_identity",
)


def main():
    parser = argparse.ArgumentParser(
        description="Run first-image-only searchlight RSA (cue/value/image models)"
    )
    parser.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument("--session", type=str, choices=["premeal", "postmeal", "both"], required=True, help="Session or combined phase to analyze")
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

    session_tags = ("premeal", "postmeal") if args.session == "both" else (args.session,)
    session_folders = [f"{session_tag}_trialwise_2mm" for session_tag in session_tags]

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.base_dir,
            "rsa_maps",
            "probe_imagewise_first_image",
            args.session,
        )

    print("=" * 80)
    print("First-image-only RSA Searchlight Analysis")
    print("=" * 80)
    print(f"Subject:            {args.sub}")
    print(f"Session:            {args.session}")
    print(f"Base dir:           {args.base_dir}")
    print(f"Session folder(s):  {', '.join(session_folders)}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Radius:             {args.radius} mm")
    print(f"N jobs:             {args.n_jobs}")
    print("Image scope:        first image only")
    print("Button model:       disabled")
    print(f"Cue subset:         {args.cue_subset}")
    print(f"Center within run:   {args.center_within_run}")
    print(f"Target:             {args.target or 'all'}")
    print("=" * 80)
    ratings_by_phase = {}
    for session_tag in session_tags:
        ratings_by_phase[session_tag] = load_subject_pleasantness_ratings(
            sub=args.sub,
            session_tag=session_tag,
            ratings_file=args.ratings_file,
        )
    if any(ratings is None for ratings in ratings_by_phase.values()):
        ratings_by_phase = None

    print(
        "Odor value model: ",
        "enabled" if ratings_by_phase is not None else "disabled",
        sep="",
    )

    if args.target == "odor_value" and ratings_by_phase is None:
        raise ValueError(
            "Target 'odor_value' requires subject pleasantness ratings. "
            "Provide --ratings_file or add the placeholder ratings CSV."
        )

    try:
        from .searchlight import searchlight_rsa_single_subject

        searchlight_rsa_single_subject(
            sub=args.sub,
            base_dir=args.base_dir,
            session_tag=session_tags,
            radius=args.radius,
            n_jobs=args.n_jobs,
            cue_subset=args.cue_subset,
            center_within_run=args.center_within_run,
            output_dir=args.output_dir,
            targets=[args.target] if args.target is not None else None,
            ratings_by_phase=ratings_by_phase,
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
