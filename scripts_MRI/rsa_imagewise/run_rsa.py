#!/usr/bin/env python3
"""
Image-wise RSA with identity model + button nuisance.

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

from .searchlight import searchlight_rsa_single_subject
from .save_maps import save_rsa_maps


def main():
    parser = argparse.ArgumentParser(
        description="Run image-wise searchlight RSA (identity model + button nuisance)"
    )
    parser.add_argument("--sub", type=str, required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument("--session", type=str, choices=["premeal", "postmeal"], required=True, help="Session to analyze")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing GLM and nii folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
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
        "--center_within_run",
        action="store_true",
        help="Center patterns within each run before correlation RSA",
    )

    args = parser.parse_args()

    session_folder = f"{args.session}_trialwise_2mm"

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.base_dir,
            "rsa_maps",
            "probe_imagewise",
            args.session,
            "input-beta",
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
    print(f"Center within run:   {args.center_within_run}")
    print("=" * 80)

    try:
        rsa_maps = searchlight_rsa_single_subject(
            sub=args.sub,
            base_dir=args.base_dir,
            session_folder=session_folder,
            session_tag=args.session,
            radius=args.radius,
            n_jobs=args.n_jobs,
            button_scope=args.button_scope,
            center_within_run=args.center_within_run,
        )
    except Exception as e:
        print(f"\nERROR during searchlight: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nSaving results...")
    save_rsa_maps(
        sub=args.sub,
        rsa_maps=rsa_maps,
        output_dir=args.output_dir,
        radius=args.radius,
        button_scope=args.button_scope,
    )

    print("\n" + "=" * 80)
    print("RSA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
