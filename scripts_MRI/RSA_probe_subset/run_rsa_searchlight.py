#!/usr/bin/env python3
"""
Run searchlight RSA for one subject.

Usage examples:
  # all cues (default)
  python run_rsa_searchlight.py --sub sub-01 --session premeal --base_dir ~/SPCD_data

  # conditioned only (B/D)
  python run_rsa_searchlight.py --sub sub-01 --session premeal --base_dir ~/SPCD_data --cue_subset conditioned

  # preconditioned only (A/C)
  python run_rsa_searchlight.py --sub sub-01 --session premeal --base_dir ~/SPCD_data --cue_subset preconditioned

  # run both models
  python run_rsa_searchlight.py --sub sub-01 --session premeal --base_dir ~/SPCD_data --models identity value
"""

import argparse
import os
import sys
from rsa_utils import (
    create_identity_model_rdm,
    create_value_model_rdm,
    searchlight_rsa_single_subject,
    save_rsa_maps
)


def main():
    parser = argparse.ArgumentParser(
        description="Run searchlight RSA for probe task"
    )
    parser.add_argument(
        "--sub",
        type=str,
        required=True,
        help="Subject ID (e.g. sub-01)"
    )
    parser.add_argument(
        "--session",
        type=str,
        choices=["premeal", "postmeal"],
        required=True,
        help="Session to analyze"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing GLM folder"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to gray matter mask (default: <base_dir>/masks/gm_0.1.nii)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <base_dir>/rsa_maps/probe_subset)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=8.0,
        help="Searchlight radius in mm (default: 8.0)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1 for all cores)"
    )
    parser.add_argument(
        "--rsa_method",
        type=str,
        choices=["regression", "spearman", "kendall"],
        default="regression",
        help="RSA scoring method (default: regression)"
    )
    parser.add_argument(
        "--neural_metric",
        type=str,
        choices=["correlation", "euclidean"],
        default="correlation",
        help="Neural similarity metric (default: correlation)"
    )
    parser.add_argument(
        "--use_pair_mask",
        action="store_true",
        help="Use odor-only pair mask (exclude clean-involving pairs)."
    )
    parser.add_argument(
        "--cue_subset",
        type=str,
        choices=["all", "conditioned", "preconditioned"],
        default="all",
        help="Run RSA on all cues (12) or subset to conditioned (B/D only) or preconditioned (A/C only)."
    )
    parser.add_argument(
        "--models",
        nargs="+", # this flag must be followed by one or more values
        choices=["identity", "value"],
        default=["identity"],
        help="Which model(s) to run (default: identity). Example: --models identity value"
    )

    args = parser.parse_args()

    # Set defaults
    if args.mask is None:
        args.mask = os.path.join(args.base_dir, "masks", "gm_0.1.nii")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, "rsa_maps", "probe_subset")

    session_folder = f"{args.session}_trialwise_2mm"

    # Check mask exists
    if not os.path.exists(args.mask):
        print(f"ERROR: Mask file not found: {args.mask}")
        sys.exit(1)

    print("=" * 80)
    print("RSA Searchlight Analysis")
    print("=" * 80)
    print(f"Subject:        {args.sub}")
    print(f"Session:        {args.session}")
    print(f"Base dir:       {args.base_dir}")
    print(f"Session folder: {session_folder}")
    print(f"Mask:           {args.mask}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Radius:         {args.radius} mm")
    print(f"N jobs:         {args.n_jobs}")
    print(f"RSA method:     {args.rsa_method}")
    print(f"Neural metric:  {args.neural_metric}")
    print(f"use_pair_mask:  {args.use_pair_mask}")
    print(f"cue_subset:     {args.cue_subset}")
    print(f"models:         {args.models}")
    print("=" * 80)

    # Create model RDMs (association skipped completely)
    print("\nCreating model RDMs...")
    all_model_rdms = {
        "identity": create_identity_model_rdm(),
        "value": create_value_model_rdm(),
    }
    model_rdms = {k: all_model_rdms[k] for k in args.models}

    print("\nModels used for analysis:")
    for name in model_rdms:
        print(f"  - {name}")

    # Run searchlight RSA
    print("\nRunning searchlight RSA...")
    try:
        rsa_maps = searchlight_rsa_single_subject(
            sub=args.sub,
            base_dir=args.base_dir,
            session_folder=session_folder,
            session_tag=args.session,
            mask_file=args.mask,
            model_rdms=model_rdms,
            radius=args.radius,
            n_jobs=args.n_jobs,
            rsa_method=args.rsa_method,
            neural_metric=args.neural_metric,
            use_pair_mask=args.use_pair_mask, 
            cue_subset=args.cue_subset, 
        )
    except Exception as e:
        print(f"\nERROR during searchlight: {e}") # print the error message
        import traceback
        traceback.print_exc() # print the full traceback of the exception that is currently being handled
        sys.exit(1) # stops python and sends exit code 1 (0 is success)

    # Save results
    print("\nSaving results...")
    save_rsa_maps(
        sub=args.sub,
        rsa_maps=rsa_maps,
        output_dir=args.output_dir,
        session_tag=args.session,
        radius=args.radius,
        rsa_method=args.rsa_method,
        neural_metric=args.neural_metric,
        use_pair_mask=args.use_pair_mask,
        cue_subset=args.cue_subset,            # <-- new (adds subset label to filename)
    )

    print("\n" + "=" * 80)
    print("RSA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
