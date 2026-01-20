#!/usr/bin/env python3
"""
Run searchlight RSA for one subject.

Usage:
    python run_rsa_searchlight.py --sub sub-01 --session premeal --base_dir ~/SPCD_data

This script:
1. Loads image-wise betas for all 12 conditions
2. Averages images within each condition
3. Creates association and identity model RDMs
4. Runs whole-brain searchlight RSA
5. Saves beta maps for each model
"""

import argparse
import os
import sys
from rsa_utils import (
    create_association_model_rdm,
    create_identity_model_rdm,
    searchlight_rsa_single_subject,
    save_rsa_maps
)


def main():
    parser = argparse.ArgumentParser( # create the parser (empty for now)
        description="Run searchlight RSA for probe task"
    )
    parser.add_argument( # start defining arguments
        "--sub",
        type=str,
        required=True,
        help="Subject ID (e.g., sub-01)"
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
        help="Output directory (default: <base_dir>/rsa_maps/probe)"
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
    
    args = parser.parse_args() # reads sys.argv, matches values to rules, convers types, returns a simple object
    
    # Set defaults
    if args.mask is None:
        args.mask = os.path.join(args.base_dir, "masks", "gm_0.1.nii")
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.base_dir, "rsa_maps", "probe")

    if args.rsa_method is None:
        args.rsa_method = "regression"

    if args.neural_metric is None:
        args.neural_metric = "correlation"
    
    # Determine session folder
    session_folder = f"{args.session}_trialwise_2mm"
    
    # Check mask exists
    if not os.path.exists(args.mask):
        print(f"ERROR: Mask file not found: {args.mask}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"RSA Searchlight Analysis")
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
    print("=" * 80)
    
    # Create model RDMs
    print("\nCreating model RDMs...")
    association_rdm = create_association_model_rdm()
    identity_rdm = create_identity_model_rdm()
    
    model_rdms = {
        'association': association_rdm,
        'identity': identity_rdm
    }
    
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
            neural_metric=args.neural_metric
        )
    except Exception as e:
        print(f"\nERROR during searchlight: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    print("\nSaving results...")
    save_rsa_maps(
        sub=args.sub,
        rsa_maps=rsa_maps,
        output_dir=args.output_dir,
        session_tag=args.session,
        radius=args.radius,
        rsa_method=args.rsa_method,
        neural_metric=args.neural_metric
    )
    
    print("\n" + "=" * 80)
    print("RSA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
