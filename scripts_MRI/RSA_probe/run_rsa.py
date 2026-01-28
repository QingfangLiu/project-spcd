#!/usr/bin/env python3
"""
Run searchlight RSA for one subject (full 12-condition analysis).

This module is intended to be executed with Python's `-m` flag from the project root directory.

Usage:
    python -m RSA_probe.run_rsa \
         --sub sub-01 \
         --session premeal \
         --base_dir ~/SPCD_data

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

from rsa_core.searchlight import searchlight_rsa_single_subject
from rsa_core.save_maps import save_rsa_maps

from rsa_core.models import (create_identity_model_rdm, 
                             create_value_model_rdm, 
                             create_association_model_rdm
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
        default="spearman",
        help="RSA scoring method (default: spearman)"
    )
    parser.add_argument(
        "--neural_metric",
        type=str,
        choices=["correlation", "euclidean"],
        default="correlation",
        help="Neural similarity metric (default: correlation)"
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
    parser.add_argument(
        "--use_pair_mask",
        action="store_true",
        help="Use odor-only pair mask (exclude clean-involving pairs)."
    )

    args = parser.parse_args() # reads sys.argv, matches values to rules, convers types, returns a simple object
    
    # Set defaults
    if args.mask is None:
        args.mask = os.path.join(args.base_dir, "masks", "gm_0.1.nii")
    
    if args.output_dir is None:
        if args.cue_subset == "all":
            args.output_dir = os.path.join(args.base_dir, "rsa_maps", "probe", args.session)
        else:
            args.output_dir = os.path.join(args.base_dir, "rsa_maps", "probe_subset", args.session)

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
    print(f"cue_subset:     {args.cue_subset}")
    print(f"models:         {args.models}")
    print(f"use_pair_mask:  {args.use_pair_mask}")
    print("=" * 80)
    
    # Create model RDMs
    print("\nCreating model RDMs...")

    all_model_rdms = {
        'association': create_association_model_rdm(),
        'identity': create_identity_model_rdm(),
        'value': create_value_model_rdm(),
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
            cue_subset=args.cue_subset, 
            use_pair_mask=args.use_pair_mask,
        )
    except Exception as e:
        print(f"\nERROR during searchlight: {e}")
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
        cue_subset=args.cue_subset, 
        use_pair_mask=args.use_pair_mask,
    )
    
    print("\n" + "=" * 80)
    print("RSA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
