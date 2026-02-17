#!/usr/bin/env python3
"""
Run searchlight RSA for one subject (cuewise GLM; cross-run RSA).

Usage:
    python -m RSA_probe_crossnobis.run_rsa \
         --sub sub-01 \
         --session premeal \
         --base_dir ~/SPCD_data \
         --stat_type beta \
         --glm_variant cuewise \
         --neural_metric crossnobis \
         --mode both

The way it works:
- Locate a package named RSA_probe_x_runs
- Inside that package, locate a module named run_rsa
- Execute that module 
- Need RSA_probe_x_runs/__init__.py exists (so it’s a package)

Key changes from the old version:
1) Supports stat_type: beta vs tmap (t maps use "tmap_" prefix and CSV column "tmap_file")
2) Does NOT average betas across runs. Uses per-run cuewise maps.
3) Neural similarity is computed across runs only (no within-run correlations), so at least 2 runs
   with data must exist if crossrun_only is enabled.
"""

import argparse
import os
import sys

from rsa_core_crossnobis.searchlight import searchlight_rsa_single_subject
from rsa_core_crossnobis.save_maps import save_rsa_maps
from rsa_core_crossnobis.models import (
    create_identity_model_rdm,
    create_value_model_rdm,
    create_association_model_rdm,
)

def main():
    parser = argparse.ArgumentParser(
        description="Run searchlight RSA for probe task (cuewise GLM; cross-run RSA)"
    )
    parser.add_argument(
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
        help="Base directory containing GLM folder (expects <base_dir>/GLM/...)"
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
        help="Output directory (default constructed from base_dir/session/stat_type/cue_subset)"
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
        choices=["corr", "crossnobis"],
        default="crossnobis",
        help="Neural RDM metric: corr (cross-run Pearson similarity) or crossnobis (distance)."
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
        nargs="+",
        choices=["association", "identity", "value"],
        default=["identity"],
        help="Which model(s) to run (default: identity). Example: --models identity value"
    )
    parser.add_argument(
        "--use_pair_mask",
        action="store_true",
        help="Use odor-only pair mask (exclude clean-involving pairs)."
    )
    parser.add_argument(
        "--stat_type",
        type=str,
        choices=["beta", "tmap"],
        default="beta",
        help="Which input maps to use from cuewise GLM (default: beta)."
    )
    parser.add_argument(
        "--glm_variant",
        type=str,
        choices=["cuewise", "cuewiseLSS"],
        default="cuewise",
        help="Which GLM variant to read maps from: cuewise (GLM/<task>_cuewise_2mm) or cuewiseLSS (GLM/<task>_cuewiseLSS_2mm)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        help="Cue GLM mode folder under subject (default: both)."
    )
    parser.add_argument(
        "--use_resms_whitening",
        action="store_true",
        help="Apply run-specific ResMS whitening before crossnobis.",
    )
    parser.add_argument(
        "--whitening_eps",
        type=float,
        default=1e-6,
        help="Small epsilon added to ResMS before inversion.",
    )
    
    args = parser.parse_args()

    if args.mask is None: # default mask
        args.mask = os.path.join(args.base_dir, "masks", "gm_0.1.nii")

    # Determine session folder (cuewise GLM)
    session_folder = f"{args.session}_{args.glm_variant}_2mm"

    # Default output dir
    if args.output_dir is None:
        # separate cuewise + stat_type
        rsa_root = "probe_cuewise_LSS" if args.glm_variant == "cuewiseLSS" else "probe_cuewise"
        base_out = os.path.join(args.base_dir, "rsa_maps", rsa_root, args.session, f"input-{args.stat_type}")
        if args.cue_subset == "all":
            args.output_dir = base_out
        else:
            args.output_dir = os.path.join(base_out, f"subset-{args.cue_subset}")

    # Checks
    if not os.path.exists(args.mask):
        print(f"ERROR: Mask file not found: {args.mask}")
        sys.exit(1)

    print("=" * 80)
    print("RSA Searchlight Analysis (cuewise GLM; cross-run RSA)")
    print("=" * 80)
    print(f"Subject:         {args.sub}")
    print(f"Session:         {args.session}")
    print(f"Base dir:        {args.base_dir}")
    print(f"Session folder:  {session_folder}")
    print(f"Mode:            {args.mode}")
    print(f"Stat type:       {args.stat_type}")
    print(f"Mask:            {args.mask}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Radius:          {args.radius} mm")
    print(f"N jobs:          {args.n_jobs}")
    print(f"RSA method:      {args.rsa_method}")
    print(f"cue_subset:      {args.cue_subset}")
    print(f"models:          {args.models}")
    print(f"use_pair_mask:   {args.use_pair_mask}")
    print(f"use_resms_whitening: {args.use_resms_whitening}")
    print(f"whitening_eps: {args.whitening_eps}")
    print("=" * 80)

    # Create model RDMs
    print("\nCreating model RDMs...")
    all_model_rdms = {
        "association": create_association_model_rdm(),
        "identity": create_identity_model_rdm(),
        "value": create_value_model_rdm(),
    }
    model_rdms = {k: all_model_rdms[k] for k in args.models}

    from rsa_core_crossnobis.models import convert_model_rdms_to_distance
    if args.neural_metric == "crossnobis":
        model_rdms = convert_model_rdms_to_distance(model_rdms, force_diag_zero=True)

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
            mode=args.mode,
            stat_type=args.stat_type,
            use_resms_whitening=args.use_resms_whitening,
            whitening_eps=args.whitening_eps,
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
        session_tag=args.session,
        radius=args.radius,
        rsa_method=args.rsa_method,
        cue_subset=args.cue_subset,
        use_pair_mask=args.use_pair_mask,
        mode=args.mode,
        stat_type=args.stat_type,
    )

    print("\n" + "=" * 80)
    print("RSA analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
