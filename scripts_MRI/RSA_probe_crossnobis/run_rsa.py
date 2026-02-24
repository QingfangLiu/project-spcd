
"""
Command-line interface for searchlight RSA.

Configurable:
- subject
- session
- GLM variant
- stat type
- radius
- RSA method
- neural metric
- cue subset
- model(s)
- odor masking
- ResMS whitening
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
        "--use_odor_mask",
        action="store_true",
        help="Use odor-only mask (exclude clean-involving pairs)."
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
    
    args = parser.parse_args()

    # Determine session folder (cuewise GLM)
    session_folder = f"{args.session}_{args.glm_variant}_2mm"

    # Default output dir
    if args.output_dir is None:
        rsa_root = "probe_cuewise_LSS" if args.glm_variant == "cuewiseLSS" else "probe_cuewise"
        args.output_dir = os.path.join(args.base_dir, "rsa_maps", rsa_root, args.session, f"input-{args.stat_type}")

    print("=" * 80)
    print("RSA Searchlight Analysis (cuewise GLM)")
    print("=" * 80)
    print(f"Subject:         {args.sub}")
    print(f"Session:         {args.session}")
    print(f"Base dir:        {args.base_dir}")
    print(f"Session folder:  {session_folder}")
    print(f"Mode:            {args.mode}")
    print(f"Stat type:       {args.stat_type}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Radius:          {args.radius} mm")
    print(f"N jobs:          {args.n_jobs}")
    print(f"RSA method:      {args.rsa_method}")
    print(f"cue_subset:      {args.cue_subset}")
    print(f"models:          {args.models}")
    print(f"use_odor_mask:   {args.use_odor_mask}")
    print(f"use_resms_whitening: {args.use_resms_whitening}")
    print("=" * 80)

    # Create model RDMs
    print("\nCreating model RDMs...")
    all_model_rdms = {
        "association": create_association_model_rdm(),
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
            model_rdms=model_rdms,
            radius=args.radius,
            n_jobs=args.n_jobs,
            rsa_method=args.rsa_method,
            neural_metric=args.neural_metric,
            cue_subset=args.cue_subset,
            use_odor_mask=args.use_odor_mask,
            mode=args.mode,
            stat_type=args.stat_type,
            use_resms_whitening=args.use_resms_whitening,
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
        cue_subset=args.cue_subset,
        mode=args.mode,
        rsa_method=args.rsa_method,
        neural_metric=args.neural_metric,
        use_odor_mask=args.use_odor_mask,
        use_resms_whitening=args.use_resms_whitening,
    )

    print("\n" + "=" * 80)
    print("RSA analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
    