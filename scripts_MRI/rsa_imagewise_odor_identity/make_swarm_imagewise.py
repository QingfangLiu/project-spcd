#!/usr/bin/env python3
"""
make_swarm_imagewise.py

Create a Biowulf swarm file for the premeal odor-identity RSA variant.
"""

import argparse
from pathlib import Path

MODEL_NAMES = (
    "odor_identity",
    "image_identity",
    "button_presence",
    "button_side",
)

CUE_SUBSETS = (
    "conditioned_odor_only",
    "preconditioned_odor_only",
)


def discover_subjects(base_dir):
    glm_dir = Path(base_dir) / "GLM" / "premeal_trialwise_2mm"
    if not glm_dir.is_dir():
        raise FileNotFoundError(f"GLM folder not found: {glm_dir}")
    subs = sorted(
        p.name for p in glm_dir.iterdir()
        if p.is_dir() and p.name.startswith("sub-")
    )
    if not subs:
        raise RuntimeError(f"No subjects found under: {glm_dir}")
    return subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/data/liuq13/SPCD_RSA_imagewise_odor_identity_subset")
    ap.add_argument("--package_name", default="rsa_imagewise_odor_identity")
    ap.add_argument("--out", default="imagewise_odor_identity_premeal.swarm")

    ap.add_argument("--subjects", nargs="+", default=None)
    ap.add_argument(
        "--cue_subsets",
        nargs="+",
        choices=CUE_SUBSETS,
        default=list(CUE_SUBSETS),
        help="Cue subsets to run; both exclude clean-air cues",
    )

    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument(
        "--button_scope",
        default="second_image_only",
        choices=["both_images", "second_image_only"],
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="If omitted, write odor_identity jobs only",
    )
    ap.add_argument(
        "--split_targets",
        action="store_true",
        help="When --targets is omitted, write one line per available target",
    )
    ap.add_argument("--center_within_run", action="store_true")

    args = ap.parse_args()

    if args.subjects is None:
        subs = discover_subjects(args.base_dir)
    else:
        subs = sorted(args.subjects)

    module_name = f"{args.package_name}.run_rsa"

    if args.targets is not None:
        job_targets = args.targets
    elif args.split_targets:
        job_targets = list(MODEL_NAMES)
    else:
        job_targets = ["odor_identity"]

    n_lines = 0
    with open(args.out, "w") as f:
        for sub in subs:
            for cue_subset in args.cue_subsets:
                for target in job_targets:
                    cmd = [
                        "python",
                        "-m",
                        module_name,
                        "--sub", sub,
                        "--session", "premeal",
                        "--base_dir", args.base_dir,
                        "--radius", str(args.radius),
                        "--n_jobs", str(args.n_jobs),
                        "--button_scope", args.button_scope,
                        "--cue_subset", cue_subset,
                        "--target", target,
                    ]
                    if args.center_within_run:
                        cmd.append("--center_within_run")

                    f.write(" ".join(cmd) + "\n")
                    n_lines += 1

    print(f"Wrote {n_lines} swarm lines to: {args.out}")


if __name__ == "__main__":
    main()
