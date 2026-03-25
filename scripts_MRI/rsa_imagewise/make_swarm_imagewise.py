#!/usr/bin/env python3
"""
make_swarm_imagewise.py

Create a Biowulf swarm file for the current image-wise RSA pipeline.
"""

import argparse
from pathlib import Path


def discover_subjects(base_dir, session):
    glm_dir = Path(base_dir) / "GLM" / f"{session}_trialwise_2mm"
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
    ap.add_argument("--base_dir", default="/data/liuq13/SPCD_RSA_imagewise_subset")
    ap.add_argument("--package_name", default="rsa_imagewise")
    ap.add_argument("--out", default="imagewise_rsa.swarm")

    ap.add_argument("--sessions", nargs="+", default=["premeal", "postmeal"])
    ap.add_argument("--subjects", nargs="+", default=None)

    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument(
        "--button_scope",
        default="both_images",
        choices=["both_images", "second_image_only"],
    )
    ap.add_argument("--center_within_run", action="store_true")

    args = ap.parse_args()

    if args.subjects is None:
        subs = discover_subjects(args.base_dir, args.sessions[0])
    else:
        subs = sorted(args.subjects)

    package_parent = Path(args.base_dir)
    module_name = f"{args.package_name}.run_rsa"

    n_lines = 0
    with open(args.out, "w") as f:
        for sub in subs:
            for sess in args.sessions:
                cmd = [
                    f"python",
                    "-m",
                    module_name,
                    "--sub", sub,
                    "--session", sess,
                    "--base_dir", args.base_dir,
                    "--radius", str(args.radius),
                    "--n_jobs", str(args.n_jobs),
                    "--button_scope", args.button_scope,
                ]
                if args.center_within_run:
                    cmd.append("--center_within_run")

                f.write(" ".join(cmd) + "\n")
                n_lines += 1

    print(f"Wrote {n_lines} swarm lines to: {args.out}")


if __name__ == "__main__":
    main()

    