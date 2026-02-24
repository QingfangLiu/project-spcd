#!/usr/bin/env python3
"""
make_swarm.py

Creates a Biowulf swarm file to run run_rsa.py (no conda activation).

Example:
  python make_swarm_simple.py \
    --base_dir /data/liuq13 \
    --script_path /data/liuq13/SPCD_RSA_cuewise_subset/run_rsa.py \
    --sessions premeal postmeal \
    --glm_variant cuewise \
    --stat_type beta \
    --mode both \
    --models identity \
    --neural_metric crossnobis \
    --rsa_method spearman \
    --cue_subset all \
    --radius 8 \
    --n_jobs 4 \
    --use_resms_whitening \
    --out rsa.swarm
"""

from __future__ import annotations

import argparse
from pathlib import Path


def discover_subjects(base_dir: str, session: str, glm_variant: str) -> list[str]:
    glm_dir = Path(base_dir) / "GLM" / f"{session}_{glm_variant}_2mm"
    if not glm_dir.is_dir():
        raise FileNotFoundError(f"GLM folder not found: {glm_dir}")
    subs = sorted(p.name for p in glm_dir.iterdir() if p.is_dir() and p.name.startswith("sub-"))
    if not subs:
        raise RuntimeError(f"No subjects found under: {glm_dir}")
    return subs


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_dir", default="/data/liuq13")
    ap.add_argument("--script_path", default="RSA_probe_crossnobis.run_rsa") 
    ap.add_argument("--out", default="swarmfile.swarm")  

    ap.add_argument("--sessions", nargs="+", default=["premeal", "postmeal"])
    ap.add_argument("--glm_variant", default="cuewise", choices=["cuewise", "cuewiseLSS"])
    ap.add_argument("--stat_type", default="beta", choices=["beta", "tmap"])
    ap.add_argument("--mode", default="both")

    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--n_jobs", type=int, default=4)

    ap.add_argument("--rsa_method", default="spearman", choices=["regression", "spearman", "kendall"])
    ap.add_argument("--neural_metric", default="crossnobis", choices=["corr", "crossnobis"])
    ap.add_argument("--cue_subset", default="all", choices=["all", "conditioned", "preconditioned"])
    ap.add_argument("--models", nargs="+", default=["identity"])

    ap.add_argument("--use_odor_mask", action="store_true")
    ap.add_argument("--use_resms_whitening", action="store_true")
    ap.add_argument("--center_within_run", action="store_true")

    args = ap.parse_args()

    # Discover subs from the first session’s GLM folder
    subs = discover_subjects(args.base_dir, args.sessions[0], args.glm_variant)

    script_path = str(Path(args.script_path))

    with open(args.out, "w") as f:
        for sub in subs:
            for sess in args.sessions:
                cmd = [
                    "python", "-m", script_path,
                    "--sub", sub,
                    "--session", sess,
                    "--base_dir", args.base_dir,
                    "--glm_variant", args.glm_variant,
                    "--stat_type", args.stat_type,
                    "--mode", args.mode,
                    "--radius", str(args.radius),
                    "--n_jobs", str(args.n_jobs),
                    "--rsa_method", args.rsa_method,
                    "--neural_metric", args.neural_metric,
                    "--cue_subset", args.cue_subset,
                    "--models", *args.models,
                ]
                if args.use_odor_mask:
                    cmd.append("--use_odor_mask")
                if args.use_resms_whitening:
                    cmd.append("--use_resms_whitening")
                if args.center_within_run:
                    cmd.append("--center_within_run")

                f.write(" ".join(cmd) + "\n")

    print(f"Wrote {len(subs) * len(args.sessions)} lines.")


if __name__ == "__main__":
    main()