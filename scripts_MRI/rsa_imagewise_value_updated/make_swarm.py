#!/usr/bin/env python3
"""Adapt rsa_imagewise/make_swarm_imagewise.py for three combined-phase value runs.

Generate locally, copy the prepared subset to Biowulf, then submit this single
Swarm file. Each line calls the unchanged rsa_imagewise.run_rsa module.
"""
import argparse
import math
from pathlib import Path, PurePosixPath
import shlex
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts_MRI"))
from rsa_imagewise.make_swarm_imagewise import (
    default_ratings_file, load_subjects_with_odor_value_ratings,
)

SKIP_SUBS = {"sub-19", "sub-31", "sub-33"}
CUE_SUBSETS = ("all", "conditioned_only", "preconditioned_only")


def select_subjects(ratings_file, subjects=None):
    """Use the original ratings eligibility rule plus current paper exclusions."""
    eligible = load_subjects_with_odor_value_ratings(Path(ratings_file))
    table = pd.read_csv(ratings_file)
    if table["sub"].duplicated().any():
        raise ValueError("Duplicate participants in compiled ratings")
    rating_columns = [f"{p}_{g}" for p in ("pre", "post") for g in ("odor1", "odor2", "clean")]
    finite = table[rating_columns].apply(lambda col: col.map(lambda v: math.isfinite(float(v)))).all(axis=1)
    eligible &= set(table.loc[finite, "sub"].astype(str))
    candidates = sorted(set(subjects if subjects is not None else
        (p.name for p in (PROJECT / "beh_processed_subs").glob("sub-*") if p.is_dir())))
    if not candidates:
        raise ValueError("No candidates found; run from a full clone or provide --subjects")
    report = [dict(sub=sub, status=("excluded_paper_rule" if sub in SKIP_SUBS else
                  "excluded_missing_ratings" if sub not in eligible else "included"))
              for sub in candidates]
    included = [r["sub"] for r in report if r["status"] == "included"]
    if not included:
        raise ValueError("No participants with complete ratings after paper exclusions")
    return included, report


def generate(args):
    subjects, report = select_subjects(args.ratings_file, args.subjects)
    base = PurePosixPath(args.base_dir)
    if not base.is_absolute():
        raise ValueError("base_dir must be an absolute Biowulf path")
    maps = args.map_dir or str(base / "rsa_maps/probe_imagewise_value_updated/both")
    ratings_remote = str(base / "compiled_pleasantness_ratings.csv")
    prefix = [shlex.join(["source", str(PurePosixPath(args.conda_root) / "etc/profile.d/conda.sh")]),
              shlex.join(["conda", "activate", args.conda_env]),
              "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
              shlex.join(["cd", str(base)])]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        for sub in subjects:
            for subset in CUE_SUBSETS:
                command = ["python", "-B", "-m", "rsa_imagewise.run_rsa", "--sub", sub,
                    "--session", "both", "--base_dir", str(base),
                    "--ratings_file", ratings_remote, "--target", "odor_value",
                    "--cue_subset", subset, "--button_scope", "second_image_only",
                    "--radius", str(args.radius), "--n_jobs", str(args.n_jobs),
                    "--output_dir", str(maps)]
                stream.write(" && ".join(prefix + [shlex.join(command)]) + "\n")
    (args.out.parent / "subjects.txt").write_text("\n".join(subjects) + "\n")
    pd.DataFrame(report).to_csv(args.out.parent / "subject_selection.csv", index=False)
    remote_swarm = str(base / args.out.name)
    logs = str(base / "logs_value_rsa_updated")
    print(f"Wrote {len(subjects) * 3} commands for {len(subjects)} participants: {args.out}")
    print("All commands call the original rsa_imagewise.run_rsa; no analyses were started.")
    print("After copying this file to Biowulf, submit manually:")
    print(shlex.join(["mkdir", "-p", logs]))
    print(shlex.join(["swarm", "--no-comment", "-f", remote_swarm, "-g", str(args.mem_gb),
                     "-t", str(args.n_jobs), "--time", args.time, "--logdir", logs]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="/data/liuq13/SPCD_mini", help="Biowulf data/code root")
    parser.add_argument("--ratings_file", type=Path, default=default_ratings_file(), help="LOCAL compiled ratings")
    parser.add_argument("--subjects", nargs="+", help="Default: participant folders in the local clone")
    parser.add_argument("--out", type=Path, default=Path("value_rsa.swarm"))
    parser.add_argument("--map_dir", help="Fresh BIOWULF map directory; default: BASE/rsa_maps/probe_imagewise_value_updated/both")
    parser.add_argument("--radius", type=float, default=8.)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--conda_root", default="/data/liuq13/conda")
    parser.add_argument("--conda_env", default="/data/liuq13/conda/envs/project_rsa")
    parser.add_argument("--mem_gb", type=int, default=16, help="Printed allocation request, not measured usage")
    parser.add_argument("--time", default="24:00:00", help="Printed allocation request per command")
    args = parser.parse_args()
    if args.n_jobs < 1 or not math.isfinite(args.radius) or args.radius <= 0 or args.mem_gb < 1:
        parser.error("n_jobs, radius, and mem_gb must be positive; radius must be finite")
    generate(args)


if __name__ == "__main__":
    main()
