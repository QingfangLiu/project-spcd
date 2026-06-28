#!/usr/bin/env python3
"""
make_swarm_imagewise.py

Create a Biowulf swarm file for the first-image-only run-pair RSA pipeline.
"""

import argparse
from pathlib import Path

import pandas as pd

MODEL_NAMES = (
    "odor_value",
    "cue_type",
    "image_identity",
)

RUN_PAIRS = (
    "run1_vs_run2",
    "run1_vs_run3",
    "run2_vs_run3",
)


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


def default_ratings_file() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "beh_processed_subs" / "compiled_pleasantness_ratings.csv"


def load_subjects_with_odor_value_ratings(ratings_file: Path) -> set[str]:
    if not ratings_file.is_file():
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

    df = pd.read_csv(ratings_file)
    required_cols = {
        "sub",
        "pre_odor1",
        "pre_odor2",
        "pre_clean",
        "post_odor1",
        "post_odor2",
        "post_clean",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Ratings file is missing required columns: {sorted(missing_cols)}"
        )

    complete_mask = df[list(required_cols - {"sub"})].notna().all(axis=1)
    return set(df.loc[complete_mask, "sub"].astype(str))


def available_targets_for_session(session: str, cue_subset: str) -> list[str]:
    available_targets = list(MODEL_NAMES)
    if cue_subset in ("conditioned_only", "preconditioned_only"):
        available_targets = [t for t in available_targets if t != "cue_type"]
    return available_targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/data/liuq13/SPCD_RSA_imagewise_first_image_run_pairs_subset")
    ap.add_argument("--package_name", default="rsa_imagewise_first_image_run_pairs")
    ap.add_argument("--out", default="imagewise_first_image_run_pairs_rsa.swarm")
    ap.add_argument(
        "--ratings_file",
        type=Path,
        default=default_ratings_file(),
        help="Compiled pleasantness ratings CSV to pass to RSA jobs for the odor_value model",
    )

    ap.add_argument(
        "--sessions",
        nargs="+",
        choices=["premeal", "postmeal"],
        default=["premeal", "postmeal"],
    )
    ap.add_argument(
        "--run_pairs",
        nargs="+",
        choices=RUN_PAIRS,
        default=list(RUN_PAIRS),
        help="Run-pair RSA modes to generate",
    )
    ap.add_argument("--subjects", nargs="+", default=None)

    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument(
        "--targets",
        nargs="+",
        choices=MODEL_NAMES,
        default=None,
        help="If provided, write one swarm line per target regressor",
    )
    ap.add_argument(
        "--split_targets",
        action="store_true",
        help="When --targets is omitted, write one swarm line per available target instead of one line that runs all targets",
    )
    ap.add_argument(
        "--cue_subset",
        default="all",
        choices=["all", "conditioned_only", "preconditioned_only"],
        help="Restrict RSA samples to all cues, conditioned cues (B/D), or preconditioned cues (A/C)",
    )
    ap.add_argument("--center_within_run", action="store_true")

    args = ap.parse_args()

    if args.subjects is None:
        subs = discover_subjects(args.base_dir, args.sessions[0])
    else:
        subs = sorted(args.subjects)

    module_name = f"{args.package_name}.run_rsa"
    odor_value_subjects = None

    needs_odor_value = (
        args.targets is None or "odor_value" in args.targets
    )
    if needs_odor_value:
        odor_value_subjects = load_subjects_with_odor_value_ratings(args.ratings_file)

    n_lines = 0
    with open(args.out, "w") as f:
        for sub in subs:
            for sess in args.sessions:
                available_targets = available_targets_for_session(
                    session=sess,
                    cue_subset=args.cue_subset,
                )

                if args.targets is not None:
                    invalid_targets = sorted(set(args.targets) - set(available_targets))
                    if invalid_targets:
                        raise ValueError(
                            f"Invalid target(s) for session={sess}, "
                            f"cue_subset={args.cue_subset}: {invalid_targets}. "
                            f"Available targets: {available_targets}"
                        )

                if args.targets is not None:
                    job_targets = args.targets
                elif args.split_targets:
                    job_targets = available_targets
                else:
                    job_targets = [None]
                for run_pair in args.run_pairs:
                    for target in job_targets:
                        if (
                            target == "odor_value"
                            and odor_value_subjects is not None
                            and sub not in odor_value_subjects
                        ):
                            print(
                                f"Skipping {sub} {sess} {run_pair} target=odor_value "
                                f"(missing compiled ratings in {args.ratings_file})"
                            )
                            continue

                        cmd = [
                            f"python",
                            "-m",
                            module_name,
                            "--sub", sub,
                            "--session", sess,
                            "--run_pair", run_pair,
                            "--base_dir", args.base_dir,
                            "--radius", str(args.radius),
                            "--n_jobs", str(args.n_jobs),
                            "--cue_subset", args.cue_subset,
                            "--ratings_file", str(args.ratings_file),
                        ]
                        if target is not None:
                            cmd.extend(["--target", target])
                        if args.center_within_run:
                            cmd.append("--center_within_run")

                        f.write(" ".join(cmd) + "\n")
                        n_lines += 1

    print(f"Wrote {n_lines} swarm lines to: {args.out}")


if __name__ == "__main__":
    main()

    
