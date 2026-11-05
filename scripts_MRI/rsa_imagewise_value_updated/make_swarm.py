#!/usr/bin/env python3
"""Prepare one Biowulf Swarm file for the updated value searchlights.

Run from the repository clone on Biowulf. This calls the existing preparation
command, then adds the established Conda setup and thread caps to every job.
It prints a submission command but never submits jobs or runs searchlights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def environment_prefix(conda_sh, conda_env):
    return " && ".join([
        shlex.join(["source", str(conda_sh)]),
        shlex.join(["conda", "activate", str(conda_env)]),
        "export " + " ".join(f"{name}=1" for name in THREAD_VARIABLES),
    ])


def generate(args):
    base = args.base_dir.expanduser().resolve()
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              base / "rsa_maps/probe_imagewise_value_updated_v1")
    conda_sh = args.conda_root.expanduser().resolve() / "etc/profile.d/conda.sh"
    conda_env = args.conda_env.expanduser().resolve()
    python = conda_env / "bin/python"
    if not conda_sh.is_file():
        raise FileNotFoundError(f"Conda initialization script not found: {conda_sh}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Executable Python not found in Conda environment: {python}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}")

    runner = HERE / "run_searchlights.py"
    prefix = environment_prefix(conda_sh, conda_env)
    # Check the target environment, even when this generator is started with a
    # different Python. Preparation then records the target interpreter in jobs.
    imports = (
        "import sys; assert sys.version_info >= (3, 10), 'Python 3.10+ required'; "
        "import numpy, pandas, scipy, nibabel, sklearn, nilearn; "
        "print('RSA environment:', sys.executable, flush=True)"
    )
    check = shlex.join([str(python), "-B", "-c", imports])
    prepare = [str(python), "-B", str(runner), "prepare", "--base_dir", str(base),
               "--output_dir", str(output), "--radius", str(args.radius),
               "--n_jobs", str(args.n_jobs)]
    if args.ratings_file:
        prepare.extend(["--ratings_file", str(args.ratings_file.expanduser().resolve())])
    if args.subjects:
        prepare.extend(["--subjects", *args.subjects])
    subprocess.run(["bash", "-c", f"{prefix} && {check} && exec {shlex.join(prepare)}"],
                   check=True)

    swarm = output / "jobs.swarm"
    commands = swarm.read_text().splitlines()
    config_path = output / "analysis_config.json"
    config = json.loads(config_path.read_text())
    if len(commands) != len(config["subjects"]) or not commands:
        raise ValueError("Prepared Swarm commands do not match the eligible cohort")
    temporary = output / "jobs.swarm.tmp"
    temporary.write_text("\n".join(f"{prefix} && {command}" for command in commands) + "\n")
    temporary.replace(swarm)
    logs = output / "swarm_logs"
    logs.mkdir()
    submit = shlex.join(["swarm", "-f", str(swarm), "-g", f"{args.mem_gb:g}",
                         "-t", str(config["n_jobs"]), "--time", args.time,
                         "--logdir", str(logs)])
    metadata = dict(
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        conda_sh=str(conda_sh), conda_env=str(conda_env),
        thread_caps={name: 1 for name in THREAD_VARIABLES},
        n_subjects=len(config["subjects"]), n_jobs=config["n_jobs"],
        memory_gb_per_job=args.mem_gb, walltime_per_job=args.time,
        swarm_file=str(swarm), swarm_sha256=hashlib.sha256(swarm.read_bytes()).hexdigest(),
        log_dir=str(logs), submission_command=submit, submitted=False,
    )
    (output / "submission_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nWrote one Swarm file: {swarm}")
    print(f"{len(commands)} participant jobs; each runs all three cue analyses.")
    print("Memory and walltime are requested allocations, not measured requirements.")
    print("Submit manually with:\n" + submit)
    return swarm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", type=Path, default=Path("/data/liuq13"),
                        help="Parent of GLM/, nii/, and masks/; default: /data/liuq13")
    parser.add_argument("--output_dir", type=Path,
                        help="Fresh output directory; default: BASE/rsa_maps/probe_imagewise_value_updated_v1")
    parser.add_argument("--conda_root", type=Path, default=Path("/data/liuq13/conda"))
    parser.add_argument("--conda_env", type=Path,
                        default=Path("/data/liuq13/conda/envs/project_rsa"))
    parser.add_argument("--ratings_file", type=Path,
                        help="Optional compiled ratings; default: current clone's participant ratings")
    parser.add_argument("--subjects", nargs="+", help="Optional candidate list; default: all repository participants")
    parser.add_argument("--n_jobs", type=int, default=4, help="Workers and Swarm -t allocation per job")
    parser.add_argument("--radius", type=float, default=8.)
    parser.add_argument("--mem_gb", type=float, default=16., help="Requested GB per job, printed as Swarm -g")
    parser.add_argument("--time", default="24:00:00", help="Requested walltime, HH:MM:SS or D-HH:MM:SS")
    args = parser.parse_args()
    if args.n_jobs < 1 or not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("n_jobs must be at least 1; radius must be positive and finite")
    if not math.isfinite(args.mem_gb) or args.mem_gb <= 0:
        parser.error("mem_gb must be positive and finite")
    if not re.fullmatch(r"(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d", args.time) or not any(
            int(n) for n in re.split(r"[-:]", args.time)):
        parser.error("time must be a positive HH:MM:SS or D-HH:MM:SS duration")
    generate(args)


if __name__ == "__main__":
    main()
