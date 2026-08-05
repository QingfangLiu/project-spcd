#!/usr/bin/env python3
"""Generate Biowulf jobs for binary first-image odor-to-probe decoding."""

import argparse
import os


CUE_SUBSETS = (
    "conditioned_odor_only",
    "preconditioned_odor_only",
)
ANALYSIS_FOLDER = "Decoding_probe_cross_task"
RUNNER_FILENAME = "run_searchlight_crosstask_odor2probe_first_image.py"


def generate_swarm_file(
    base_dir,
    output_file="decode_crosstask_odor2probe_first_image.swarm",
    cue_subsets=CUE_SUBSETS,
    radius_mm=8.0,
    n_jobs=4,
    session_folder="premeal_trialwise_2mm",
    session_tag="premeal",
    odor_glm_folder="odor_scan_trialwise_2mm",
    sub14_original_events_dir=None,
):
    """Write conditioned and preconditioned binary jobs for every subject."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer for Biowulf")
    invalid = sorted(set(cue_subsets) - set(CUE_SUBSETS))
    if invalid:
        raise ValueError(f"Unknown cue subsets: {invalid}")

    base_dir = os.path.abspath(base_dir)
    script_path = os.path.join(base_dir, ANALYSIS_FOLDER, RUNNER_FILENAME)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Searchlight runner not found: {script_path}")

    glm_probe = os.path.join(base_dir, "GLM", session_folder)
    glm_odor = os.path.join(base_dir, "GLM", odor_glm_folder)
    subs_probe = {
        name
        for name in os.listdir(glm_probe)
        if name.startswith("sub-") and os.path.isdir(os.path.join(glm_probe, name))
    }
    subs_odor = {
        name
        for name in os.listdir(glm_odor)
        if name.startswith("sub-") and os.path.isdir(os.path.join(glm_odor, name))
    }
    subjects = sorted(subs_probe & subs_odor)

    with open(output_file, "w") as swarm_file:
        for sub in subjects:
            for cue_subset in cue_subsets:
                command = (
                    f"python {script_path} "
                    f"--sub {sub} "
                    f"--base_dir {base_dir} "
                    f"--mask_img {os.path.join(base_dir, 'masks', 'gm_0.1.nii')} "
                    f"--cue_subset {cue_subset} "
                    f"--radius_mm {radius_mm} "
                    f"--n_jobs {n_jobs} "
                    f"--session_folder {session_folder} "
                    f"--session_tag {session_tag}"
                )
                if sub == "sub-14" and sub14_original_events_dir:
                    command += (
                        " --sub14_original_events_dir "
                        f"{sub14_original_events_dir}"
                    )
                swarm_file.write(command + "\n")

    total_jobs = len(subjects) * len(cue_subsets)
    print(f"Swarm file saved: {output_file}")
    print(
        f"{total_jobs} binary jobs for {len(subjects)} subjects "
        f"x {len(cue_subsets)} cue subsets"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate conditioned/preconditioned binary searchlight jobs."
    )
    parser.add_argument("--base_dir", default="/data/liuq13/SPCD_mini")
    parser.add_argument(
        "--output_file",
        default="decode_crosstask_odor2probe_first_image.swarm",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="CPUs used by each searchlight job; match this to swarm -t.",
    )
    parser.add_argument(
        "--sub14_original_events_dir",
        help=(
            "Optional full-data directory containing sub-14 premeal "
            "*_events_orig.tsv files."
        ),
    )
    args = parser.parse_args()
    generate_swarm_file(
        base_dir=args.base_dir,
        output_file=args.output_file,
        n_jobs=args.n_jobs,
        sub14_original_events_dir=args.sub14_original_events_dir,
    )


if __name__ == "__main__":
    main()
