#!/usr/bin/env python3
"""Generate Biowulf jobs for first-image-only odor-to-probe searchlights."""

import argparse
import os


def generate_swarm_file(
    base_dir,
    script_name="run_searchlight_crosstask_odor2probe_first_image.py",
    output_file="decode_crosstask_odor2probe_first_image.swarm",
    cue_types=("AB",),
    radius_mm=8.0,
    n_jobs=4,
    session_folder="premeal_trialwise_2mm",
    session_tag="premeal",
    odor_glm_folder="odor_scan_trialwise_2mm",
):
    """Write one AB, three-class first-image searchlight job per subject."""
    if n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer for Biowulf")

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
            for cue in cue_types:
                command = (
                    f"python {script_name} "
                    f"--sub {sub} "
                    f"--base_dir {base_dir} "
                    f"--mask_img {os.path.join(base_dir, 'masks', 'gm_0.1.nii')} "
                    f"--cue_type {cue} "
                    f"--n_classes 3 "
                    f"--radius_mm {radius_mm} "
                    f"--n_jobs {n_jobs} "
                    f"--session_folder {session_folder} "
                    f"--session_tag {session_tag}"
                )
                swarm_file.write(command + "\n")

    total_jobs = len(subjects) * len(cue_types)
    print(f"Swarm file saved: {output_file}")
    print(f"{total_jobs} jobs for {len(subjects)} subjects")


def main():
    parser = argparse.ArgumentParser(
        description="Generate the first-image-only cross-task searchlight swarm."
    )
    parser.add_argument("--base_dir", default="/data/liuq13/SPCD_mini")
    parser.add_argument(
        "--script_name",
        default="run_searchlight_crosstask_odor2probe_first_image.py",
    )
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
    args = parser.parse_args()
    generate_swarm_file(
        base_dir=args.base_dir,
        script_name=args.script_name,
        output_file=args.output_file,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
