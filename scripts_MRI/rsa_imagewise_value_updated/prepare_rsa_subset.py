#!/usr/bin/env python3
"""Stage code, ratings, and probe events locally using the old subset helpers.

Betas and the gray-matter mask already on Biowulf are reused. Optional flags
copy them too when needed. This script never connects to Biowulf.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts_MRI"))
from rsa_imagewise.prepare_rsa_subset import (
    PACKAGE_FILES, copy_file, copy_package_files, copy_events_for_subject,
    copy_glm_subject_folder,
)
from rsa_imagewise.make_swarm_imagewise import default_ratings_file
from make_swarm import select_subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_data_dir", type=Path, default=Path("/Users/liuq13/SPCD_data"))
    parser.add_argument("--subset_dir", type=Path, required=True, help="Local staging folder to rsync into SPCD_mini")
    parser.add_argument("--ratings_file", type=Path, default=default_ratings_file())
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--copy_betas", action="store_true", help="Also stage pre/post betas and index CSVs")
    parser.add_argument("--copy_mask", action="store_true", help="Also stage masks/gm_0.1.nii")
    args = parser.parse_args()
    subjects, report = select_subjects(args.ratings_file, args.subjects)
    source = HERE.parent / "rsa_imagewise"
    for name in [*PACKAGE_FILES, "ratings.py"]:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination = args.subset_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    copy_package_files(source, destination)
    # The old copier omits this dependency; the value runner needs it.
    copy_file(source / "ratings.py", destination / "rsa_imagewise/ratings.py")
    copy_file(args.ratings_file, destination / "compiled_pleasantness_ratings.csv")
    copy_file(HERE / "aggregate_rsa_group.m", destination / "aggregate_rsa_group.m")
    (destination / "subjects.txt").write_text("\n".join(subjects) + "\n")
    pd.DataFrame(report).to_csv(destination / "subject_selection.csv", index=False)
    missing, n_events = [], 0
    for sub in subjects:
        for phase in ("premeal", "postmeal"):
            events = args.full_data_dir / "nii" / sub / "func_PAID" / phase
            for run in (1, 2, 3):
                path = events / f"{sub}_task-{phase}_run-{run}_events.tsv"
                if not path.is_file():
                    missing.append(dict(sub=sub, phase=phase, run=run, path=str(path)))
            if events.is_dir():
                n_events += copy_events_for_subject(args.full_data_dir, destination, sub, phase)
            if args.copy_betas:
                copy_glm_subject_folder(args.full_data_dir, destination, sub, phase)
    if args.copy_mask:
        mask = args.full_data_dir / "masks/gm_0.1.nii"
        if not mask.is_file():
            raise FileNotFoundError(mask)
        copy_file(mask, destination / "masks/gm_0.1.nii")
    pd.DataFrame(missing, columns=["sub", "phase", "run", "path"]).to_csv(
        destination / "missing_events.csv", index=False)
    print(f"Prepared {len(subjects)} participants; copied {n_events} event files: {destination}")
    print(f"Missing required event files: {len(missing)}; see missing_events.csv")
    print("Betas copied:", args.copy_betas, "| GM mask copied:", args.copy_mask)
    print("Next: generate value_rsa.swarm into this folder, then rsync its contents into SPCD_mini.")


if __name__ == "__main__":
    main()
