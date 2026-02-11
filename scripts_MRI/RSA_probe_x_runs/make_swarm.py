#!/usr/bin/env python3
"""
Generate swarm file for cuewise GLM cross-run RSA on Biowulf.

Usage:
    python make_swarm.py

Then submit with (example):
    swarm -f <swarm_file> -g 8 -t 4
"""

import os
from typing import Iterable, List, Optional, Tuple, Union


def _ensure_models_list(models: Union[str, Iterable[str]]) -> List[str]:
    """
    Convert models input into a list of model names compatible with
    run_rsa.py's '--models' nargs+ interface.
    """
    if isinstance(models, str):
        # allow either "identity value" or "identity"
        parts = [p.strip() for p in models.split() if p.strip()]
        if len(parts) == 0:
            raise ValueError("Empty models string provided.")
        return parts
    parts = [str(m).strip() for m in models if str(m).strip()]
    if len(parts) == 0:
        raise ValueError("Empty models iterable provided.")
    return parts


def generate_swarm_file(
    base_dir: str = "/data/liuq13",
    module_name: str = "RSA_probe_x_runs.run_rsa",
    output_file: Optional[str] = None,
    sessions: Tuple[str, ...] = ("premeal", "postmeal"),
    radius_mm: float = 8.0,
    n_jobs: int = 4,  # cores per swarm job (match with swarm -t value; do NOT use -1 on Biowulf)
    rsa_method: str = "spearman",
    mask_path: Optional[str] = None,
    cue_subset: str = "all",
    models: Union[str, Iterable[str]] = ("identity",),
    use_pair_mask: bool = False,
    stat_type: str = "tmap",       # "beta" or "tmap"
    mode: str = "both",            # "both", "img1only", "img2only"
    runs: str = "1,2,3",           # comma-separated runs to attempt
    glm_variant: str = "cuewiseLSS",  # "cuewise" or "cuewiseLSS" (matches run_rsa.py)
    session_folder_template: str = "{session}_{glm_variant}_2mm",  # for subject discovery
):
    """
    Generate a swarm file for cuewise GLM cross-run searchlight RSA.

    Notes
    -----
    - run_rsa.py expects:
        --models <one-or-more tokens>   (nargs="+")
        --glm_variant {cuewise,cuewiseLSS}
      so this script writes the swarm lines accordingly.
    """
    # Defaults
    if mask_path is None:
        mask_path = os.path.join(base_dir, "masks", "gm_0.1.nii")

    models_list = _ensure_models_list(models)

    # Auto-generate output filename if not provided
    if output_file is None:
        mask_tag = "_mask-odor" if use_pair_mask else ""
        models_tag = "-".join(models_list)  # ["identity","value"] -> "identity-value"
        output_file = (
            f"rsa_{glm_variant}_{stat_type}_{rsa_method}_cue_subset_{cue_subset}_"
            f"{models_tag}{mask_tag}_mode_{mode}.swarm"
        )

    # Discover subjects from one session folder (based on glm_variant)
    discover_folder = session_folder_template.format(session=sessions[0], glm_variant=glm_variant)
    glm_path = os.path.join(base_dir, "GLM", discover_folder)

    if not os.path.isdir(glm_path):
        raise FileNotFoundError(f"GLM folder not found: {glm_path}")

    sub_list = sorted([d for d in os.listdir(glm_path) if d.startswith("sub-")])
    if len(sub_list) == 0:
        raise RuntimeError(f"No subjects found in: {glm_path}")

    # Write swarm file
    with open(output_file, "w") as f:
        for sub in sub_list:
            for session in sessions:
                cmd_parts = [
                    "python", "-m", module_name,
                    "--sub", sub,
                    "--session", session,
                    "--base_dir", base_dir,
                    "--mask", mask_path,
                    "--radius", str(radius_mm),
                    "--n_jobs", str(n_jobs),
                    "--rsa_method", rsa_method,
                    "--cue_subset", cue_subset,
                    "--models", *models_list,            # <-- nargs+ tokens
                    "--stat_type", stat_type,
                    "--glm_variant", glm_variant,         # <-- new
                    "--mode", mode,
                    "--runs", runs,
                ]

                if use_pair_mask:
                    cmd_parts.append("--use_pair_mask")

                f.write(" ".join(cmd_parts) + "\n")

    total_jobs = len(sub_list) * len(sessions)
    print(f"✅ Swarm file saved to: {output_file}")
    print(f"📌 {total_jobs} jobs written for {len(sub_list)} subjects × {len(sessions)} sessions")
    print()
    print("Parameters:")
    print(f"  base_dir:      {base_dir}")
    print(f"  mask:          {mask_path}")
    print(f"  stat_type:     {stat_type}")
    print(f"  glm_variant:   {glm_variant}")
    print(f"  mode:          {mode}")
    print(f"  runs:          {runs}")
    print(f"  radius:        {radius_mm} mm")
    print(f"  n_jobs:        {n_jobs}")
    print(f"  rsa_method:    {rsa_method}")
    print(f"  cue_subset:    {cue_subset}")
    print(f"  models:        {models_list}")
    print(f"  use_pair_mask: {use_pair_mask}")
    print()
    print("Submit with:")
    print(f"  swarm -f {output_file} -g 8 -t {n_jobs}")


if __name__ == "__main__":
    generate_swarm_file()
