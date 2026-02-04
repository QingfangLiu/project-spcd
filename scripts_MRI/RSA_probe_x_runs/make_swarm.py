#!/usr/bin/env python3
"""
Generate swarm file for cuewise GLM cross-run RSA on Biowulf.

Usage:
    python make_swarm.py

Then submit with:
    swarm -f <swarm_file> -g 8 -t 4
"""

import os

def generate_swarm_file(
    base_dir="/data/liuq13",
    module_name="RSA_probe_x_runs.run_rsa",
    output_file=None,
    sessions=("premeal", "postmeal"),
    radius_mm=8.0,
    n_jobs=4,  # cores per swarm job (match with swarm -t value; do NOT use -1 on Biowulf)
    rsa_method="spearman",
    neural_metric="correlation",
    mask_path=None,
    cue_subset="all",
    models="identity",
    use_pair_mask=False,
    stat_type="tmap",       # "beta" or "tmap"
    mode="both",            # "both", "img1only", "img2only"
    runs="1,2,3",           # comma-separated runs to attempt
    session_folder_template="{session}_cuewise_2mm",  # for subject discovery
):
    """
    Generate a swarm file for cuewise GLM cross-run searchlight RSA.

    Parameters
    ----------
    base_dir : str
        Base directory containing GLM/ and masks/
    module_name : str
        Python module to run
    output_file : str or None
        Name of the swarm file to create. If None, auto-generated.
    sessions : tuple
        Sessions to run (e.g., ("premeal",) or ("premeal", "postmeal"))
    radius_mm : float
        Searchlight radius in mm
    n_jobs : int
        Number of CPU cores used INSIDE each job (pass to --n_jobs).
        Should match your swarm -t value. Do NOT use -1 on Biowulf.
    rsa_method : str
        "regression", "spearman", or "kendall"
    neural_metric : str
        Method for neural similarity: "correlation" (default) or "euclidean"
    mask_path : str or None
        Path to gray matter mask. If None, uses <base_dir>/masks/gm_0.1.nii
    cue_subset : str
        "all" (12 cues), "conditioned" (B/D only), or "preconditioned" (A/C only)
    models : str
        Space-separated model names: "identity", "value", "association"
        Example: "identity value" to run both
    use_pair_mask : bool
        If True, exclude clean-involving pairs (odor-odor only)
    stat_type : str
        Input maps from cuewise GLM: "beta" or "tmap"
    mode : str
        GLM mode folder: "both", "img1only", or "img2only"
    runs : str
        Comma-separated runs to attempt (e.g., "1,2,3"). Missing runs are skipped.
    session_folder_template : str
        Template for locating subjects: GLM/<session_folder>/sub-*
    """
    # Defaults
    if mask_path is None:
        mask_path = os.path.join(base_dir, "masks", "gm_0.1.nii")

    # Auto-generate output filename if not provided
    if output_file is None:
        mask_tag = "_mask-odor" if use_pair_mask else ""
        models_tag = models.replace(" ", "-")  # "identity value" -> "identity-value"
        output_file = (
            f"rsa_cuewise_{stat_type}_{rsa_method}_{cue_subset}_{models_tag}{mask_tag}.swarm"
        )

    # Discover subjects from one session folder
    discover_folder = session_folder_template.format(session=sessions[0])
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
                    "--neural_metric", neural_metric,
                    "--cue_subset", cue_subset,
                    "--models", models,
                    "--stat_type", stat_type,
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
    print(f"  mode:          {mode}")
    print(f"  runs:          {runs}")
    print(f"  radius:        {radius_mm} mm")
    print(f"  n_jobs:        {n_jobs}")
    print(f"  rsa_method:    {rsa_method}")
    print(f"  neural_metric: {neural_metric}")
    print(f"  cue_subset:    {cue_subset}")
    print(f"  models:        {models}")
    print(f"  use_pair_mask: {use_pair_mask}")
    print()
    print("Submit with:")
    print(f"  swarm -f {output_file} -g 8 -t {n_jobs} --time 2:00:00 --module python/3.9")


if __name__ == "__main__":
    # Example: generate swarm for identity model, spearman correlation, all cues
    generate_swarm_file(
        sessions=("premeal", "postmeal"),
        rsa_method="spearman",
        cue_subset="all",
        models="identity",
        stat_type="beta",
        use_pair_mask=False,
    )