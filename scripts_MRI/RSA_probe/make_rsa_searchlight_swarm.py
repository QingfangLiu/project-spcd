import os

def generate_swarm_file(
    base_dir="/data/liuq13",
    script_name="run_rsa_searchlight.py",
    output_file=None,
    sessions=("premeal", "postmeal"), 
    radius_mm=8.0,
    n_jobs=2,  # cores per swarm job (do NOT set -1 on Biowulf)
    rsa_method="spearman",
    neural_metric="correlation",
    mask_path=None,
    output_dir=None,
    session_folder_template="{session}_trialwise_2mm",
    use_pair_mask=True,
):
    """
    Generate a swarm file for probe searchlight RSA.

    Parameters
    ----------
    base_dir : str
        Base directory containing GLM/ and masks/
    script_name : str
        RSA script to run (run_rsa_searchlight.py)
    output_file : str
        Name of the swarm file to create
    sessions : tuple
        Sessions to run (e.g., ("premeal",) or ("premeal", "postmeal"))
    radius_mm : float
        Searchlight radius in mm
    n_jobs : int
        Number of CPU cores used INSIDE each job (pass to --n_jobs)
        Recommended: set equal to swarm -t value.
    rsa_method : str
        "regression", "spearman", or "kendall"
    neural_metric: str
        Method to conduct neural similarity: 
        'correlation' (default) or 'euclidean'
    mask_path : str or None
        Path to gray matter mask. If None, uses <base_dir>/masks/gm_0.1.nii
    output_dir : str or None
        Output directory root. If None, uses <base_dir>/rsa_maps/probe
        (run_rsa_searchlight.py also defaults this if not passed)
    session_folder_template : str
        Used only for locating subjects (GLM/<session_folder>/sub-*)
    """
    if mask_path is None:
        mask_path = os.path.join(base_dir, "masks", "gm_0.1.nii")

    if output_dir is None:
        output_dir=os.path.join(base_dir, "rsa_maps", "probe")

    if output_file is None:
        mask_tag = "_mask-odor" if use_pair_mask else ""
        output_file = f"rsa_probe_searchlight_{rsa_method}_{neural_metric}{mask_tag}.swarm"

    # Use one session folder to discover subjects (assumes same sub list across sessions)
    discover_folder = session_folder_template.format(session=sessions[0])
    glm_path = os.path.join(base_dir, "GLM", discover_folder)

    if not os.path.isdir(glm_path):
        raise FileNotFoundError(f"GLM folder not found: {glm_path}")

    sub_list = sorted([d for d in os.listdir(glm_path) if d.startswith("sub-")])

    if len(sub_list) == 0:
        raise RuntimeError(f"No subjects found in: {glm_path}")

    # If output_dir is None, you can omit it entirely and let run_rsa_searchlight.py set defaults.
    # But passing it makes paths explicit and session-separated if you choose.
    with open(output_file, "w") as f:
        for sub in sub_list:
            for session in sessions:
                cmd_parts = [
                    "python", script_name,
                    "--sub", sub,
                    "--session", session,
                    "--base_dir", base_dir,
                    "--mask", mask_path,
                    "--radius", str(radius_mm),
                    "--n_jobs", str(n_jobs),
                    "--rsa_method", rsa_method,
                    "--neural_metric", neural_metric,
                ]
                if use_pair_mask:
                    cmd_parts += ["--use_pair_mask"]

                if output_dir is not None:
                    # you can make it session-specific if you want
                    out_sess = os.path.join(output_dir, session)
                    cmd_parts += ["--output_dir", out_sess]

                f.write(" ".join(cmd_parts) + "\n")

    total_jobs = len(sub_list) * len(sessions)
    print(f"✅ Swarm file saved to: {output_file}")
    print(f"📌 {total_jobs} jobs written for {len(sub_list)} subjects × {len(sessions)} sessions")
    print(f"Mask: {mask_path}")
    print(f"Radius: {radius_mm} mm | n_jobs per job: {n_jobs}")


if __name__ == "__main__":
    generate_swarm_file()
        
