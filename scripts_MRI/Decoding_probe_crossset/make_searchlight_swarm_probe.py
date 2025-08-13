import os

def generate_swarm_file(
    base_dir,
    script_name="run_searchlight_cue.py",
    output_file="decode_cue_crossset.swarm",
    cue_types=("A", "B"),
    mode="strict",
    radius_mm=8.0,
    n_jobs=-1,
    session_folder="premeal_trialwise_2mm",
    session_tag="premeal"
):
    """
    Generate a swarm file for cue cross-set decoding.

    Parameters
    ----------
    base_dir : str
        Path to the subset dataset root on Biowulf.
    script_name : str
        Name of the decoding script to run (should be in the same folder or in PATH).
    output_file : str
        Name of the swarm file to create.
    cue_types : tuple
        Cue types to run (e.g., ("A", "B")).
    mode : str
        Decoding mode: 'strict' or 'set2set'
    radius_mm : float
        Searchlight radius in mm.
    n_jobs : int
        Number of CPU cores per job.
    session_folder : str
        GLM subfolder name under GLM/.
    session_tag : str
        Session tag for the script's arguments.
    """
    glm_path = os.path.join(base_dir, "GLM", session_folder)
    sub_list = sorted([d for d in os.listdir(glm_path) if d.startswith("sub-")])

    with open(output_file, "w") as f:
        for sub in sub_list:
            for cue in cue_types:
                cmd = (
                    f"python {script_name} "
                    f"--sub {sub} "
                    f"--base_dir {base_dir} "
                    f"--mask_img {os.path.join(base_dir, 'masks', 'gm_0.1.nii')} "
                    f"--cue_type {cue} "
                    f"--n_classes 2 "
                    f"--mode {mode} "
                    f"--radius_mm {radius_mm} "
                    f"--n_jobs {n_jobs} "
                    f"--session_folder {session_folder} "
                    f"--session_tag {session_tag}"
                )
                f.write(cmd + "\n")

    total_jobs = len(sub_list) * len(cue_types)
    print(f"✅ Swarm file saved to: {output_file}")
    print(f"📌 {total_jobs} jobs written for {len(sub_list)} subjects × {len(cue_types)} cue types")


if __name__ == "__main__":
    # Update paths for your Biowulf environment
    base_dir = "/data/liuq13/SPCD_decoding_probe_subset"
    generate_swarm_file(base_dir)
