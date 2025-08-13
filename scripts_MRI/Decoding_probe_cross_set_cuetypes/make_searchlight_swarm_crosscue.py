
# make_searchlight_swarm_crosscue.py
import os

def generate_swarm_file(
    base_dir,
    script_name="run_searchlight_crosscue.py",
    output_file="decode_crosscue.swarm",
    radius_mm=8.0,
    n_jobs=-1,
    session_folder="premeal_trialwise_2mm",
    session_tag="premeal",
    n_classes=2,
    mode="crosscue_set2set"
):
    """Generate a swarm file for cross-cue + cross-set decoding with chosen mode."""
    glm_path = os.path.join(base_dir, "GLM", session_folder)
    sub_list = sorted([d for d in os.listdir(glm_path) if d.startswith("sub-")])

    with open(output_file, "w") as f:
        for sub in sub_list:
            cmd = (
                f"python {script_name} "
                f"--sub {sub} "
                f"--base_dir {base_dir} "
                f"--mask_img {os.path.join(base_dir, 'masks', 'gm_0.1.nii')} "
                f"--n_classes {n_classes} "
                f"--radius_mm {radius_mm} "
                f"--n_jobs {n_jobs} "
                f"--session_folder {session_folder} "
                f"--session_tag {session_tag} "
                f"--mode {mode}"
            )
            f.write(cmd + "\n")

    print(f"✅ Swarm file saved to: {output_file}")
    print(f"📌 {len(sub_list)} jobs written (one per subject).")


if __name__ == "__main__":
    base_dir = "/data/liuq13/SPCD_decoding_probe_subset"
    # change 'mode' to "crosscue_strict" to generate a strict swarm
    generate_swarm_file(base_dir, mode="crosscue_set2set")
