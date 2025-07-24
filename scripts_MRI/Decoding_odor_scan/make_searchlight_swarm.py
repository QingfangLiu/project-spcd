
import os

def generate_swarm_file(base_dir, script_name="run_searchlight_one.py", output_file="decode.swarm"):
    # Get all subjects from GLM folder
    glm_path = os.path.join(base_dir, "GLM", "odor_scan_trialwise_2mm")
    sub_list = sorted([d for d in os.listdir(glm_path) if d.startswith("sub-")])
    norm_options = ["none", "across_voxels", "across_trials"]

    with open(output_file, "w") as f:
        for sub in sub_list:
            for norm in norm_options:
                cmd = f"python {script_name} --sub {sub} --norm {norm}"
                f.write(cmd + "\n")

    print(f"✅ Swarm file saved to: {output_file}")
    print(f"📌 {len(sub_list) * len(norm_options)} jobs written")

if __name__ == "__main__":
    # Update this path on Biowulf if needed
    base_dir = "/data/liuq13/SPCD_decoding_subset"
    generate_swarm_file(base_dir)
