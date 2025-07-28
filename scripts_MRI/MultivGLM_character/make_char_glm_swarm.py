import os

def generate_char_glm_swarm(base_dir, output_file="char_glm.swarm"):
    nii_dir = os.path.join(base_dir, "nii")
    sub_list = sorted([
        d for d in os.listdir(nii_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(nii_dir, d))
    ])

    with open(output_file, "w") as f:
        for sub in sub_list:
            cmd = f'module load spm; matlab -nodisplay -r "run_singletrial_GLM(\'{sub}\'); exit"'
            f.write(cmd + "\n")

if __name__ == "__main__":
    # Update this path to your Biowulf base path
    base_dir = "/Users/liuq13/SPCD_data"
    generate_char_glm_swarm(base_dir)
