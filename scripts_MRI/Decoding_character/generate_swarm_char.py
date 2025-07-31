
import os

# -------- CONFIG --------
base_dir = "/data/liuq13/SPCD_decoding_subset"
sub_dir = os.path.join(base_dir, "GLM", "character_trialwise_2mm")
subs = sorted([d for d in os.listdir(sub_dir) if d.startswith("sub-")])
norm_list = ["none"]  # you can add "across_voxels", "across_trials" if needed

# -------- WRITE SWARM FILE --------
with open("swarm_char_decode.swarm", "w") as f:
    for sub in subs:
        for norm in norm_list:
            cmd = f"python run_searchlight_char.py --sub {sub} --norm {norm}"
            f.write(cmd + "\n")
