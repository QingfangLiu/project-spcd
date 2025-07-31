
import os

# -------- CONFIG --------
base_dir = "/data/liuq13/SPCD_decoding_subset"
sub_dir = os.path.join(base_dir, "GLM", "place_trialwise_2mm")
subs = sorted([d for d in os.listdir(sub_dir) if d.startswith("sub-")])
norm_list = ["none", "across_trials"]  # you can add "across_voxels", "across_trials" if needed

# -------- WRITE SWARM FILE --------
with open("swarm_place_decode.swarm", "w") as f:
    for sub in subs:
        for norm in norm_list:
            cmd = f"python run_searchlight_place.py --sub {sub} --norm {norm}"
            f.write(cmd + "\n")
