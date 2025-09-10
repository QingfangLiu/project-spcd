# make_searchlight_swarm_crosstask.py
import os

def generate_swarm_file(
    base_dir,
    script_name="run_searchlight_crosstask_odor2probe.py",
    output_file="decode_crosstask_odor2probe.swarm",
    cue_types=("A", "B"),
    radius_mm=8.0,
    n_jobs=-1,
    session_folder="premeal_trialwise_2mm",
    session_tag="premeal",
    odor_glm_folder="odor_scan_trialwise_2mm"
):
    """
    Swarm for cross-task decoding: TRAIN (odor) -> TEST (pre-meal probe).
    - Uses presence of both GLM folders to define the subject list.
    - Runs A and B separately, n_classes=2 by default (drop clean-air cues).
    """
    glm_probe = os.path.join(base_dir, "GLM", session_folder)
    glm_odor  = os.path.join(base_dir, "GLM", odor_glm_folder)

    subs_probe = {d for d in os.listdir(glm_probe)
                  if d.startswith("sub-") and os.path.isdir(os.path.join(glm_probe, d))}
    subs_odor  = {d for d in os.listdir(glm_odor)
                  if d.startswith("sub-") and os.path.isdir(os.path.join(glm_odor, d))}
    sub_list = sorted(subs_probe & subs_odor)

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
                    f"--radius_mm {radius_mm} "
                    f"--n_jobs {n_jobs} "
                    f"--session_folder {session_folder} "
                    f"--session_tag {session_tag}"
                )
                f.write(cmd + "\n")

    total_jobs = len(sub_list) * len(cue_types)
    print(f"✅ Swarm file saved → {output_file}")
    print(f"📌 {total_jobs} jobs for {len(sub_list)} subjects × {len(cue_types)} cue types")

if __name__ == "__main__":
    # ✏️ Set to your Biowulf subset
    base_dir = "/data/liuq13/SPCD_mini"
    generate_swarm_file(base_dir)
