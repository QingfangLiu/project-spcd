# make_searchlight_swarm_cond2precond.py
#
# Swarm generator for Goal 1: cond (B) → precond (A) cue reactivation.

import os

def generate_swarm_file(
    base_dir,
    script_name="run_searchlight_crosstask_cond2precond.py",
    output_file="decode_cond2precond.swarm",
    radius_mm=8.0,
    n_jobs=-1,
    session_folder="premeal_trialwise_2mm",
    session_tag="premeal",
):
    """
    Build a swarm file that:
      - finds subjects with both a Day-1 localizer GLM (character/place)
        and Day-2 premeal probe GLM
      - runs Goal 1 decoding for each subject
    """

    glm_probe = os.path.join(base_dir, "GLM", session_folder)
    glm_char  = os.path.join(base_dir, "GLM", "character_trialwise_2mm")
    glm_place = os.path.join(base_dir, "GLM", "place_trialwise_2mm")

    subs_probe = {
        d for d in os.listdir(glm_probe)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_probe, d))
    }
    subs_char = {
        d for d in os.listdir(glm_char)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_char, d))
    }
    subs_place = {
        d for d in os.listdir(glm_place)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_place, d))
    }

    # Day-1 localizer exists if subject is in either char or place GLM folder
    subs_localizer = subs_char | subs_place
    sub_list = sorted(subs_probe & subs_localizer)

    with open(output_file, "w") as f:
        for sub in sub_list:
            cmd = (
                f"python {script_name} "
                f"--sub {sub} "
                f"--base_dir {base_dir} "
                f"--mask_img {os.path.join(base_dir, 'masks', 'gm_0.1.nii')} "
                f"--radius_mm {radius_mm} "
                f"--n_jobs {n_jobs} "
                f"--session_folder {session_folder} "
                f"--session_tag {session_tag}"
            )
            f.write(cmd + "\n")

    total_jobs = len(sub_list)
    print(f"✅ Swarm file saved → {output_file}")
    print(f"📌 {total_jobs} jobs for {len(sub_list)} subjects")


if __name__ == "__main__":
    # Example usage: adjust base_dir to your Biowulf path
    base_dir = "/data/liuq13/SPCD_mini"
    generate_swarm_file(base_dir)
