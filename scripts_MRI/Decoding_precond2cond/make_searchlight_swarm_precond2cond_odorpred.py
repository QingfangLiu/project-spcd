# make_searchlight_swarm_precond2cond_odorpred.py
#
# Swarm generator for Goal 4:
#   Day-1 pre-conditioned A cues (localizer) →
#   Day-2 conditioned B cues during odor prediction.
#
# This builds a swarm file that:
#   - finds subjects who have BOTH
#       (a) a Day-1 localizer GLM (character OR place), and
#       (b) a Day-2 odor prediction cue GLM (EARLY or EXT5s)
#   - writes one line per subject to run:
#       run_searchlight_crosstask_precond2cond_odorpred.py
#
# NOTE
# ----
# Exclusion of sub-35 run 3 is handled inside
# `load_decoding_data_conditioned_B_odorpred`, so nothing special is
# required here.

import os


def generate_swarm_file(
    base_dir,
    script_name="run_searchlight_crosstask_precond2cond_odorpred.py",
    output_file="decode_precond2cond_odorpred.swarm",
    radius_mm=8.0,
    n_jobs=-1,
    odorpred_glm_folder="odorpred_cueEARLY_trialwise_2mm",
    mask_name="gm_0.1.nii",
):
    """
    Parameters
    ----------
    base_dir : str
        Project base directory on Biowulf (e.g., '/data/liuq13/SPCD_data').
    script_name : str
        Name of the decoding script to call.
    output_file : str
        Swarm filename to write.
    radius_mm : float
        Searchlight radius in mm (must match decoding script).
    n_jobs : int
        Number of parallel jobs *within* each searchlight call.
        Use -1 to let nilearn use all cores on a node.
    odorpred_glm_folder : str
        Name of the odor-prediction GLM folder under GLM/, e.g.:
          - 'odorpred_cueEARLY_trialwise_2mm'
          - 'odorpred_cueEXT5s_trialwise_2mm'
    mask_name : str
        Filename of the group/GM mask under <base_dir>/masks/.
    """

    glm_odorpred = os.path.join(base_dir, "GLM", odorpred_glm_folder)
    glm_char = os.path.join(base_dir, "GLM", "character_trialwise_2mm")
    glm_place = os.path.join(base_dir, "GLM", "place_trialwise_2mm")

    # Subjects with odor-prediction GLM
    subs_odorpred = {
        d
        for d in os.listdir(glm_odorpred)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_odorpred, d))
    }

    # Subjects with Day-1 localizer (character or place)
    subs_char = {
        d
        for d in os.listdir(glm_char)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_char, d))
    }
    subs_place = {
        d
        for d in os.listdir(glm_place)
        if d.startswith("sub-") and os.path.isdir(os.path.join(glm_place, d))
    }
    subs_localizer = subs_char | subs_place

    # Only subjects with BOTH odorpred + localizer
    sub_list = sorted(subs_odorpred & subs_localizer)

    mask_path = os.path.join(base_dir, "masks", mask_name)

    with open(output_file, "w") as f:
        for sub in sub_list:
            cmd = (
                f"python {script_name} "
                f"--sub {sub} "
                f"--base_dir {base_dir} "
                f"--mask_img {mask_path} "
                f"--radius_mm {radius_mm} "
                f"--n_jobs {n_jobs} "
                f"--odorpred_glm_folder {odorpred_glm_folder}"
            )
            f.write(cmd + "\n")

    total_jobs = len(sub_list)
    print(f"✅ Swarm file saved → {output_file}")
    print(f"📌 {total_jobs} jobs for {len(sub_list)} subjects")
    print(f"  GLM (odorpred): {odorpred_glm_folder}")
    print(f"  Mask: {mask_path}")


if __name__ == "__main__":
    # Example usage: adjust base_dir to your Biowulf path
    base_dir = "/data/liuq13/SPCD_mini"

    # For EARLY GLM version
    generate_swarm_file(
        base_dir=base_dir,
        output_file="decode_precond2cond_odorpred_EARLY.swarm",
        odorpred_glm_folder="odorpred_cueEARLY_trialwise_2mm",
    )

    # If you also want a swarm for EXT5s, uncomment:
    # generate_swarm_file(
    #     base_dir=base_dir,
    #     output_file="decode_precond2cond_odorpred_EXT5s.swarm",
    #     odorpred_glm_folder="odorpred_cueEXT5s_trialwise_2mm",
    # )
