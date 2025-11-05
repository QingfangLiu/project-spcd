# prepare_decoding_cond2precond_subset.py
#
# Prepare data subset for Goal 1:
#   Day-1 conditioned cues (B) → Day-2 (or Day-3) probe A cues
#
# Copies, for each eligible subject:
#   - GLM/<probe_folder>/<sub-XX>/          (probe betas + CSVs, e.g. premeal or postmeal)
#   - GLM/{character|place}_trialwise_2mm/<sub-XX>/ (Day-1 localizer betas)
#   - nii/<sub-XX>/func_PAID/{character|place}/*.tsv (localizer events)
# And also:
#   - masks/gm_0.1.nii
#   - task_info/subject_info.csv
#
# Eligibility:
#   subject must have both:
#     - GLM/<probe_folder>/<sub-XX>
#     - Day-1 GLM in the folder implied by task_info/subject_info.csv:
#         pairing == 'c' -> character_trialwise_2mm
#         pairing == 'p' -> place_trialwise_2mm

import os
import shutil
import pandas as pd


def _copytree(src, dst):
    """Copy file or directory, creating any missing parent dirs."""
    if os.path.isdir(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _load_subject_info(full_data_dir):
    """Load task_info/subject_info.csv as a DataFrame indexed by 'subject'."""
    path = os.path.join(full_data_dir, "task_info", "subject_info.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"subject_info.csv not found: {path}")
    df = pd.read_csv(path)
    if "subject" not in df.columns or "pairing" not in df.columns:
        raise ValueError("subject_info.csv must have columns 'subject' and 'pairing'")
    return df.set_index("subject")


def prepare_goal1_subset(
    full_data_dir,
    subset_dir,
    probe_folder="premeal_trialwise_2mm",   # ← can be 'premeal_trialwise_2mm' or 'postmeal_trialwise_2mm'
    char_glm_folder="character_trialwise_2mm",
    place_glm_folder="place_trialwise_2mm",
):
    """
    Prepare subset of files for Goal 1 (cond → precond cue reactivation).

    Parameters
    ----------
    full_data_dir : str
        Root of your full SPCD_data (contains GLM, nii, masks, task_info).
    subset_dir : str
        Destination root for the subset to sync to Biowulf.
    probe_folder : str
        Name of the probe GLM folder under GLM/ (e.g. 'premeal_trialwise_2mm'
        or 'postmeal_trialwise_2mm').
    char_glm_folder : str
        Name of the character localizer GLM folder under GLM/.
    place_glm_folder : str
        Name of the place localizer GLM folder under GLM/.
    """
    subject_info = _load_subject_info(full_data_dir)

    probe_root = os.path.join(full_data_dir, "GLM", probe_folder)
    char_root = os.path.join(full_data_dir, "GLM", char_glm_folder)
    place_root = os.path.join(full_data_dir, "GLM", place_glm_folder)

    subs_probe = {
        d
        for d in os.listdir(probe_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(probe_root, d))
    }

    # Determine, for each subject with probe GLM, whether the required
    # Day-1 GLM (character or place) exists, based on pairing.
    eligible_subs = []
    for sub in sorted(subs_probe):
        if sub not in subject_info.index:
            print(f"⚠️ {sub} not in subject_info.csv, skipping.")
            continue
        pairing = subject_info.loc[sub, "pairing"]
        if pairing == "c":
            glm_root = char_root
        elif pairing == "p":
            glm_root = place_root
        else:
            print(f"⚠️ {sub}: unknown pairing '{pairing}', skipping.")
            continue

        glm_sub_dir = os.path.join(glm_root, sub)
        if not os.path.isdir(glm_sub_dir):
            print(f"⚠️ {sub}: expected Day-1 GLM not found at {glm_sub_dir}, skipping.")
            continue

        eligible_subs.append(sub)

    print(f"🧾 Subjects eligible for Goal 1 subset ({probe_folder}): {len(eligible_subs)}")
    if not eligible_subs:
        print("No subjects found; nothing to copy.")
        return

    # 1) Copy mask
    mask_src = os.path.join(full_data_dir, "masks", "gm_0.1.nii")
    mask_dst = os.path.join(subset_dir, "masks", "gm_0.1.nii")
    os.makedirs(os.path.dirname(mask_dst), exist_ok=True)
    shutil.copy2(mask_src, mask_dst)
    print(f"✅ Copied mask → {mask_dst}")

    # 2) Copy subject_info.csv
    subjinfo_src = os.path.join(full_data_dir, "task_info", "subject_info.csv")
    subjinfo_dst = os.path.join(subset_dir, "task_info", "subject_info.csv")
    os.makedirs(os.path.dirname(subjinfo_dst), exist_ok=True)
    shutil.copy2(subjinfo_src, subjinfo_dst)
    print(f"✅ Copied subject_info.csv → {subjinfo_dst}")

    # 3) Copy per-subject GLMs + events
    for sub in eligible_subs:
        pairing = subject_info.loc[sub, "pairing"]
        print(f"📦 {sub} (pairing={pairing})")

        # Probe GLM (premeal or postmeal)
        src_probe = os.path.join(probe_root, sub)
        dst_probe = os.path.join(subset_dir, "GLM", probe_folder, sub)
        _copytree(src_probe, dst_probe)

        # Day-1 GLM (character or place)
        if pairing == "c":
            src_glm = os.path.join(char_root, sub)
            dst_glm = os.path.join(subset_dir, "GLM", char_glm_folder, sub)
            stim_type = "character"
        else:  # pairing == 'p'
            src_glm = os.path.join(place_root, sub)
            dst_glm = os.path.join(subset_dir, "GLM", place_glm_folder, sub)
            stim_type = "place"

        _copytree(src_glm, dst_glm)

        # Localizer events TSVs:
        # nii/<sub>/func_PAID/<stim_type>/*.tsv
        src_events_dir = os.path.join(
            full_data_dir, "nii", sub, "func_PAID", stim_type
        )
        if os.path.isdir(src_events_dir):
            dst_events_dir = os.path.join(
                subset_dir, "nii", sub, "func_PAID", stim_type
            )
            os.makedirs(dst_events_dir, exist_ok=True)
            for fname in os.listdir(src_events_dir):
                if fname.endswith(".tsv"):
                    shutil.copy2(
                        os.path.join(src_events_dir, fname),
                        os.path.join(dst_events_dir, fname),
                    )
        else:
            print(f"⚠️ Localizer events folder missing for {sub}: {src_events_dir}")


if __name__ == "__main__":
    # ✏️ Edit these before running on your laptop

    full_data_dir = "/Users/liuq13/SPCD_data"

    # Example 1: premeal subset (what you already did)
    # subset_dir = "/Users/liuq13/SPCD_premeal_subset"
    # prepare_goal1_subset(full_data_dir, subset_dir,
    #                      probe_folder="premeal_trialwise_2mm")

    # Example 2: **postmeal** subset (what you want now)
    subset_dir = "/Users/liuq13/SPCD_postmeal_subset"
    prepare_goal1_subset(full_data_dir, subset_dir,
                         probe_folder="postmeal_trialwise_2mm")
