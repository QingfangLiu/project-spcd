from nilearn.decoding import SearchLight
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.image import load_img, new_img_like
import numpy as np
import os

from decoding_utils import load_decoding_data, load_betas


if __name__ == "__main__":
    # ---- Paths ----
    base_dir = "/Users/liuq13/SPCD_data"
    mask_img = load_img('/Users/liuq13/SPCD_data/masks/gm_0.1.nii')

    normalization_options = ['none', 'across_voxels', 'across_trials']
    parent_out_dir = os.path.join(base_dir, "searchlight_maps", "odor_scan_2odors")
    os.makedirs(parent_out_dir, exist_ok=True)

    # ---- Subject list ----
    sub_list = sorted([
        d for d in os.listdir(os.path.join(base_dir, 'GLM', 'odor_scan_trialwise_2mm'))
        if d.startswith("sub-")
    ])

    for normalization in normalization_options:
        out_dir = os.path.join(parent_out_dir, normalization)
        os.makedirs(out_dir, exist_ok=True)

        for sub in sub_list:
            out_path = os.path.join(out_dir, f"{sub}_searchlight.nii.gz")
            if os.path.exists(out_path):
                print(f"⏭️ Skipping {sub} ({normalization} already processed)")
            continue

        print(f"\nRunning {normalization} normalization for {sub}...")

        try:
            all_imgs, all_y, groups = load_decoding_data(sub, base_dir, normalization=normalization)

            searchlight = SearchLight(
                mask_img=mask_img,
                radius=3,
                estimator=SVC(kernel='linear'),
                n_jobs=-1,
                cv=LeaveOneGroupOut(),
                verbose=1,
                scoring='accuracy'
            )

            searchlight.fit(all_imgs, all_y, groups)
            searchlight_img = new_img_like(mask_img, searchlight.scores_)
            searchlight_img.to_filename(out_path)
            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"❌ Error with {sub} ({normalization}): {e}")
