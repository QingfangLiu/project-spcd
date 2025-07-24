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
    out_dir = os.path.join(base_dir, "searchlight_maps", "odor_scan")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Subject list ----
    sub_list = sorted([
        d for d in os.listdir(os.path.join(base_dir, 'GLM', 'odor_scan_trialwise_2mm'))
        if d.startswith("sub-")
    ])

    for sub in sub_list:
        out_path = os.path.join(out_dir, f"{sub}_searchlight.nii.gz")
        if os.path.exists(out_path):
            print(f"⏭️ Skipping {sub} (already processed)")
            continue

        print(f"\nRunning searchlight for {sub}...")

        try:
            all_imgs, all_y, groups = load_decoding_data(sub, base_dir)

            # Set up searchlight
            searchlight = SearchLight(
                mask_img=mask_img,
                radius=3,
                estimator=SVC(kernel='linear'),
                n_jobs=-1,  # all CPUs
                cv=LeaveOneGroupOut(),
                verbose=1,
                scoring='accuracy'
            )

            searchlight.fit(all_imgs, all_y, groups)

            # Save result as Nifti
            searchlight_img = new_img_like(mask_img, searchlight.scores_)
            searchlight_img.to_filename(out_path)
            print(f"✅ Saved: {out_path}")

        except Exception as e:
            print(f"❌ Error with {sub}: {e}")
