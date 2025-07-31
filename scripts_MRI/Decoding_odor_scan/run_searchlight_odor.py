import os
import argparse
from nilearn.decoding import SearchLight
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from nilearn.image import load_img, new_img_like
from odor_decoding_utils import load_decoding_data

def run_one_subject(sub, base_dir, mask_img_path, normalization):
    out_dir = os.path.join(base_dir, "searchlight_maps", "odor_scan_2odors", normalization)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{sub}_searchlight.nii.gz")

    if os.path.exists(out_path):
        print(f"⏭️ Skipping {sub} ({normalization} already processed)")
        return

    print(f"\nRunning {normalization} normalization for {sub}...")

    try:
        all_imgs, all_y, groups = load_decoding_data(sub, base_dir, normalization=normalization)

        searchlight = SearchLight(
            mask_img=load_img(mask_img_path),
            radius=4, # use 4mm for odor decoding
            estimator=SVC(kernel='linear'),
            n_jobs=-1,
            cv=LeaveOneGroupOut(),
            verbose=1,
            scoring='accuracy'
        )

        searchlight.fit(all_imgs, all_y, groups)
        searchlight_img = new_img_like(load_img(mask_img_path), searchlight.scores_)
        searchlight_img.to_filename(out_path)
        print(f"✅ Saved: {out_path}")

    except Exception as e:
        print(f"❌ Error with {sub} ({normalization}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, help="Subject ID (e.g., sub-01)")
    parser.add_argument("--norm", required=True, help="Normalization: none, across_voxels, across_trials")
    parser.add_argument("--base_dir", default="/data/liuq13/SPCD_decoding_subset", help="Base directory of data")
    parser.add_argument("--mask_img", default="/data/liuq13/SPCD_decoding_subset/masks/gm_0.1.nii", help="Path to gray matter mask")
    args = parser.parse_args()

    run_one_subject(args.sub, args.base_dir, args.mask_img, args.norm)
