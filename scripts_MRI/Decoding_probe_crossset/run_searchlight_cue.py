
import os
import argparse
import numpy as np
import nibabel as nib

from nilearn.decoding import SearchLight
from nilearn.image import load_img, new_img_like
from sklearn.svm import SVC

from probe_decoding_utils import load_decoding_data_cue

def _strict_cv_splits(run_ids, set_ids):
    # Build strict cross-validation folds with set-based constraints.
    #
    # For each run `r`, create two folds:
    #   Case 1: Train on set 1 from all runs except `r`, test on set 2 from run `r`.
    #   Case 2: Train on set 2 from all runs except `r`, test on set 1 from run `r`.
    splits = []
    unique_runs = sorted(np.unique(run_ids))
    n = len(run_ids)
    all_idx = np.arange(n)
    for r in unique_runs:
        # Case 1: set 1 → train, set 2 → test
        train_mask = (set_ids == 1) & (run_ids != r)
        test_mask  = (set_ids == 2) & (run_ids == r)
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            splits.append((all_idx[train_mask], all_idx[test_mask]))
        # Case 2: set 2 → train, set 1 → test
        train_mask = (set_ids == 2) & (run_ids != r)
        test_mask  = (set_ids == 1) & (run_ids == r)
        if train_mask.sum() > 0 and test_mask.sum() > 0:
            splits.append((all_idx[train_mask], all_idx[test_mask]))
    if len(splits) == 0:
        raise RuntimeError("No valid strict CV folds (check runs/sets).")
    return splits

def _set2set_cv_splits(run_ids, set_ids):
    # Pooled set-to-set generalization:
    #   Fold 1: Train on ALL trials in set 1 (across all runs), test on ALL trials in set 2.
    #   Fold 2: Train on ALL trials in set 2, test on ALL trials in set 1.
    n = len(run_ids)
    all_idx = np.arange(n)
    splits = []
    # Fold 1: set1 -> train, set2 -> test
    train_mask = (set_ids == 1)
    test_mask  = (set_ids == 2)
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        splits.append((all_idx[train_mask], all_idx[test_mask]))
    # Fold 2: set2 -> train, set1 -> test
    train_mask = (set_ids == 2)
    test_mask  = (set_ids == 1)
    if train_mask.sum() > 0 and test_mask.sum() > 0:
        splits.append((all_idx[train_mask], all_idx[test_mask]))
    if len(splits) == 0:
        raise RuntimeError("No valid set2set CV folds (check sets).")
    return splits

def run_one_subject(sub, base_dir, mask_img_path,
                    cue_type, n_classes=2,
                    radius_mm=8.0, n_jobs=-1,
                    mode="strict",
                    session_folder="premeal_trialwise_2mm", session_tag="premeal"):
    # Decoding with leakage-safe normalization (set-wise within run) and fixed-C linear SVM.
    # 'strict' mode controls for run context; 'set2set' pools across runs per set.
    out_dir = os.path.join(base_dir, "searchlight_maps", "probe_crossset")
    os.makedirs(out_dir, exist_ok=True)
    tag_stem = f"{sub}_session-{session_tag}_mode-{mode}_cue-{cue_type}_classes-{n_classes}_radius-{int(radius_mm)}mm"
    # Load & normalize (SET-wise within each run; no leakage)
    all_imgs, all_y, run_ids, set_ids = load_decoding_data_cue(
        sub, base_dir,
        runs=(1, 2, 3),
        cue_type=cue_type,
        n_classes=n_classes,
        normalize=True,
        session_folder=session_folder,
        session_tag=session_tag
    )
    mask_img = load_img(mask_img_path)
    out_path = os.path.join(out_dir, f"{tag_stem}.nii.gz")
    if os.path.exists(out_path):
        print(f"⏭️ Skipping {sub} (exists): {out_path}")
        return
    print(f"\nSubject {sub} | cue={cue_type} | classes={n_classes} | mode={mode} | radius={radius_mm}mm")
    # Build CV splits per mode
    if mode == "strict":
        splits = _strict_cv_splits(run_ids, set_ids)
    elif mode == "set2set":
        splits = _set2set_cv_splits(run_ids, set_ids)
    else:
        raise ValueError("mode must be 'strict' or 'set2set'")
    # Searchlight with fixed-C linear SVM
    searchlight = SearchLight(
        mask_img=mask_img,
        radius=radius_mm,
        estimator=SVC(kernel='linear', C=1.0),
        n_jobs=n_jobs,
        cv=splits,
        verbose=1,
        scoring='accuracy'
    )
    searchlight.fit(all_imgs, all_y)
    out_img = new_img_like(mask_img, searchlight.scores_.astype(np.float32))
    out_img.to_filename(out_path)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, help="e.g., sub-01")
    parser.add_argument("--base_dir", default="/data/liuq13/SPCD_decoding_cue_subset")
    parser.add_argument("--mask_img", default="/data/liuq13/SPCD_decoding_cue_subset/masks/gm_0.1.nii")
    parser.add_argument("--cue_type", choices=["A", "B"], default="A")
    parser.add_argument("--n_classes", type=int, choices=[2, 3], default=2)
    parser.add_argument("--radius_mm", type=float, default=8.0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--mode", choices=["strict", "set2set"], default="strict")
    parser.add_argument("--session_folder", default="premeal_trialwise_2mm")
    parser.add_argument("--session_tag", default="premeal")
    args = parser.parse_args()
    run_one_subject(
        sub=args.sub,
        base_dir=args.base_dir,
        mask_img_path=args.mask_img,
        cue_type=args.cue_type,
        n_classes=args.n_classes,
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        mode=args.mode,
        session_folder=args.session_folder,
        session_tag=args.session_tag
    )
