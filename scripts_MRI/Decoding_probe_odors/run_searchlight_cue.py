import os
import argparse
import numpy as np
from nilearn.decoding import SearchLight
from nilearn.image import load_img, new_img_like
from sklearn.svm import SVC

from odor_decoding_utils import load_decoding_data_cue

def _strict_cv_splits(run_ids, set_ids):
    """
    Build 3 folds: for each run r,
      train = (set==1 & run!=r)
      test  = (set==2 & run==r)
    Skip folds with empty train/test.
    Returns a list of (train_idx, test_idx).
    """
    splits = []
    unique_runs = sorted(np.unique(run_ids))
    n = len(run_ids)
    all_idx = np.arange(n)

    for r in unique_runs:
        train_mask = (set_ids == 1) & (run_ids != r)
        test_mask  = (set_ids == 2) & (run_ids == r)

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        train_idx = all_idx[train_mask]
        test_idx  = all_idx[test_mask]
        splits.append((train_idx, test_idx))

    if len(splits) == 0:
        raise RuntimeError("No valid strict CV folds (check runs/sets).")

    return splits

def run_one_subject(sub, base_dir, mask_img_path,
                    cue_type="A", n_classes=2,
                    mode="strict", radius_mm=8.0, n_jobs=-1,
                    session_folder="premeal_trialwise_2mm", session_tag="premeal"):

    out_dir = os.path.join(base_dir, "searchlight_maps", "cue_crossset")
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{sub}_session-{session_tag}_mode-{mode}_cue-{cue_type}_classes-{n_classes}_radius-{int(radius_mm)}mm.nii.gz"
    out_path = os.path.join(out_dir, tag)

    if os.path.exists(out_path):
        print(f"⏭️ Skipping {sub} (exists): {out_path}")
        return

    print(f"\nSubject {sub} | cue={cue_type} | classes={n_classes} | mode={mode} | radius={radius_mm}mm")

    # Load & normalize (run-wise across all trials)
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

    if mode == "strict":
        # Custom CV with 3 run-matched folds; SearchLight will average across folds
        splits = _strict_cv_splits(run_ids, set_ids)

        searchlight = SearchLight(
            mask_img=mask_img,
            radius=radius_mm,            # mm
            estimator=SVC(kernel='linear', C=1.0),
            n_jobs=n_jobs,
            cv=splits,                   # our custom (train_idx, test_idx) folds
            verbose=1,
            scoring='accuracy'
        )
        searchlight.fit(all_imgs, all_y)

        # Average across folds is already in scores_
        out_img = new_img_like(mask_img, searchlight.scores_)
        out_img.to_filename(out_path)
        print(f"✅ Saved: {out_path}")
        return

    elif mode == "tune_final":
        # --- Lightweight tuning on Set 1 (LOGO by run), pick a single C, then train=set1 test=set2 ---
        from sklearn.model_selection import LeaveOneGroupOut

        # Restrict to Set 1 for tuning
        idx_set1 = np.where(set_ids == 1)[0]
        if len(idx_set1) == 0:
            raise RuntimeError("No Set 1 trials for tuning.")
        groups_set1 = run_ids[idx_set1]
        logo = LeaveOneGroupOut()

        # Try a tiny grid (kept small for searchlight speed)
        C_grid = [0.1, 1.0, 10.0]
        mean_scores = []

        for C in C_grid:
            sl = SearchLight(
                mask_img=mask_img,
                radius=radius_mm,
                estimator=SVC(kernel='linear', C=C),
                n_jobs=n_jobs,
                cv=logo.split(idx_set1, groups=groups_set1),
                verbose=0,
                scoring='accuracy'
            )
            # Fit only on set1 indices
            sl.fit(all_imgs.slicer[..., idx_set1], all_y[idx_set1])
            mean_scores.append(np.nanmean(sl.scores_))

        best_C = C_grid[int(np.argmax(mean_scores))]
        print(f"🔧 tune_final selected C={best_C} (scores={mean_scores})")

        # Now build a single split: train=set1, test=set2
        idx_set2 = np.where(set_ids == 2)[0]
        if len(idx_set2) == 0:
            raise RuntimeError("No Set 2 trials for final test.")

        single_split = [(idx_set1, idx_set2)]

        sl_final = SearchLight(
            mask_img=mask_img,
            radius=radius_mm,
            estimator=SVC(kernel='linear', C=best_C),
            n_jobs=n_jobs,
            cv=single_split,
            verbose=1,
            scoring='accuracy'
        )
        sl_final.fit(all_imgs, all_y)

        out_img = new_img_like(mask_img, sl_final.scores_)
        out_img.to_filename(out_path)
        print(f"✅ Saved: {out_path}")
        return

    else:
        raise ValueError("mode must be 'strict' or 'tune_final'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, help="e.g., sub-01")
    parser.add_argument("--base_dir", default="/Users/liuq13/SPCD_data")
    parser.add_argument("--mask_img", default="/Users/liuq13/SPCD_data/masks/gm_0.1.nii")
    parser.add_argument("--cue_type", choices=["A", "B"], default="A")
    parser.add_argument("--n_classes", type=int, choices=[2, 3], default=2)  # default: odor_1 vs odor_2
    parser.add_argument("--mode", choices=["strict", "tune_final"], default="strict")
    parser.add_argument("--radius_mm", type=float, default=8.0)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--session_folder", default="premeal_trialwise_2mm")
    parser.add_argument("--session_tag", default="premeal")
    args = parser.parse_args()

    run_one_subject(
        sub=args.sub,
        base_dir=args.base_dir,
        mask_img_path=args.mask_img,
        cue_type=args.cue_type,
        n_classes=args.n_classes,
        mode=args.mode,
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        session_folder=args.session_folder,
        session_tag=args.session_tag
    )
