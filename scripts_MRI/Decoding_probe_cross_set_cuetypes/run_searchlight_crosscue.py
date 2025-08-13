
# run_searchlight_crosscue.py
# Approach 2 runner with TWO modes:
#   - crosscue_set2set  (pooled across runs)
#   - crosscue_strict   (run-matched; held-out run r)
import os
import argparse
import numpy as np
from nilearn.decoding import SearchLight
from nilearn.image import load_img, new_img_like
from sklearn.svm import SVC

from probe_decoding_utils_xcue import load_decoding_data_both_cues

def _splits_crosscue_set2set(set_ids, cue_types):
    idx = np.arange(len(set_ids))
    train1 = idx[(set_ids == 1) & (cue_types == 'A')]
    test1  = idx[(set_ids == 2) & (cue_types == 'B')]
    train2 = idx[(set_ids == 1) & (cue_types == 'B')]
    test2  = idx[(set_ids == 2) & (cue_types == 'A')]
    splits = []
    if len(train1) > 0 and len(test1) > 0: splits.append((train1, test1))
    if len(train2) > 0 and len(test2) > 0: splits.append((train2, test2))
    if not splits:
        raise RuntimeError("No valid crosscue set2set splits (check sets/cue balance).")
    return splits

def _splits_crosscue_strict(run_ids, set_ids, cue_types):
    """Run-matched cross-cue + cross-set.
    For each run r:
      - Train: (run != r) & (set==1 & cue=='A')  -> Test: (run==r) & (set==2 & cue=='B')
      - Train: (run != r) & (set==1 & cue=='B')  -> Test: (run==r) & (set==2 & cue=='A')
    """
    runs = np.unique(run_ids)
    idx = np.arange(len(run_ids))
    splits = []
    for r in runs:
        tr1 = idx[(run_ids != r) & (set_ids == 1) & (cue_types == 'A')]
        te1 = idx[(run_ids == r) & (set_ids == 2) & (cue_types == 'B')]
        tr2 = idx[(run_ids != r) & (set_ids == 1) & (cue_types == 'B')]
        te2 = idx[(run_ids == r) & (set_ids == 2) & (cue_types == 'A')]
        if len(tr1) > 0 and len(te1) > 0: splits.append((tr1, te1))
        if len(tr2) > 0 and len(te2) > 0: splits.append((tr2, te2))
    if not splits:
        raise RuntimeError("No valid crosscue strict splits (check runs/sets/cues).")
    return splits

def run_one_subject(sub, base_dir, mask_img_path,
                    n_classes=2, radius_mm=8.0, n_jobs=-1,
                    session_folder='premeal_trialwise_2mm', session_tag='premeal',
                    mode='crosscue_set2set'):
    out_dir = os.path.join(base_dir, 'searchlight_maps', 'probe_crosscue')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSubject {sub} | mode={mode} | classes={n_classes} | radius={radius_mm}mm")

    all_imgs, all_y, run_ids, set_ids, cue_types = load_decoding_data_both_cues(
        sub, base_dir,
        runs=(1,2,3),
        n_classes=n_classes,
        normalize=True,
        session_folder=session_folder,
        session_tag=session_tag
    )

    if mode == 'crosscue_set2set':
        splits = _splits_crosscue_set2set(set_ids, cue_types)
    elif mode == 'crosscue_strict':
        splits = _splits_crosscue_strict(run_ids, set_ids, cue_types)
    else:
        raise ValueError("mode must be one of: crosscue_set2set, crosscue_strict")

    mask_img = load_img(mask_img_path)
    tag = f"{sub}_session-{session_tag}_mode-{mode}_cue-AB_classes-{n_classes}_radius-{int(radius_mm)}mm"
    out_path = os.path.join(out_dir, f"{tag}.nii.gz")
    if os.path.exists(out_path):
        print(f"⏭️ Skipping {sub} (exists): {out_path}")
        return

    sl = SearchLight(
        mask_img=mask_img,
        radius=radius_mm,
        estimator=SVC(kernel='linear', C=1.0),
        cv=splits,
        n_jobs=n_jobs,
        verbose=1,
        scoring='accuracy'
    )
    sl.fit(all_imgs, all_y)
    out_img = new_img_like(mask_img, sl.scores_.astype(np.float32))
    out_img.to_filename(out_path)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--mask_img", required=True)
    ap.add_argument("--n_classes", type=int, choices=[2,3], default=2)
    ap.add_argument("--radius_mm", type=float, default=8.0)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--session_folder", default="premeal_trialwise_2mm")
    ap.add_argument("--session_tag", default="premeal")
    ap.add_argument("--mode", choices=["crosscue_set2set", "crosscue_strict"], default="crosscue_set2set")
    args = ap.parse_args()

    run_one_subject(
        sub=args.sub,
        base_dir=args.base_dir,
        mask_img_path=args.mask_img,
        n_classes=args.n_classes,
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        session_folder=args.session_folder,
        session_tag=args.session_tag,
        mode=args.mode
    )
