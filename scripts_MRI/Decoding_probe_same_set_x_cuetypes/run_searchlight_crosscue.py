
# run_searchlight_crosscue.py
# Runner with two modes:
#   - withinset_set2set       (NEW; within-set + cross-cue; pooled across runs)
#   - withinset_strict        (NEW; within-set + cross-cue; run-matched)

import os
import argparse
import numpy as np
from nilearn.decoding import SearchLight
from nilearn.image import load_img, new_img_like
from sklearn.svm import SVC

from probe_decoding_utils_xcue import load_decoding_data_both_cues

# -------------------------
# Diagnostics / monitoring
# -------------------------
def _print_balance(run_ids, set_ids, cue_types, y):
    idx = np.arange(len(y))
    print("  [monitor] total trials:", len(y))
    for s in [1, 2]:
        for c in ["A", "B"]:
            m = idx[(set_ids == s) & (cue_types == c)]
            if len(m) == 0:
                print(f"  [monitor] set={s} cue={c}: 0")
            else:
                # counts per class label
                uniq, cnt = np.unique(y[m], return_counts=True)
                pairs = ", ".join([f"{int(u)}:{int(k)}" for u, k in zip(uniq, cnt)])
                print(f"  [monitor] set={s} cue={c}: n={len(m)} | y {{{pairs}}}")
    for r in np.unique(run_ids):
        mr = idx[(run_ids == r)]
        print(f"  [monitor] run={int(r)}: n={len(mr)}")

# -------------------------
# NEW splits: within-set + cross-cue
# -------------------------
def _splits_withinset_set2set(set_ids, cue_types):
    """
    Pooled across runs, but within each set s:
      - Train: (set==s & cue=='A') -> Test: (set==s & cue=='B')
      - Train: (set==s & cue=='B') -> Test: (set==s & cue=='A')
    Splits from set 1 and set 2 are concatenated.
    """
    idx = np.arange(len(set_ids))
    splits = []
    for s in [1, 2]:
        tr1 = idx[(set_ids == s) & (cue_types == 'A')]
        te1 = idx[(set_ids == s) & (cue_types == 'B')]
        tr2 = idx[(set_ids == s) & (cue_types == 'B')]
        te2 = idx[(set_ids == s) & (cue_types == 'A')]
        if len(tr1) > 0 and len(te1) > 0: splits.append((tr1, te1))
        if len(tr2) > 0 and len(te2) > 0: splits.append((tr2, te2))
    if not splits:
        raise RuntimeError("No valid withinset_set2set splits (check within-set cue balance).")
    return splits

def _splits_withinset_strict(run_ids, set_ids, cue_types):
    """
    Run-matched within-set cross-cue.
    For each run r and each set s:
      - Train: (run != r) & (set==s & cue=='A') -> Test: (run==r) & (set==s & cue=='B')
      - Train: (run != r) & (set==s & cue=='B') -> Test: (run==r) & (set==s & cue=='A')
    """
    runs = np.unique(run_ids)
    idx = np.arange(len(run_ids))
    splits = []
    for r in runs:
        for s in [1, 2]:
            tr1 = idx[(run_ids != r) & (set_ids == s) & (cue_types == 'A')]
            te1 = idx[(run_ids == r) & (set_ids == s) & (cue_types == 'B')]
            tr2 = idx[(run_ids != r) & (set_ids == s) & (cue_types == 'B')]
            te2 = idx[(run_ids == r) & (set_ids == s) & (cue_types == 'A')]
            if len(tr1) > 0 and len(te1) > 0: splits.append((tr1, te1))
            if len(tr2) > 0 and len(te2) > 0: splits.append((tr2, te2))
    if not splits:
        raise RuntimeError("No valid withinset_strict splits (check runs/within-set cue balance).")
    return splits

# -------------------------
# Runner
# -------------------------
def run_one_subject(sub, base_dir, mask_img_path,
                    n_classes=2, radius_mm=8.0, n_jobs=-1,
                    session_folder='premeal_trialwise_2mm', session_tag='premeal',
                    mode='withinset_set2set',
                    monitor=True):
    # Separate output folders to avoid mixing with old cross-set results
    if mode.startswith("withinset_"):
        out_dir = os.path.join(base_dir, 'searchlight_maps', 'probe_withinset_crosscue')
    else:
        out_dir = os.path.join(base_dir, 'searchlight_maps', 'probe_crosscue')

    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSubject {sub} | mode={mode} | classes={n_classes} | radius={radius_mm}mm")

    all_imgs, all_y, run_ids, set_ids, cue_types = load_decoding_data_both_cues(
        sub, base_dir,
        runs=(1, 2, 3),
        n_classes=n_classes,
        normalize=True,
        session_folder=session_folder,
        session_tag=session_tag
    )

    if monitor:
        _print_balance(run_ids, set_ids, cue_types, all_y)

    if mode == 'withinset_set2set':
        splits = _splits_withinset_set2set(set_ids, cue_types)
    elif mode == 'withinset_strict':
        splits = _splits_withinset_strict(run_ids, set_ids, cue_types)

    print(f"  [monitor] n_folds={len(splits)}")
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
    ap.add_argument("--n_classes", type=int, choices=[2, 3], default=2)
    ap.add_argument("--radius_mm", type=float, default=8.0)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--session_folder", default="premeal_trialwise_2mm")
    ap.add_argument("--session_tag", default="premeal")
    ap.add_argument(
        "--mode",
        choices=["withinset_set2set", "withinset_strict"],
        default="withinset_set2set"
    )
    ap.add_argument("--no_monitor", action="store_true", help="Disable monitoring prints")
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
        mode=args.mode,
        monitor=(not args.no_monitor)
    )
