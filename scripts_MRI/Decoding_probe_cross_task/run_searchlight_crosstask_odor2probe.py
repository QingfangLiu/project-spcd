# run_searchlight_crosstask_odor2probe.py
import os
import argparse
import numpy as np
from nilearn.decoding import SearchLight
from nilearn.image import load_img, concat_imgs, new_img_like
from sklearn.svm import SVC

from odor_training_utils import load_decoding_data_odor
from probe_decoding_utils import load_decoding_data_cue  # your existing loader

def run_one_subject(sub, base_dir, mask_img_path,
                    cue_type, n_classes=2,
                    radius_mm=8.0, n_jobs=-1,
                    session_folder="premeal_trialwise_2mm", session_tag="premeal"):
    """
    Cross-task searchlight:
      - TRAIN: Day-1 odor betas (runs 1–2)
      - TEST : Day-2 pre-meal probe betas (runs 1–3), A or B separately
      - Normalization: fit μ,σ on TRAIN only; apply to both train/test
    """
    out_dir = os.path.join(base_dir, "searchlight_maps", "probe_crosstask")
    os.makedirs(out_dir, exist_ok=True)
    tag_stem = f"{sub}_session-{session_tag}_mode-crosstask_cue-{cue_type}_classes-{n_classes}_radius-{int(radius_mm)}mm"
    out_path = os.path.join(out_dir, f"{tag_stem}.nii.gz")
    if os.path.exists(out_path):
        print(f"⏭️ Skipping {sub} (exists): {out_path}")
        return

    # ---- Load TRAIN (odor) ----
    odor_imgs, y_train = load_decoding_data_odor(
        sub=sub, base_dir=base_dir, runs=(1, 2), n_classes=n_classes
    )

    # ---- Load TEST (probe pre-meal, A or B) WITHOUT its own normalization ----
    probe_imgs, y_test, _, _ = load_decoding_data_cue(
        sub=sub, base_dir=base_dir, runs=(1, 2, 3),
        cue_type=cue_type, n_classes=n_classes,
        normalize=True,                         # dataset-internal z-scoring for PROBE
        session_folder=session_folder, session_tag=session_tag
    )

    X = concat_imgs([odor_imgs, probe_imgs])
    y = np.concatenate([y_train, y_test])

    n_train = len(y_train)
    n_total = len(y)
    train_idx = np.arange(n_train)
    test_idx  = np.arange(n_train, n_total)
    splits = [(train_idx, test_idx)]

    # ---- Searchlight ----
    mask_img = load_img(mask_img_path)
    sl = SearchLight(
        mask_img=mask_img,
        radius=radius_mm,
        estimator=SVC(kernel="linear", C=1.0),
        n_jobs=n_jobs,
        cv=splits,
        scoring="accuracy",
        verbose=1
    )
    print(f"\nSubject {sub} | cue={cue_type} | classes={n_classes} | radius={radius_mm}mm | TRAIN=odor | TEST=probe({session_tag})")
    sl.fit(X, y)
    out_img = new_img_like(mask_img, sl.scores_.astype(np.float32))
    out_img.to_filename(out_path)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True, help="e.g., sub-01")
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--mask_img", required=True)
    parser.add_argument("--cue_type", choices=["A", "B"], default="A")
    parser.add_argument("--n_classes", type=int, choices=[2, 3], default=2)
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
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        session_folder=args.session_folder,
        session_tag=args.session_tag
    )
