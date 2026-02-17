# io.py
import os
import numpy as np
import pandas as pd
import glob
from nilearn.image import load_img, new_img_like

def load_cuewise_maps_per_run(
    sub: str,
    base_dir: str,
    session_folder: str,
    session_tag: str,
    mode: str = "both",   # "both", "img1only", "img2only"
    stat_type: str = "beta",   # "beta" or "tmap"
    runs=(1, 2, 3),
    use_resms_whitening=False,
    whitening_eps=1e-6,
):
    """
    Load cuewise GLM maps per run (one map per (run, cue_index)), using the per-run index CSV.

    Expected layout:
      <base_dir>/GLM/<session_folder>/<sub>/<mode>/
         sub-01_task-premeal_mode-both_run-01_index.csv
         beta_task-premeal_run-01_cue-01.nii
         tmap_task-premeal_run-01_cue-01.nii
         ...

    CSV columns:
      run, cue_index, beta_file, tmap_file, ...
    """
    if stat_type not in ("beta", "tmap"):
        raise ValueError(f"stat_type must be 'beta' or 'tmap', got: {stat_type}")

    # Folder that contains BOTH the maps and the per-run index CSVs
    sub_mode_dir = os.path.join(base_dir, "GLM", session_folder, sub, mode)
    if not os.path.isdir(sub_mode_dir):
        raise FileNotFoundError(f"Subject/mode folder not found: {sub_mode_dir}")

    file_col = "beta_file" if stat_type == "beta" else "tmap_file"

    # cue_index -> condition label (my task naming)
    cue_to_condition = {
        1: "A11",
        2: "B11",
        3: "A21",
        4: "B21",
        5: "C1",
        6: "D1",
        7: "A12",
        8: "B12",
        9: "A22",
        10: "B22",
        11: "C2",
        12: "D2",
    }

    # canonical order to organize files for RSA
    condition_order = ['A11', 'A21', 'C1', 'B11', 'B21', 'D1',
                       'A12', 'A22', 'C2', 'B12', 'B22', 'D2']

    run_condition_imgs = {}  # (run, condition) -> Nifti1Image

    for run in runs:
        # CSV lives in the same folder (sub_mode_dir)
        csv_glob = os.path.join(
            sub_mode_dir,
            f"{sub}_task-{session_tag}_mode-{mode}_run-{run:02d}_index.csv"
        )
        matches = glob.glob(csv_glob)
        if len(matches) == 0:
            # missing run is allowed
            print(f"[load_cuewise_maps_per_run] Missing CSV for run {run:02d}: {csv_glob} (skipping run)")
            continue
        if len(matches) > 1:
            raise RuntimeError(f"Multiple index CSVs matched for run {run:02d}: {matches}")
        csv_path = matches[0]

        df = pd.read_csv(csv_path)

        # optional: run-specific ResMS whitening
        do_whiten_this_run = False # record if whitening has been done for this run
        scale_img = None

        if use_resms_whitening:
            spm_dir = os.path.join(
                base_dir,
                "GLM",
                session_folder,
                sub,
                mode,
                f"run-{run:02d}_spm",
            )
            resms_path = os.path.join(spm_dir, "ResMS.nii")

            if os.path.exists(resms_path):
                resms_img = load_img(resms_path)
                resms_data = resms_img.get_fdata()
                scale_data = 1.0 / np.sqrt(resms_data + whitening_eps)
                scale_img = new_img_like(resms_img, scale_data)
                do_whiten_this_run = True
            else:
                print(f"[Whitening fallback] ResMS not found for run {run}, proceeding without whitening.")


        required = {"run", "cue_index", file_col}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Index CSV missing columns {sorted(missing_cols)}: {csv_path}. "
                f"Found: {list(df.columns)}"
            )

        # load each cue_index (1..12) in this run
        for _, row in df.iterrows():
            cue_idx = int(row["cue_index"])
            if cue_idx not in cue_to_condition:
                continue
            cond = cue_to_condition[cue_idx]

            fname = str(row[file_col])
            fpath = os.path.join(sub_mode_dir, fname)

            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Map file not found: {fpath}")

            key = (run, cond)
            if key in run_condition_imgs:
                raise ValueError(f"Duplicate entry for run={run}, condition={cond} in {csv_path}")

            run_condition_imgs[key] = load_img(fpath)

            # update, optionally do whitening after loading the image
            if use_resms_whitening and do_whiten_this_run:
                cond_data = cond_img.get_fdata()
                scale_data = scale_img.get_fdata()
                whitened_data = cond_data * scale_data
                cond_img = new_img_like(cond_img, whitened_data)

            run_condition_imgs[key] = cond_img

    return run_condition_imgs, condition_order

