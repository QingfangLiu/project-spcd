#!/usr/bin/env python3
"""Synthetic tests for functional-mask odor-to-probe ROI decoding."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_roi_crosstask_odor2preconditioned_functional_mask as decoder


def _write_image(
    path: Path,
    values: tuple[float, float],
    affine: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(values, dtype=np.float32).reshape(2, 1, 1)
    nib.Nifti1Image(data, affine).to_filename(path)


class FunctionalMaskLoaderTests(unittest.TestCase):
    def test_analyze_pair_uses_only_finite_positive_voxels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_path = Path(tmp) / "functional_mask.hdr"
            data = np.asarray([1.0, 0.0, np.nan], dtype=np.float32).reshape(
                3, 1, 1
            )
            nib.Nifti1Pair(data, np.eye(4)).to_filename(mask_path)

            indices, shape, affine = decoder._load_functional_mask(mask_path)

            np.testing.assert_array_equal(indices, [[0, 0, 0]])
            self.assertEqual(shape, (3, 1, 1))
            np.testing.assert_array_equal(affine, np.eye(4))

    def test_beta_space_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            beta_path = Path(tmp) / "beta.nii"
            shifted = np.eye(4)
            shifted[0, 3] = 2.0
            _write_image(beta_path, (1.0, 2.0), shifted)

            with self.assertRaisesRegex(ValueError, "not in the same space"):
                decoder._extract_mask_values(
                    beta_path,
                    np.asarray([[0, 0, 0]], dtype=int),
                    (2, 1, 1),
                    np.eye(4),
                )


class FunctionalMaskEndToEndTests(unittest.TestCase):
    def test_subject_analysis_decodes_preconditioned_first_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "SPCD_data"
            sub = "sub-01"
            affine = np.eye(4)
            affine[0, 0] = 2.0
            affine[0, 3] = -2.0

            mask_path = root / "functional_mask.hdr"
            nib.Nifti1Pair(
                np.ones((2, 1, 1), dtype=np.float32),
                affine,
            ).to_filename(mask_path)
            voxel_indices, mask_shape, mask_affine = (
                decoder._load_functional_mask(mask_path)
            )

            for run in (1, 2):
                events_dir = (
                    base_dir / "nii" / sub / "func_PAID" / "odor_scan"
                )
                events_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "trial_type": [
                            "clean_air",
                            "odor_1",
                            "odor_1",
                            "odor_2",
                            "odor_2",
                        ]
                    }
                ).to_csv(
                    events_dir / f"{sub}_task-odor_run-{run}_events.tsv",
                    sep="\t",
                    index=False,
                )
                passive_values = [
                    (0.0, 0.0),
                    (-3.0, -1.0),
                    (-2.0, -0.5),
                    (2.0, 0.5),
                    (3.0, 1.0),
                ]
                for trial, values in enumerate(passive_values, start=1):
                    _write_image(
                        base_dir
                        / "GLM"
                        / "odor_scan_trialwise_2mm"
                        / sub
                        / f"beta_run{run}_trial{trial:02d}.nii",
                        values,
                        affine,
                    )

                probe_dir = (
                    base_dir / "GLM" / "premeal_trialwise_2mm" / sub
                )
                probe_rows = []
                samples = [
                    ("pre1.nii", 1, 1, "cue_img1", (-3.0, -1.0)),
                    ("pre3.nii", 2, 3, "cue_img1", (3.0, 1.0)),
                    ("pre7.nii", 3, 7, "cue_img1", (-2.0, -0.5)),
                    ("pre9.nii", 4, 9, "cue_img1", (2.0, 0.5)),
                    ("conditioned.nii", 5, 2, "cue_img1", (20.0, 20.0)),
                    ("second.nii", 1, 1, "cue_img2", (20.0, 20.0)),
                ]
                for name, trial, cue_index, trial_type, values in samples:
                    _write_image(probe_dir / name, values, affine)
                    probe_rows.append(
                        {
                            "beta_file": name,
                            "run": run,
                            "set": 1,
                            "trial": trial,
                            "img_index": 1 if trial_type == "cue_img1" else 2,
                            "trial_type": trial_type,
                            "cue_index": cue_index,
                            "odor_index": 1 if cue_index in (1, 7) else 2,
                        }
                    )
                pd.DataFrame(probe_rows).to_csv(
                    probe_dir / f"{sub}_premeal_run-{run}_beta_index.csv",
                    index=False,
                )

            prediction_dir = root / "predictions"
            row = decoder._analyze_subject(
                base_dir=base_dir,
                sub=sub,
                roi_name="functional_roi",
                mask_path=mask_path,
                voxel_indices=voxel_indices,
                mask_shape=mask_shape,
                mask_affine=mask_affine,
                odor_runs=(1, 2),
                probe_runs=(1, 2),
                svm_c=1.0,
                sub14_original_events_dir=root / "unused",
                prediction_dir=prediction_dir,
            )

            self.assertEqual(row["cue_subset"], decoder.CUE_SUBSET)
            self.assertEqual(row["analysis_scope"], "exploratory_post_selection")
            self.assertEqual(row["cue_balanced_accuracy"], 1.0)
            self.assertEqual(row["passive_cv_balanced_accuracy"], 1.0)
            self.assertEqual(row["n_cue_samples"], 8)
            self.assertEqual(row["n_voxels_mask"], 2)
            self.assertEqual(row["n_voxels_used"], 2)
            self.assertEqual(row["n_voxels_left"], 1)
            self.assertEqual(row["n_voxels_midline"], 1)
            self.assertTrue(
                (
                    prediction_dir
                    / sub
                    / "preconditioned_odor_only_cue_predictions.csv"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
