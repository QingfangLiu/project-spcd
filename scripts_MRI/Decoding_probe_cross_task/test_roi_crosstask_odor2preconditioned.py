#!/usr/bin/env python3
"""Synthetic tests for ROI passive-odor to cue cross-task decoding."""

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

import aggregate_roi_crosstask_odor2preconditioned as aggregate
import run_roi_crosstask_odor2preconditioned as decoder


def _write_beta(
    path: Path,
    first: float,
    second: float,
    affine: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray([first, second], dtype=np.float32).reshape(2, 1, 1)
    image_affine = np.eye(4) if affine is None else affine
    nib.Nifti1Image(data, image_affine).to_filename(path)


class PassiveLoaderTests(unittest.TestCase):
    def test_passive_beta_numbers_preserve_unfiltered_event_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            sub = "sub-01"
            expected_first_voxel = []

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
                            "clean_air",
                            "odor_2",
                        ]
                    }
                ).to_csv(
                    events_dir
                    / f"{sub}_task-odor_run-{run}_events.tsv",
                    sep="\t",
                    index=False,
                )
                for trial in range(1, 5):
                    value = run * 10 + trial
                    _write_beta(
                        base_dir
                        / "GLM"
                        / "odor_scan_trialwise_2mm"
                        / sub
                        / f"beta_run{run}_trial{trial:02d}.nii",
                        value,
                        -value,
                    )
                expected_first_voxel.extend([run * 10 + 2, run * 10 + 4])

            patterns, labels, run_ids, metadata, affine = (
                decoder._load_passive_odor_patterns(
                    base_dir,
                    sub,
                    (1, 2),
                    np.asarray([[0, 0, 0], [1, 0, 0]], dtype=int),
                )
            )

            np.testing.assert_array_equal(
                patterns[:, 0],
                expected_first_voxel,
            )
            np.testing.assert_array_equal(labels, [0, 1, 0, 1])
            np.testing.assert_array_equal(run_ids, [1, 1, 2, 2])
            self.assertEqual(
                metadata["beta_trial_number"].tolist(),
                [2, 4, 2, 4],
            )
            np.testing.assert_array_equal(affine, np.eye(4))


class HemisphereVoxelTests(unittest.TestCase):
    def test_world_x_split_excludes_midline_from_unilateral_masks(self) -> None:
        voxel_indices = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
            ],
            dtype=int,
        )
        affine = np.eye(4)
        affine[0, 0] = 2.0
        affine[0, 3] = -2.0

        bilateral, midline = decoder._hemisphere_voxel_mask(
            voxel_indices,
            affine,
            "bilateral",
        )
        left, _ = decoder._hemisphere_voxel_mask(
            voxel_indices,
            affine,
            "left",
        )
        right, _ = decoder._hemisphere_voxel_mask(
            voxel_indices,
            affine,
            "right",
        )

        np.testing.assert_array_equal(bilateral, [True, True, True])
        np.testing.assert_array_equal(midline, [False, True, False])
        np.testing.assert_array_equal(left, [True, False, False])
        np.testing.assert_array_equal(right, [False, False, True])


class Sub14LabelTests(unittest.TestCase):
    def test_sub14_labels_come_from_original_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp)
            samples = pd.DataFrame(
                {
                    "run": [1, 1, 2],
                    "trial": [5, 2, 3],
                    "img_index": [1, 1, 1],
                }
            )
            run1 = pd.DataFrame(
                {
                    "trial_type": ["cue_img1", "cue_img1", "decision"],
                    "run": [1, 1, 1],
                    "trial": [2, 5, 5],
                    "img_index": [1, 1, 0],
                    "cue_index": [1, 3, 0],
                    "odor_index": [1, 2, 0],
                }
            )
            run2 = pd.DataFrame(
                {
                    "trial_type": ["cue_img1"],
                    "run": [2],
                    "trial": [3],
                    "img_index": [1],
                    "cue_index": [7],
                    "odor_index": [1],
                }
            )
            run1.to_csv(
                events_dir / "sub-14_task-premeal_run-1_events_orig.tsv",
                sep="\t",
                index=False,
            )
            run2.to_csv(
                events_dir / "sub-14_task-premeal_run-2_events_orig.tsv",
                sep="\t",
                index=False,
            )

            labels = decoder._load_sub14_original_cue_labels(
                samples,
                events_dir,
            )
            np.testing.assert_array_equal(labels, [1, 0, 0])


class DecoderMetricTests(unittest.TestCase):
    def test_linear_decoder_generalizes_matching_odor_axis(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        run_ids = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
        patterns = np.asarray(
            [
                [-3.0, -1.0],
                [-2.0, -0.5],
                [2.0, 0.5],
                [3.0, 1.0],
                [-2.5, -0.7],
                [-1.5, -0.3],
                [1.5, 0.3],
                [2.5, 0.7],
            ]
        )
        normalized = decoder._runwise_zscore(patterns, run_ids)
        metrics, _, _ = decoder._passive_leave_one_run_out(
            normalized,
            labels,
            run_ids,
            svm_c=1.0,
        )
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["auc"], 1.0)
        self.assertGreater(metrics["mean_signed_distance"], 0)

    def test_group_summary_centers_accuracy_at_half(self) -> None:
        values = np.asarray([0.6, 0.7, 0.8])
        summary = aggregate._one_sample_summary(
            values,
            metric="cue_balanced_accuracy",
            null_value=0.5,
            cue_subset="preconditioned_odor_only",
            test="one_sample",
            contrast="greater_than_chance",
        )
        self.assertAlmostEqual(summary["mean"], 0.7)
        self.assertAlmostEqual(summary["mean_minus_null"], 0.2)
        self.assertEqual(summary["n_above_null"], 3)


class EndToEndSubjectTests(unittest.TestCase):
    def test_subject_analysis_uses_saved_cue_roi_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "SPCD_data"
            diagnostics_dir = root / "diagnostics"
            subject_dir = diagnostics_dir / "sub-01"
            subject_dir.mkdir(parents=True)

            conditions = np.asarray(
                ["A11", "A12", "A21", "A22"] * 2
            )
            run_ids = np.asarray([1] * 4 + [2] * 4)
            cue_patterns = np.asarray(
                [
                    [-3.0, -1.0],
                    [-2.0, -0.5],
                    [2.0, 0.5],
                    [3.0, 1.0],
                    [-2.5, -0.7],
                    [-1.5, -0.3],
                    [1.5, 0.3],
                    [2.5, 0.7],
                ]
            )
            voxel_indices = np.asarray(
                [[0, 0, 0], [1, 0, 0]],
                dtype=int,
            )
            np.savez_compressed(
                subject_dir
                / "preconditioned_odor_only_roi_patterns.npz",
                raw_patterns=cue_patterns,
                run_ids=run_ids,
                conditions=conditions,
                raw_roi_voxel_indices=voxel_indices,
            )
            pd.DataFrame(
                {
                    "sample_index": np.arange(8),
                    "sub": ["sub-01"] * 8,
                    "run": run_ids,
                    "trial": [1, 2, 3, 4] * 2,
                    "img_index": [1] * 8,
                    "condition": conditions,
                }
            ).to_csv(
                subject_dir
                / "preconditioned_odor_only_samples.csv",
                index=False,
            )

            hemisphere_affine = np.eye(4)
            hemisphere_affine[0, 0] = 2.0
            hemisphere_affine[0, 3] = -1.0
            for run in (1, 2):
                events_dir = (
                    base_dir
                    / "nii"
                    / "sub-01"
                    / "func_PAID"
                    / "odor_scan"
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
                    events_dir
                    / f"sub-01_task-odor_run-{run}_events.tsv",
                    sep="\t",
                    index=False,
                )
                values = [
                    (0.0, 0.0),
                    (-3.0, -1.0),
                    (-2.0, -0.5),
                    (2.0, 0.5),
                    (3.0, 1.0),
                ]
                for trial, (first, second) in enumerate(values, start=1):
                    _write_beta(
                        base_dir
                        / "GLM"
                        / "odor_scan_trialwise_2mm"
                        / "sub-01"
                        / f"beta_run{run}_trial{trial:02d}.nii",
                        first,
                        second,
                        affine=hemisphere_affine,
                    )

            prediction_dir = root / "predictions"
            row = decoder._analyze_subject(
                base_dir=base_dir,
                diagnostics_dir=diagnostics_dir,
                sub="sub-01",
                roi_name="synthetic_roi",
                hemisphere="bilateral",
                cue_subset="preconditioned_odor_only",
                odor_runs=(1, 2),
                svm_c=1.0,
                sub14_original_events_dir=root / "unused",
                prediction_dir=prediction_dir,
            )

            self.assertEqual(row["cue_balanced_accuracy"], 1.0)
            self.assertEqual(
                row["passive_cv_balanced_accuracy"],
                1.0,
            )
            self.assertEqual(row["n_voxels_used"], 2)
            self.assertEqual(row["hemisphere"], "bilateral")
            self.assertEqual(row["n_voxels_bilateral_raw"], 2)
            self.assertEqual(row["n_voxels_raw"], 2)
            self.assertTrue(
                (
                    prediction_dir
                    / "sub-01"
                    / "preconditioned_odor_only_cue_predictions.csv"
                ).is_file()
            )

            for hemisphere in ("left", "right"):
                unilateral_row = decoder._analyze_subject(
                    base_dir=base_dir,
                    diagnostics_dir=diagnostics_dir,
                    sub="sub-01",
                    roi_name=f"synthetic_roi_{hemisphere}",
                    hemisphere=hemisphere,
                    cue_subset="preconditioned_odor_only",
                    odor_runs=(1, 2),
                    svm_c=1.0,
                    sub14_original_events_dir=root / "unused",
                    prediction_dir=None,
                )
                self.assertEqual(
                    unilateral_row["cue_balanced_accuracy"],
                    1.0,
                )
                self.assertEqual(
                    unilateral_row["passive_cv_balanced_accuracy"],
                    1.0,
                )
                self.assertEqual(unilateral_row["n_voxels_raw"], 1)
                self.assertEqual(unilateral_row["n_voxels_used"], 1)
                self.assertEqual(
                    unilateral_row["n_voxels_excluded_by_hemisphere"],
                    1,
                )


if __name__ == "__main__":
    unittest.main()
