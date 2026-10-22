#!/usr/bin/env python3
"""Synthetic tests for ROI passive-odor to cue cross-task decoding."""

from __future__ import annotations

import sys
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch
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
    def test_postmeal_uses_postmeal_originals_instead_of_premeal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp)
            samples = pd.DataFrame({
                "run": [1, 1], "trial": [2, 1], "img_index": [1, 1],
                "condition": ["A21", "A11"],
            })
            for phase, odor_labels in (("premeal", [1, 2]), ("postmeal", [2, 1])):
                pd.DataFrame({
                    "trial_type": ["cue_img1", "cue_img1"],
                    "run": [1, 1], "trial": [1, 2], "img_index": [1, 1],
                    "cue_index": [1, 3], "odor_index": odor_labels,
                }).to_csv(
                    events_dir / f"sub-14_task-{phase}_run-1_events_orig.tsv",
                    sep="\t", index=False,
                )
            labels, source = decoder._cue_labels(
                "sub-14", samples, events_dir, phase="postmeal",
            )
            np.testing.assert_array_equal(labels, [0, 1])
            self.assertEqual(source, "original_postmeal_events")
            (events_dir / "sub-14_task-postmeal_run-1_events_orig.tsv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "postmeal events"):
                decoder._cue_labels("sub-14", samples, events_dir, phase="postmeal")

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


class PostmealSubjectTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base_dir = self.root / "SPCD_data"
        self.diagnostics_dir = self.root / "diagnostics"
        self.affine = np.eye(4)
        self.voxels = np.asarray([[0, 0, 0], [1, 0, 0]])
        diagnostic_dir = self.diagnostics_dir / "sub-01"
        diagnostic_dir.mkdir(parents=True)
        for subset, condition_map in decoder.PROBE_CUE_CONDITIONS.items():
            conditions = list(condition_map.values()) * 2
            labels = decoder._labels_from_conditions(pd.Series(conditions))
            patterns = np.column_stack([2 * labels - 1, 2 * labels - 1])
            np.savez_compressed(
                diagnostic_dir / f"{subset}_roi_patterns.npz",
                raw_patterns=patterns,
                raw_roi_voxel_indices=self.voxels,
                run_ids=np.repeat([1, 2], 4), conditions=conditions,
            )
            pd.DataFrame({
                "sample_index": range(8), "sub": "sub-01",
                "run": np.repeat([1, 2], 4), "trial": list(range(1, 5)) * 2,
                "img_index": 1, "condition": conditions,
            }).to_csv(diagnostic_dir / f"{subset}_samples.csv", index=False)

        for run in (1, 2):
            event_dir = self.base_dir / "nii" / "sub-01" / "func_PAID" / "odor_scan"
            event_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"trial_type": ["clean_air", "odor_1", "odor_2"]}).to_csv(
                event_dir / f"sub-01_task-odor_run-{run}_events.tsv", sep="\t", index=False,
            )
            for trial, value in enumerate([0, -1, 1], start=1):
                _write_beta(
                    self.base_dir / "GLM" / "odor_scan_trialwise_2mm" / "sub-01"
                    / f"beta_run{run}_trial{trial:02d}.nii", value, value,
                )

        for run in (1, 2, 3):
            subject_dir = self.base_dir / "GLM" / "postmeal_trialwise_2mm" / "sub-01"
            entries = []
            cue_indices = [1, 3, 7, 9, 2, 4, 8, 10, 5, 1]
            for image, cue_index in enumerate(cue_indices, start=1):
                # Postmeal signals are deliberately reversed relative to cached premeal.
                value = 1 if cue_index in {1, 7, 2, 8} else -1
                beta_file = f"beta_run{run}_img{image}.nii"
                _write_beta(subject_dir / beta_file, value, value)
                entries.append({
                    "beta_file": beta_file, "run": run, "trial": image,
                    "img_index": 2 if image == 10 else 1,
                    "trial_type": "cue_img2" if image == 10 else "cue_img1",
                    "cue_index": cue_index,
                    # Ignored corrected odor labels must not replace cue identities.
                    "odor_index": 2 if value == 1 else 1,
                })
            pd.DataFrame(entries).to_csv(
                subject_dir / f"sub-01_postmeal_run-{run}_beta_index.csv", index=False,
            )

    def analyze(self, **kwargs) -> dict:
        return decoder._analyze_subject(
            base_dir=self.base_dir, diagnostics_dir=self.diagnostics_dir,
            sub="sub-01", roi_name="synthetic_roi", hemisphere="bilateral",
            cue_subset=kwargs.pop("cue_subset", "preconditioned_odor_only"),
            odor_runs=(1, 2), svm_c=1.0,
            sub14_original_events_dir=self.root / "unused",
            prediction_dir=self.root / "predictions", **kwargs,
        )

    def test_postmeal_uses_actual_first_cue_betas_for_both_subsets(self) -> None:
        for subset in decoder.PROBE_CUE_CONDITIONS:
            with self.subTest(cue_subset=subset):
                premeal = self.analyze(cue_subset=subset)
                postmeal = self.analyze(cue_subset=subset, phase="postmeal")
                self.assertEqual(premeal["cue_balanced_accuracy"], 1.0)
                self.assertEqual(postmeal["cue_balanced_accuracy"], 0.0)
                self.assertEqual(postmeal["passive_cv_balanced_accuracy"], 1.0)
                self.assertEqual(postmeal["n_cue_samples"], 12)
                self.assertEqual(postmeal["cue_runs"], "1,2,3")
                self.assertEqual(postmeal["missing_cue_runs"], "")
                self.assertEqual(postmeal["phase"], "postmeal")
                self.assertEqual(postmeal["cue_diagnostic_npz"], "")
                predictions = pd.read_csv(
                    self.root / "predictions" / "sub-01" / f"{subset}_cue_predictions.csv"
                )
                self.assertTrue(predictions["img_index"].eq(1).all())
                self.assertEqual(set(predictions["cue_index"]), set(decoder.PROBE_CUE_CONDITIONS[subset]))
                self.assertEqual(set(predictions["phase"]), {"postmeal"})

    def test_postmeal_qc_drops_nonfinite_voxels(self) -> None:
        beta = self.base_dir / "GLM" / "postmeal_trialwise_2mm" / "sub-01" / "beta_run1_img1.nii"
        _write_beta(beta, 1, np.nan)
        row = self.analyze(phase="postmeal")
        self.assertEqual(row["n_voxels_raw"], 2)
        self.assertEqual(row["n_voxels_used"], 1)
        self.assertEqual(row["n_voxels_dropped_nonfinite"], 1)

    def test_missing_run_is_reported_without_reusing_premeal(self) -> None:
        subject_dir = self.base_dir / "GLM" / "postmeal_trialwise_2mm" / "sub-01"
        (subject_dir / "sub-01_postmeal_run-3_beta_index.csv").unlink()
        row = self.analyze(phase="postmeal")
        self.assertEqual(row["missing_cue_runs"], "3")
        self.assertEqual(row["n_cue_samples"], 8)
        (subject_dir / "sub-01_postmeal_run-2_beta_index.csv").unlink()
        with self.assertRaisesRegex(ValueError, "at least two runs"):
            self.analyze(phase="postmeal")

    def test_mismatched_probe_image_space_fails(self) -> None:
        beta = self.base_dir / "GLM" / "postmeal_trialwise_2mm" / "sub-01" / "beta_run1_img1.nii"
        shifted = np.eye(4)
        shifted[0, 3] = 10
        _write_beta(beta, 1, 1, affine=shifted)
        with self.assertRaisesRegex(ValueError, "do not share one image space"):
            self.analyze(phase="postmeal")

    def test_missing_postmeal_beta_does_not_fall_back_to_cached_patterns(self) -> None:
        beta = self.base_dir / "GLM" / "postmeal_trialwise_2mm" / "sub-01" / "beta_run1_img1.nii"
        beta.unlink()
        with self.assertRaises(FileNotFoundError):
            self.analyze(phase="postmeal")

    def test_cli_keeps_postmeal_outputs_separate_and_summary_labeled(self) -> None:
        arguments = [
            "decoder", "--base_dir", str(self.base_dir), "--phase", "postmeal",
            "--diagnostics_dir", str(self.diagnostics_dir), "--roi_name", "synthetic_roi",
            "--cue_subsets", "conditioned_odor_only", "preconditioned_odor_only",
            "--save_predictions", "--strict",
        ]
        with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            decoder.main()
        output_dir = self.base_dir / "roi_decoding" / "probe_crosstask_odor2preconditioned_postmeal"
        result_path = output_dir / "roi-synthetic_roi_subject_results.csv"
        results = pd.read_csv(result_path)
        self.assertEqual(set(results["phase"]), {"postmeal"})
        self.assertEqual(len(results), 2)
        self.assertFalse((output_dir.parent / "probe_crosstask_odor2preconditioned").exists())
        with patch.object(sys, "argv", ["aggregate", "--input_csv", str(result_path)]), \
                contextlib.redirect_stdout(io.StringIO()):
            aggregate.main()
        summary = pd.read_csv(output_dir / "roi-synthetic_roi_group_summary.csv")
        self.assertEqual(set(summary["phase"]), {"postmeal"})
        results.loc[0, "phase"] = "premeal"
        results.to_csv(result_path, index=False)
        with patch.object(sys, "argv", ["aggregate", "--input_csv", str(result_path)]):
            with self.assertRaisesRegex(ValueError, "one valid probe phase"):
                aggregate.main()


if __name__ == "__main__":
    unittest.main()
