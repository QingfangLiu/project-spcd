"""Synthetic transfer, label provenance, input QC and complete workflow checks."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Decoding_conditioning_cross_task import run_decoding as analysis


class SyntheticData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="conditioning-decoder-test-")
        self.root = Path(self.temp.name)
        self.base, self.behavior, self.masks = [self.root / name for name in ("data", "behavior", "masks")]
        self.output, self.plots = self.root / "results", self.root / "plots"
        self.affine = np.diag([2., 2., 2., 1.])
        self.shape = (3, 3, 3)
        self.originals = self.base / "nii" / "sub-14" / "func_PAID" / "odorpred"
        self.save_image(self.base / "masks" / "gm_0.1.nii", np.ones(self.shape))
        for spec in analysis.CLUSTERS.values():
            self.save_image(self.masks / spec.mask_filename, np.ones(self.shape))
        self.add_subject("sub-01")

    def tearDown(self):
        self.temp.cleanup()

    def save_image(self, path, values, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32),
                               self.affine if affine is None else affine), path)

    def add_subject(self, sub):
        rng = np.random.default_rng(21)
        odor_pattern = rng.normal(size=self.shape)
        behavior = []
        for run in analysis.CONDITIONING_RUNS:
            index, originals = [], []
            for trial, cue in enumerate([2, 4, 8, 10, 6, 12] * 3, start=1):
                odor = 0 if cue in (6, 12) else (1 if cue in (2, 8) else 2)
                physical = 3 - odor if sub == "sub-14" and odor else odor
                responded = trial % 9 != 0
                behavior.append({
                    "Run": run, "trial": trial, "cue_idx": cue, "odor_num": physical,
                    "resp": trial % 3 if responded else np.nan,
                    "acc": int(responded and trial % 4 != 0), "rt": 0.8 if responded else np.nan,
                })
                index.append({
                    "beta_file": f"beta_run{run}_trial{trial:03d}.nii",
                    "run": run, "trial": trial, "trial_type": "cue", "cue_idx": cue,
                    "odor_num": odor, "acc": np.nan, "rt": np.nan,
                })
                originals.append({"run": run, "trial": trial, "trial_type": "cue_stim",
                                  "cue_idx": cue, "odor_num": physical})
                values = (1 if odor == 1 else -1) * odor_pattern + rng.normal(0, 0.12, self.shape) + 20 * run
                values.flat[0] = np.nan
                for window, folder in analysis.WINDOWS.items():
                    # Different offsets/scales must not confound window-specific normalization.
                    window_values = values if window == "cueEARLY" else values * 3 + 40
                    self.save_image(self.base / "GLM" / folder / sub / index[-1]["beta_file"], window_values)
            for folder in analysis.WINDOWS.values():
                pd.DataFrame(index).to_csv(self.base / "GLM" / folder / sub /
                                          f"{sub}_odorpred_run-{run}_beta_index.csv", index=False)
            if sub == "sub-14":
                self.originals.mkdir(parents=True, exist_ok=True)
                # Labels must be joined by keys, not TSV row order.
                pd.DataFrame(originals).sample(frac=1, random_state=run).to_csv(
                    self.original_path(run), sep="\t", index=False)
        destination = self.behavior / sub
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(behavior).to_csv(destination / "odor_pred_dat.csv", index=False)
        for run in analysis.PASSIVE_RUNS:
            events = []
            for trial, odor in enumerate([0, 1, 2] * 10, start=1):
                physical = 3 - odor if sub == "sub-14" and odor else odor
                events.append({"trial_type": f"odor_{physical}" if odor else "clean_air"})
                values = (1 if odor == 1 else -1) * odor_pattern + rng.normal(0, 0.12, self.shape) - 30 * run
                values.flat[0] = np.nan
                self.save_image(self.base / "GLM" / "odor_scan_trialwise_2mm" / sub /
                                f"beta_run{run}_trial{trial:02d}.nii", values)
            path = analysis.decoder._passive_event_path(self.base, sub, run)
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(events).to_csv(path, sep="\t", index=False)

    def original_path(self, run=1):
        return self.originals / f"sub-14_task-odorpred_run-{run}_events_orig.tsv"

    def load_cues(self, sub="sub-01"):
        return analysis.load_matched_cues(self.base, self.behavior, sub, list(analysis.WINDOWS), self.originals)

    def run_roi(self, sub="sub-01", cluster="cluster2"):
        frames, quality, reason = self.load_cues(sub)
        self.assertEqual(reason, "")
        spec = analysis.CLUSTERS[cluster]
        mask = self.masks / spec.mask_filename
        roi_info = analysis.conditioning.roi._load_mask(mask, self.base / "masks" / "gm_0.1.nii")
        return analysis.analyze_roi(self.base, sub, frames, quality, roi_info, spec, mask, self.output)

    def command_args(self, subjects, plots=False):
        args = ["--base_dir", str(self.base), "--behavior_dir", str(self.behavior),
                "--mask_dir", str(self.masks), "--output_dir", str(self.output),
                "--plot_dir", str(self.plots), "--subjects", *subjects]
        return args if plots else args + ["--skip_plots"]

    def run_main(self, args):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            # The deliberately strong transfer signal can produce zero-variance group scores.
            warnings.simplefilter("ignore", RuntimeWarning)
            analysis.main(args)

    def test_synthetic_identity_transfers_in_both_windows_and_keeps_beta_numbering(self):
        rows = self.run_roi()
        for window, row in rows.items():
            self.assertGreater(row["cue_balanced_accuracy"], 0.95)
            self.assertGreater(row["passive_cv_balanced_accuracy"], 0.95)
            self.assertEqual(row["n_passive_samples"], 40)
            self.assertEqual(row["n_voxels_used"], 26)
            folder = self.output / window / analysis.CLUSTERS["cluster2"].key / "predictions" / "sub-01"
            training = pd.read_csv(folder / "passive_cv_predictions.csv")
            self.assertEqual(training["beta_trial_number"].iloc[0], 2)
            self.assertFalse(training["trial_type"].eq("clean_air").any())
            cues = pd.read_csv(folder / "conditioning_predictions.csv")
            self.assertTrue(cues["behavior_acc"].eq(1).all())
            self.assertTrue(cues["behavior_rt"].gt(0).all())
            self.assertEqual(set(cues["cue_idx"]), {2, 4, 8, 10})
            self.assertEqual(set(cues["run"]), {1, 2, 3})

    def test_subject14_original_labels_align_with_passive_and_override_beta_labels(self):
        self.add_subject("sub-14")
        frames, quality, _ = self.load_cues("sub-14")
        for frame in frames.values():
            np.testing.assert_array_equal(frame["physical_odor_num"], 3 - frame["odor_num"])
            self.assertTrue(frame["physical_odor_num"].eq(frame["behavior_odor_num"]).all())
        self.assertEqual(quality["cueEARLY"]["odor_label_relation"], "global_food_label_swap")
        rows = self.run_roi("sub-14")
        self.assertGreater(rows["cueEXT5s"]["cue_balanced_accuracy"], 0.95)
        self.assertEqual(rows["cueEXT5s"]["cue_label_source"], "sub14_original_conditioning_events")

    def test_subject14_missing_originals_fail_without_fallback(self):
        self.add_subject("sub-14")
        self.original_path().unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Original sub-14 conditioning events required"):
            self.load_cues("sub-14")

    def test_subject14_wrong_original_orientation_is_rejected(self):
        self.add_subject("sub-14")
        path = self.original_path()
        original = pd.read_csv(path, sep="\t")
        original["odor_num"] = original["odor_num"].replace({1: 2, 2: 1})
        original.to_csv(path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "label conventions disagree"):
            self.load_cues("sub-14")

    def test_subject14_duplicate_original_keys_are_rejected(self):
        self.add_subject("sub-14")
        path = self.original_path()
        original = pd.read_csv(path, sep="\t")
        pd.concat([original, original.iloc[[0]]]).to_csv(path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate run/trial"):
            self.load_cues("sub-14")

    def test_unexpected_global_swap_in_other_subject_is_rejected(self):
        path = self.behavior / "sub-01" / "odor_pred_dat.csv"
        behavior = pd.read_csv(path)
        behavior["odor_num"] = behavior["odor_num"].replace({1: 2, 2: 1})
        behavior.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Unexpected beta/behavior odor-label swap"):
            self.load_cues()

    def test_common_finite_voxels_and_independent_task_window_run_normalization(self):
        rng = np.random.default_rng(100)
        runs = np.repeat([1, 2], 8)
        passive = rng.normal(size=(16, 6)) + runs[:, None] * 100
        early = rng.normal(size=(16, 6)) + runs[:, None] * 20
        extended = rng.normal(size=(16, 6)) * 7 - runs[:, None] * 90
        passive[0, 0], extended[0, 1] = np.nan, np.inf
        train, cues, finite = analysis.prepare_classification(
            passive, runs, {"cueEARLY": early, "cueEXT5s": extended},
            {"cueEARLY": runs, "cueEXT5s": runs})
        np.testing.assert_array_equal(finite, [False, False, True, True, True, True])
        self.assertEqual(train.shape[1], 4)
        for patterns in [train, *cues.values()]:
            for run in (1, 2):
                np.testing.assert_allclose(patterns[runs == run].mean(axis=0), 0, atol=1e-12)
                np.testing.assert_allclose(patterns[runs == run].std(axis=0), 1, atol=1e-12)
        self.assertFalse(np.allclose(cues["cueEARLY"], cues["cueEXT5s"]))

    def test_no_finite_voxels_fails(self):
        with self.assertRaisesRegex(ValueError, "No ROI voxels remain"):
            analysis.prepare_classification(np.ones((4, 2)), np.array([1, 1, 2, 2]),
                                            {"cueEARLY": np.full((4, 2), np.nan)},
                                            {"cueEARLY": np.array([1, 1, 2, 2])})

    def test_passive_wrong_class_count_fails(self):
        path = analysis.decoder._passive_event_path(self.base, "sub-01", 1)
        events = pd.read_csv(path, sep="\t")
        events.loc[1, "trial_type"] = "clean_air"
        events.to_csv(path, sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "10 trials per food odor"):
            self.run_roi()

    def test_passive_and_conditioning_grid_mismatch_fails(self):
        for run in analysis.PASSIVE_RUNS:
            folder = self.base / "GLM" / "odor_scan_trialwise_2mm" / "sub-01"
            for path in folder.glob(f"beta_run{run}_*.nii"):
                self.save_image(path, nib.load(path).get_fdata(), np.diag([3., 3., 3., 1.]))
        with self.assertRaisesRegex(ValueError, "not on the same grid"):
            self.run_roi()

    def test_missing_extended_beta_fails_both_windows(self):
        path = self.base / "GLM" / analysis.WINDOWS["cueEXT5s"] / "sub-01" / "beta_run2_trial001.nii"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Selected beta image missing"):
            self.load_cues()

    def test_subject35_contributes_runs_one_and_two(self):
        self.add_subject("sub-35")
        frames, quality, reason = self.load_cues("sub-35")
        self.assertEqual(reason, "")
        for window, frame in frames.items():
            self.assertEqual(set(frame["run"]), {1, 2})
            self.assertEqual(quality[window]["known_excluded_runs"], "3")

    def test_strict_failure_persists_qc_and_failed_manifest(self):
        path = self.base / "GLM" / analysis.WINDOWS["cueEXT5s"] / "sub-01" / "beta_run2_trial001.nii"
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_main(self.command_args(["sub-01"]) + ["--strict"])
        self.assertEqual(json.loads((self.output / "run_manifest.json").read_text())["run_status"], "failed")
        for window in analysis.WINDOWS:
            for spec in analysis.CLUSTERS.values():
                folder = self.output / window / spec.key
                self.assertTrue(pd.read_csv(folder / "subject_results.csv").empty)
                self.assertTrue(pd.read_csv(folder / "qc.csv")["status"].eq("failed").all())

    def test_non_strict_run_preserves_successes_and_reports_label_failure(self):
        self.add_subject("sub-14")
        self.original_path().unlink()
        with self.assertRaisesRegex(RuntimeError, "Original sub-14 conditioning events required"):
            self.run_main(self.command_args(["sub-01", "sub-14"]))
        self.assertEqual(json.loads((self.output / "run_manifest.json").read_text())["run_status"], "failed")
        folder = self.output / "cueEXT5s" / analysis.CLUSTERS["cluster2"].key
        self.assertEqual(pd.read_csv(folder / "subject_results.csv")["sub"].tolist(), ["sub-01"])
        qc = pd.read_csv(folder / "qc.csv")
        self.assertEqual(qc.loc[qc["sub"] == "sub-14", "status"].iloc[0], "failed")

    def test_initial_input_failure_replaces_previous_complete_manifest(self):
        self.output.mkdir()
        manifest = self.output / "run_manifest.json"
        manifest.write_text(json.dumps({"run_status": "complete"}))
        (self.base / "masks" / "gm_0.1.nii").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Gray-matter mask not found"):
            self.run_main(self.command_args(["sub-01"]))
        self.assertEqual(json.loads(manifest.read_text())["run_status"], "failed")

    def test_end_to_end_both_rois_windows_summaries_predictions_and_plots(self):
        self.add_subject("sub-14")
        self.add_subject("sub-35")
        # Introduce different transfer errors for a finite, checkable group t statistic.
        for sub, count in (("sub-01", 1), ("sub-35", 2)):
            for folder in analysis.WINDOWS.values():
                for trial in (1, 2)[:count]:
                    path = self.base / "GLM" / folder / sub / f"beta_run1_trial{trial:03d}.nii"
                    self.save_image(path, -nib.load(path).get_fdata())
        self.run_main(self.command_args(["sub-01", "sub-14", "sub-35", "sub-19"], plots=True) + ["--strict"])
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["run_status"], "complete")
        self.assertEqual(manifest["failures"], [])
        for window in analysis.WINDOWS:
            for spec in analysis.CLUSTERS.values():
                folder = self.output / window / spec.key
                results = pd.read_csv(folder / "subject_results.csv")
                self.assertEqual(set(results["sub"]), {"sub-01", "sub-14", "sub-35"})
                self.assertTrue(results["phase"].eq("conditioning").all())
                self.assertEqual(results.loc[results["sub"] == "sub-35", "cue_runs"].iloc[0], "1,2")
                qc = pd.read_csv(folder / "qc.csv")
                self.assertEqual(qc.loc[qc["sub"] == "sub-19", "status"].iloc[0], "excluded")
                summary = pd.read_csv(folder / "group_summary.csv")
                primary = summary.loc[summary["metric"] == "cue_balanced_accuracy"].iloc[0]
                expected = stats.ttest_1samp(results["cue_balanced_accuracy"], .5, alternative="greater")
                self.assertAlmostEqual(primary["p_greater_than_null"], expected.pvalue)
                self.assertEqual(primary["n"], 3)
                self.assertTrue((folder / "predictions" / "sub-14" / "conditioning_predictions.csv").is_file())
            for suffix in ("png", "pdf"):
                self.assertTrue((self.plots / window / f"ofc_conditioning_decoding.{suffix}").is_file())

    def test_plot_failure_sets_failed_manifest(self):
        with patch.object(analysis, "plot_results", side_effect=OSError("synthetic plot failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic plot failure"):
                self.run_main(self.command_args(["sub-01"], plots=True))
        manifest = json.loads((self.output / "run_manifest.json").read_text())
        self.assertEqual(manifest["run_status"], "failed")
        self.assertTrue(any("plotting" in reason for reason in manifest["failures"]))

    def test_dry_run_requires_no_data_or_output_and_reports_defaults(self):
        completed = subprocess.run([
            sys.executable, str(Path(analysis.__file__)), "--base_dir", str(self.root / "offline"),
            "--output_dir", str(self.output), "--dry_run",
        ], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(self.output.exists())
        config = json.loads(completed.stdout)
        self.assertEqual(set(config["windows"]), set(analysis.WINDOWS))
        self.assertEqual(config["conditioning_runs"], [1, 2, 3])
        self.assertEqual(set(config["skip_subs"]), {"sub-19", "sub-31", "sub-33", "sub-26", "sub-38"})


if __name__ == "__main__":
    unittest.main()
