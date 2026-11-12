"""Synthetic input and end-to-end checks; no experimental MRI data required."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_rsa as analysis


class SyntheticData(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="conditioning-rsa-test-")
        self.root = Path(self.temp.name)
        self.base = self.root / "data"
        self.behavior = self.root / "behavior"
        self.masks = self.root / "masks"
        self.masks.mkdir()
        (self.base / "masks").mkdir(parents=True)
        self.affine = np.diag([2., 2., 2., 1.])
        self.shape = (3, 3, 3)
        self.save_image(self.base / "masks" / "gm_0.1.nii", np.ones(self.shape))
        for spec in analysis.CLUSTERS.values():
            self.save_image(self.masks / spec.mask_filename, np.ones(self.shape))
        self.add_subject("sub-01")

    def tearDown(self):
        self.temp.cleanup()

    def save_image(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), self.affine), path)

    def add_subject(self, sub, global_swap=False):
        rng = np.random.default_rng(21)
        odor_pattern = rng.normal(size=self.shape)
        behavior = []
        for run in (1, 2, 3):
            index = []
            for trial, cue in enumerate([2, 4, 8, 10, 6, 12] * 3, start=1):
                odor = 0 if cue in (6, 12) else (1 if cue in (2, 8) else 2)
                behavior_odor = 3 - odor if global_swap and odor else odor
                # Every fourth food trial is incorrect, every ninth trial missed.
                responded = trial % 9 != 0
                correct = responded and trial % 4 != 0
                behavior.append({
                    "Run": run, "trial": trial, "cue_idx": cue,
                    "odor_num": behavior_odor, "resp": trial % 3 if responded else np.nan,
                    "acc": int(correct), "rt": 0.8 if responded else np.nan,
                })
                index.append({
                    "beta_file": f"beta_run{run}_trial{trial:03d}.nii",
                    "run": run, "trial": trial, "trial_type": "cue",
                    "cue_idx": cue, "odor_num": odor,
                    # Deliberately reproduce the missing beta-index behavior fields.
                    "acc": np.nan, "rt": np.nan,
                })
                signal = odor_pattern * (1 if odor == 1 else -1)
                values = signal + rng.normal(scale=0.12, size=self.shape) + run
                # A nonfinite voxel must be removed before centering/RDM estimation.
                values.flat[0] = np.nan
                for window, folder in analysis.WINDOWS.items():
                    destination = self.base / "GLM" / folder / sub
                    self.save_image(destination / index[-1]["beta_file"], values)
            for folder in analysis.WINDOWS.values():
                destination = self.base / "GLM" / folder / sub
                pd.DataFrame(index).to_csv(destination / f"{sub}_odorpred_run-{run}_beta_index.csv", index=False)
        destination = self.behavior / sub
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(behavior).to_csv(destination / "odor_pred_dat.csv", index=False)

    def index_path(self, window="cueEARLY", sub="sub-01", run=2):
        return self.base / "GLM" / analysis.WINDOWS[window] / sub / f"{sub}_odorpred_run-{run}_beta_index.csv"

    def test_labels_and_correct_trials_come_from_keyed_behavior(self):
        frames, quality = analysis.matched_samples(
            self.base, self.behavior, "sub-01", list(analysis.WINDOWS), (2, 3),
        )
        frame = frames["cueEARLY"]
        self.assertTrue(frame["behavior_acc"].eq(1).all())
        self.assertTrue(frame["behavior_resp"].notna().all())
        self.assertEqual(set(frame["odor_num"]), {1, 2})
        self.assertEqual(set(frame["cue_idx"]), {2, 4, 8, 10})
        self.assertEqual(set(frame["run"]), {2, 3})
        self.assertEqual(quality["cueEARLY"]["odor_label_relation"], "identical")
        self.assertEqual(frame[["run", "trial"]].values.tolist(), frames["cueEXT5s"][["run", "trial"]].values.tolist())

    def test_subject14_global_label_swap_preserves_model(self):
        self.add_subject("sub-14", global_swap=True)
        frames, quality = analysis.matched_samples(self.base, self.behavior, "sub-14", list(analysis.WINDOWS), (2, 3))
        self.assertEqual(quality["cueEARLY"]["odor_label_relation"], "global_food_label_swap")
        frame = frames["cueEARLY"]
        beta_odors = frame["odor_num"].to_numpy()
        behavior_odors = frame["behavior_odor_num"].to_numpy()
        np.testing.assert_array_equal(beta_odors[:, None] != beta_odors, behavior_odors[:, None] != behavior_odors)

    def test_responded_trial_with_missing_rt_is_flagged_and_excluded(self):
        path = self.behavior / "sub-01" / "odor_pred_dat.csv"
        behavior = pd.read_csv(path)
        behavior.loc[(behavior["Run"] == 2) & (behavior["trial"] == 1), "rt"] = np.nan
        behavior.to_csv(path, index=False)
        frames, quality = analysis.matched_samples(self.base, self.behavior, "sub-01", list(analysis.WINDOWS), (2, 3))
        self.assertEqual(quality["cueEARLY"]["n_food_invalid_rt"], 1)
        self.assertFalse(((frames["cueEARLY"]["run"] == 2) & (frames["cueEARLY"]["trial"] == 1)).any())

    def test_mixed_label_errors_are_rejected(self):
        path = self.index_path()
        index = pd.read_csv(path)
        index.loc[0, "odor_num"] = 2
        index.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "changes odor labels"):
            analysis.matched_samples(self.base, self.behavior, "sub-01", list(analysis.WINDOWS), (2, 3))

    def test_duplicate_trial_keys_are_rejected(self):
        path = self.behavior / "sub-01" / "odor_pred_dat.csv"
        behavior = pd.read_csv(path)
        pd.concat([behavior, behavior.iloc[[0]]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate behavioral"):
            analysis.load_behavior(self.behavior, "sub-01")

    def test_missing_behavioral_match_is_rejected(self):
        path = self.behavior / "sub-01" / "odor_pred_dat.csv"
        behavior = pd.read_csv(path)
        behavior.loc[~((behavior["Run"] == 2) & (behavior["trial"] == 1))].to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "no behavioral match"):
            analysis.matched_samples(self.base, self.behavior, "sub-01", list(analysis.WINDOWS), (2, 3))

    def test_windows_use_common_trials_and_record_loss(self):
        path = self.index_path("cueEXT5s")
        index = pd.read_csv(path)
        index.loc[index["trial"] != 1].to_csv(path, index=False)
        frames, quality = analysis.matched_samples(self.base, self.behavior, "sub-01", list(analysis.WINDOWS), (2, 3))
        self.assertEqual(quality["cueEARLY"]["n_unmatched_window_trials"], 1)
        self.assertEqual(quality["cueEXT5s"]["n_unmatched_window_trials"], 0)
        self.assertFalse(((frames["cueEARLY"]["run"] == 2) & (frames["cueEARLY"]["trial"] == 1)).any())

    def test_missing_selected_beta_fails_before_either_window_is_analyzed(self):
        path = self.base / "GLM" / analysis.WINDOWS["cueEXT5s"] / "sub-01" / "beta_run2_trial001.nii"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Selected beta image missing"):
            analysis.matched_samples(self.base, self.behavior, "sub-01", list(analysis.WINDOWS), (2, 3))

    def test_subject35_uses_two_runs_only_in_sensitivity(self):
        self.add_subject("sub-35")
        primary, _ = analysis.matched_samples(self.base, self.behavior, "sub-35", list(analysis.WINDOWS), (2, 3))
        self.assertEqual(analysis.eligibility(primary["cueEARLY"], (2, 3), True), "fewer_than_two_usable_runs")
        sensitivity, _ = analysis.matched_samples(self.base, self.behavior, "sub-35", list(analysis.WINDOWS), (1, 2, 3))
        self.assertEqual(set(sensitivity["cueEARLY"]["run"]), {1, 2})
        self.assertEqual(analysis.eligibility(sensitivity["cueEARLY"], (1, 2, 3), False), "")

    def test_synthetic_identity_signal_and_cross_set_mask(self):
        frames, _ = analysis.matched_samples(self.base, self.behavior, "sub-01", list(analysis.WINDOWS), (2, 3))
        frame = frames["cueEARLY"]
        roi_img, roi_mask, *_ = analysis.roi._load_mask(next(self.masks.glob("*cluster1*")))
        raw = analysis.roi._load_roi_patterns(frame.to_dict("records"), roi_img, roi_mask)
        values, details = analysis.analyze_patterns(frame, raw)
        self.assertEqual(details[3]["n_voxels_dropped_nonfinite"], 1)
        for scope in analysis.PAIR_SCOPES:
            self.assertGreater(values[scope][analysis.SPEARMAN_COL], 0.7)
        tri = np.tril_indices(len(frame), k=-1)
        mask = details[-1]["cross_run_cross_set"]
        i, j = tri[0][mask], tri[1][mask]
        self.assertTrue((frame["run"].to_numpy()[i] != frame["run"].to_numpy()[j]).all())
        self.assertTrue((frame["cue_set"].to_numpy()[i] != frame["cue_set"].to_numpy()[j]).all())
        self.assertTrue((frame["cue_idx"].to_numpy()[i] != frame["cue_idx"].to_numpy()[j]).all())

    def test_constant_sample_patterns_are_rejected(self):
        frame = pd.DataFrame({"run": [2, 2, 3, 3], "odor_num": [1, 2, 1, 2], "cue_set": [1, 2, 1, 2]})
        with self.assertRaisesRegex(ValueError, "constant"):
            analysis.analyze_patterns(frame, np.ones((4, 8)))

    def test_holm_correction(self):
        np.testing.assert_allclose(analysis.holm_adjust([0.04, 0.01]), [0.04, 0.02])
        # A missing second-window result must not reduce the planned family size.
        np.testing.assert_allclose(analysis.holm_adjust([0.04, 1.0]), [0.08, 1.0])

    def test_end_to_end_both_windows_rois_run_selections_and_diagnostics(self):
        self.add_subject("sub-14", global_swap=True)
        self.add_subject("sub-35")
        output, plots = self.root / "results", self.root / "plots"
        command = [
            sys.executable, str(Path(analysis.__file__)), "--base_dir", str(self.base),
            "--behavior_dir", str(self.behavior), "--mask_dir", str(self.masks),
            "--repo_output_dir", str(output), "--plot_dir", str(plots),
            "--subjects", "sub-01", "sub-14", "sub-35", "sub-19", "--save_diagnostics", "--strict",
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        manifest = json.loads((output / "run_manifest.json").read_text())
        self.assertEqual(set(manifest["windows"]), set(analysis.WINDOWS))
        for window in analysis.WINDOWS:
            for selection in analysis.ANALYSES:
                expected = {"sub-01", "sub-14"} | ({"sub-35"} if selection.startswith("sensitivity") else set())
                for spec in analysis.CLUSTERS.values():
                    folder = output / window / selection / spec.key
                    results = pd.read_csv(folder / "subject_results.csv")
                    self.assertEqual(set(results["sub"]), expected)
                    self.assertEqual(set(results["pair_scope"]), set(analysis.PAIR_SCOPES))
                    self.assertEqual(len(results), len(expected) * 2)
                    qc = pd.read_csv(folder / "qc.csv")
                    self.assertEqual(qc.loc[qc["sub"] == "sub-19", "status"].iloc[0], "excluded")
                    self.assertTrue((plots / window / selection / f"{spec.key}.png").is_file())
                    self.assertTrue((plots / window / selection / f"{spec.key}.pdf").is_file())
                    summary = pd.read_csv(folder / "group_summary.csv")
                    if spec.key.startswith("cluster2") and selection.startswith("primary"):
                        primary = summary.loc[(summary["metric"] == analysis.SPEARMAN_COL) & (summary["pair_scope"] == "cross_run")]
                        self.assertTrue(primary["p_primary_lateral_holm"].notna().all())
                    diag = self.base / "rsa_roi" / "conditioning_odor_identity" / "diagnostics" / window / selection / spec.key / "sub-01"
                    self.assertTrue((diag / "patterns_and_rdms.npz").is_file())
                    self.assertTrue((diag / "samples.csv").is_file())
                    pairs = pd.read_csv(diag / "rdm_pairs.csv")
                    self.assertTrue(pairs["cross_run"].all())

    def test_dry_run_does_not_require_external_data_or_write_results(self):
        output = self.root / "dry-results"
        completed = subprocess.run([
            sys.executable, str(Path(analysis.__file__)), "--base_dir", str(self.root / "offline"),
            "--repo_output_dir", str(output), "--dry_run",
        ], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(output.exists())
        self.assertIn("primary_runs2-3", completed.stdout)


if __name__ == "__main__":
    unittest.main()
