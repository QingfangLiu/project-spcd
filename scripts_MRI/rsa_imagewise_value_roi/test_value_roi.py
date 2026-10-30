"""Synthetic image-data tests; run with python -B -m unittest discover here."""

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

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from rsa_imagewise.estimator import SingleModelRSA
from rsa_imagewise_value_roi.aggregate_roi_rsa import ROI_NAMES, summarize_subject_betas
from rsa_imagewise_value_roi.run_roi_rsa import BARTRA_SOURCE, OFC_MASKS, build_samples


class ValueROITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="spcd-value-roi-")
        cls.root = Path(cls.temp.name)
        cls.project, cls.data = cls.root / "project", cls.root / "data"
        cls.affine = np.diag([4., 4., 4., 1.])
        cls.affine[:3, 3] = [-8., 36., -14.]
        cls.shape = (5, 5, 5)
        gm_path = cls.data / "masks/gm_0.1.nii"
        gm_path.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.ones(cls.shape, dtype=np.uint8), cls.affine), gm_path)
        bartra_path = cls.project / BARTRA_SOURCE
        bartra_path.parent.mkdir(parents=True)
        bartra_path.symlink_to(SCRIPTS.parent / BARTRA_SOURCE)
        ofc = cls.project / "masks/OFC_clusterMap_K2_gm0p1"
        ofc.mkdir(parents=True)
        for idx, filename in enumerate(OFC_MASKS):
            values = np.zeros(cls.shape, dtype=np.uint8)
            if idx == 0:
                values[:2] = 1
            else:
                values[3:] = 1
            nib.save(nib.Nifti1Image(values, cls.affine), ofc / filename)
        for sub in ("sub-01", "sub-02", "sub-03", "sub-19", "sub-31", "sub-33", "sub-36", "sub-38"):
            folder = cls.project / "beh_processed_subs" / sub
            folder.mkdir(parents=True)
            if sub != "sub-36":
                pd.DataFrame(dict(odor_num=[0, 1, 2], pre_mean=[0., 3., 2.],
                                  post_mean=[0., 3., -1.])).to_csv(
                                      folder / "pleasantness_summary.csv", index=False)
        rng = np.random.default_rng(1234)
        for sub in ("sub-01", "sub-02", "sub-03"):
            for phase in ("premeal", "postmeal"):
                for run in (1, 2):
                    folder = cls.data / "GLM" / f"{phase}_trialwise_2mm" / sub
                    events = cls.data / "nii" / sub / "func_PAID" / phase
                    folder.mkdir(parents=True, exist_ok=True)
                    events.mkdir(parents=True, exist_ok=True)
                    rows = []
                    for cue in range(1, 13):
                        odor = {1: 1, 2: 1, 3: 2, 4: 2, 5: 0, 6: 0,
                                7: 1, 8: 1, 9: 2, 10: 2, 11: 0, 12: 0}[cue]
                        for pos in (1, 2):
                            name = f"run-{run}_cue-{cue}_img-{pos}.nii.gz"
                            values = rng.normal(size=cls.shape).astype(np.float32)
                            # Exercise voxel QC and spatial-pattern extraction.
                            values[0, 0, 0] = np.nan
                            if sub == "sub-03":
                                # This participant fails only in the third ROI.
                                values[3:] = 2.
                            nib.save(nib.Nifti1Image(values, cls.affine), folder / name)
                            rows.append(dict(beta_file=name, trial=cue, img_index=pos,
                                             cue_index=cue, odor_index=odor))
                    pd.DataFrame(rows).to_csv(
                        folder / f"{sub}_{phase}_run-{run}_beta_index.csv", index=False)
                    pd.DataFrame(dict(trial=range(1, 13), trial_type="button",
                                      response_hand=["left", "right"] * 6)).to_csv(
                        events / f"{sub}_task-{phase}_run-{run}_events.tsv", sep="\t", index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_pipeline(self, name, *extra):
        output = self.root / name
        command = [sys.executable, "-B", str(SCRIPTS / "rsa_imagewise_value_roi/run_roi_rsa.py"),
                   "--base_dir", str(self.data), "--project_root", str(self.project),
                   "--output_dir", str(output), *extra]
        return output, subprocess.run(command, capture_output=True, text=True)

    def test_complete_pipeline_and_searchlight_equivalence(self):
        output, result = self.run_pipeline(
            "complete", "--save_diagnostics", "--subjects", "sub-01", "sub-02", "sub-19",
            "sub-31", "sub-33", "sub-36")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        table = pd.read_csv(output / "subject_betas.csv")
        self.assertEqual(len(table), 6)
        summary = pd.read_csv(output / "group_summary.csv")
        self.assertTrue((summary["n"] == 2).all())
        self.assertEqual(set(summary.roi), set(ROI_NAMES))
        inclusion = pd.read_csv(output / "subject_inclusion.csv").set_index("sub")
        for sub in ("sub-19", "sub-31", "sub-33"):
            self.assertEqual(inclusion.loc[sub, "status"], "excluded_paper_rule")
        self.assertEqual(inclusion.loc["sub-36", "status"], "excluded_missing_ratings")
        self.assertTrue(table.qc_incomplete_runs.all())
        self.assertTrue((table.n_samples == 96).all())
        self.assertTrue((table.n_pairs == 3456).all())
        self.assertTrue((table.loc[table.roi == ROI_NAMES[1], "n_voxels_dropped_nonfinite"] == 1).all())
        config = json.loads((output / "analysis_config.json").read_text())
        self.assertFalse(config["center_within_run"])
        for row in table.itertuples():
            folder = output / "diagnostics" / row.sub / row.roi
            diagnostic = np.load(folder / "rsa.npz")
            samples = pd.read_csv(folder / "samples.csv")
            self.assertTrue((samples.loc[samples.img_index == 1, "button_label"] == "none").all())
            self.assertEqual(set(samples.loc[samples.session == "postmeal", "run"]), {101, 102})
            models = {name[6:]: diagnostic[name] for name in diagnostic.files if name.startswith("model_")}
            estimate = SingleModelRSA(models, "odor_value", diagnostic["run_ids"]).score(
                diagnostic["analysis_patterns"])
            self.assertAlmostEqual(row.odor_value_beta, estimate, places=12)
            # Independent least-squares calculation on the exact retained pairs.
            tri = np.tril_indices(len(samples), -1)
            keep = diagnostic["pair_mask"]
            design = np.column_stack([np.ones(int(keep.sum()))] +
                                      [models[m][tri][keep] for m in models])
            expected = np.linalg.lstsq(design, diagnostic["neural_rdm"][tri][keep], rcond=None)[0][1]
            self.assertAlmostEqual(row.odor_value_beta, expected, places=12)
        # A second invocation must not replace existing results.
        _, repeated = self.run_pipeline("complete", "--subjects", "sub-01", "sub-02")
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("not empty", repeated.stderr)

    def test_failure_requires_explicit_partial_group(self):
        output, result = self.run_pipeline("failure", "--subjects", "sub-01", "sub-02", "sub-38")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((output / "group_summary.csv").exists())
        inclusion = pd.read_csv(output / "subject_inclusion.csv").set_index("sub")
        self.assertEqual(inclusion.loc["sub-38", "status"], "failed")
        output, result = self.run_pipeline("partial", "--allow_partial", "--subjects",
                                           "sub-01", "sub-02", "sub-38")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((pd.read_csv(output / "group_summary.csv").n == 2).all())

    def test_centered_and_wide_ratings(self):
        ratings_path = self.root / "ratings.csv"
        pd.DataFrame(dict(sub=["sub-01", "sub-02"], pre_clean=0., pre_odor1=3., pre_odor2=2.,
                          post_clean=0., post_odor1=3., post_odor2=-1.)).to_csv(ratings_path, index=False)
        output, result = self.run_pipeline("centered", "--subjects", "sub-01", "sub-02",
                                           "--center_within_run", "--save_diagnostics",
                                           "--ratings_file", str(ratings_path))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        diagnostic = np.load(output / "diagnostics/sub-01/Bartra_vmPFC/rsa.npz")
        for run in np.unique(diagnostic["run_ids"]):
            mean = diagnostic["analysis_patterns"][diagnostic["run_ids"] == run].mean(axis=0)
            np.testing.assert_allclose(mean, 0., atol=1e-14)

    def test_single_roi_failure_removes_subject_from_every_roi(self):
        output, result = self.run_pipeline("single_roi_failure", "--subjects",
                                           "sub-01", "sub-02", "sub-03", "--allow_partial",
                                           "--save_diagnostics")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        table = pd.read_csv(output / "subject_betas.csv")
        self.assertEqual(set(table["sub"]), {"sub-01", "sub-02"})
        self.assertEqual(len(table), 6)
        inclusion = pd.read_csv(output / "subject_inclusion.csv").set_index("sub")
        self.assertIn("constant spatial pattern", inclusion.loc["sub-03", "reason"])
        # Successful earlier ROI diagnostics may remain, but no coefficient rows
        # from this participant can enter any group test.
        self.assertTrue((output / "diagnostics/sub-03" / ROI_NAMES[0] / "rsa.npz").exists())

    def test_group_rejects_mismatched_participants(self):
        rows = [dict(sub=sub, roi=roi, odor_value_beta=i + j * 0.1)
                for i, sub in enumerate(("sub-01", "sub-02")) for j, roi in enumerate(ROI_NAMES)]
        with self.assertRaisesRegex(ValueError, "all three ROIs"):
            summarize_subject_betas(pd.DataFrame(rows[:-1]))

    def test_present_broken_run_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            folder = base / "GLM/premeal_trialwise_2mm/sub-01"
            folder.mkdir(parents=True)
            pd.DataFrame(dict(beta_file=["missing.nii"], trial=[1], img_index=[1],
                              cue_index=[1], odor_index=[1])).to_csv(
                folder / "sub-01_premeal_run-1_beta_index.csv", index=False)
            with self.assertRaisesRegex(FileNotFoundError, "Events TSV"):
                build_samples(base, "sub-01")

    def run_export(self, name, *extra):
        output = self.root / name
        command = [sys.executable, "-B", str(SCRIPTS / "rsa_imagewise_value_roi/export_roi_patterns.py"),
                   "--base_dir", str(self.data), "--project_root", str(self.project),
                   "--output_dir", str(output), *extra]
        return output, subprocess.run(command, capture_output=True, text=True)

    def make_coverage_subject(self, sub, missing_image=False, wrong_grid=False):
        # No ratings: extraction must still work and flag value-analysis eligibility.
        (self.project / "beh_processed_subs" / sub).mkdir(parents=True, exist_ok=True)
        for phase in ("premeal", "postmeal"):
            folder = self.data / "GLM" / f"{phase}_trialwise_2mm" / sub
            events = self.data / "nii" / sub / "func_PAID" / phase
            folder.mkdir(parents=True)
            events.mkdir(parents=True)
            rows = []
            for position in (1, 2):
                name = f"img-{position}.nii.gz"
                values = np.arange(np.prod(self.shape), dtype=float).reshape(self.shape)
                if phase == "postmeal" and position == 2:
                    values[:] = np.nan
                    if wrong_grid:
                        values = values[:-1]
                if not (missing_image and phase == "postmeal" and position == 2):
                    nib.save(nib.Nifti1Image(values, self.affine), folder / name)
                rows.append(dict(beta_file=name, trial=1, img_index=position,
                                 cue_index=1 if position == 1 else 3,
                                 odor_index=1 if position == 1 else 2))
            pd.DataFrame(rows).to_csv(folder / f"{sub}_{phase}_run-1_beta_index.csv", index=False)
            pd.DataFrame(dict(trial=[1], trial_type=["button"], response_hand=["left"])).to_csv(
                events / f"{sub}_task-{phase}_run-1_events.tsv", sep="\t", index=False)

    def test_export_preserves_raw_patterns_metadata_and_coordinates(self):
        output, result = self.run_export("raw_export", "--subjects", "sub-01", "sub-03", "sub-31")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = pd.read_csv(output / "subject_export.csv").set_index("sub")
        self.assertEqual(report.loc["sub-31", "status"], "excluded_paper_rule")
        self.assertFalse((output / "sub-31").exists())
        # sub-03 fails the original RSA due to a constant pattern, but all raw ROIs export.
        self.assertEqual(report.loc["sub-03", "n_rois_exported"], 3)
        ratings = pd.read_csv(output / "ratings.csv")
        self.assertEqual(len(ratings), 12)
        self.assertTrue(ratings.rating_available.all())
        for roi in ROI_NAMES:
            samples = pd.read_csv(output / "sub-01/samples.csv")
            with np.load(output / "sub-01" / f"roi-{roi}_patterns.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["raw_patterns"].shape[0], 96)
                self.assertEqual(raw["raw_patterns"].dtype, np.float64)
                np.testing.assert_array_equal(raw["sample_index"], samples.sample_index)
                np.testing.assert_array_equal(raw["phase_labels"], samples.session)
                np.testing.assert_array_equal(raw["run_ids"], samples.run)
                np.testing.assert_allclose(raw["voxel_mni"],
                                           nib.affines.apply_affine(raw["affine"], raw["voxel_ijk"]))
                for index in (0, 23, 95):
                    image = nib.load(samples.iloc[index].beta_path)
                    expected = np.asarray(image.dataobj)[tuple(raw["voxel_ijk"].T)]
                    np.testing.assert_array_equal(raw["raw_patterns"][index], expected)
                # Read every archive member without permitting object/pickle data.
                for key in raw.files:
                    self.assertNotEqual(raw[key].dtype.kind, "O")
        first = samples.loc[samples.img_index == 1]
        self.assertTrue((first.button_label == "none").all())
        self.assertIn("left", set(first.trial_button_label))
        self.assertEqual(set(samples.odor_group), {"odor1", "odor2", "clean"})
        self.assertEqual(set(samples.cue_set), {1, 2})
        _, repeated = self.run_export("raw_export", "--subjects", "sub-01")
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("not empty", repeated.stderr)

    def test_export_nonfinite_roi_and_missing_ratings_keep_all_rois(self):
        self.make_coverage_subject("sub-04")
        output, result = self.run_export("nonfinite_export", "--subjects", "sub-04")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = pd.read_csv(output / "subject_export.csv")
        self.assertFalse(bool(report.iloc[0].ratings_complete))
        self.assertEqual(report.iloc[0].n_rois_exported, 3)
        coverage = pd.read_csv(output / "coverage_summary.csv")
        all_samples = coverage.loc[coverage.scope == "all"]
        self.assertTrue((all_samples.n_voxels_finite_all_samples == 0).all())
        self.assertTrue((all_samples.n_all_nonfinite_samples == 1).all())
        first_only = coverage.loc[(coverage.scope == "image") & (coverage.img_index == "1")]
        self.assertEqual(len(first_only), 3)
        self.assertTrue((first_only.n_voxels_finite_all_samples == first_only.n_voxels).all())
        for roi in ROI_NAMES:
            with np.load(output / "sub-04" / f"roi-{roi}_patterns.npz") as raw:
                self.assertTrue(np.isnan(raw["raw_patterns"][-1]).all())
                self.assertTrue((raw["beta_read_status"] == "ok").all())

    def test_export_input_errors_preserve_alignment_and_continue(self):
        self.make_coverage_subject("sub-05", missing_image=True)
        self.make_coverage_subject("sub-06", wrong_grid=True)
        output, result = self.run_export("input_error_export", "--subjects", "sub-05", "sub-06", "sub-02")
        self.assertNotEqual(result.returncode, 0)
        report = pd.read_csv(output / "subject_export.csv").set_index("sub")
        self.assertEqual(report.loc["sub-02", "status"], "exported")
        for sub in ("sub-05", "sub-06"):
            self.assertEqual(report.loc[sub, "status"], "exported_with_input_errors")
            self.assertEqual(report.loc[sub, "n_rois_exported"], 3)
            with np.load(output / sub / f"roi-{ROI_NAMES[0]}_patterns.npz") as raw:
                self.assertEqual(len(raw["raw_patterns"]), 4)
                self.assertEqual(raw["beta_read_status"][-1], "error")
                self.assertTrue(np.isnan(raw["raw_patterns"][-1]).all())
        issues = pd.read_csv(output / "input_issues.csv")
        self.assertEqual(set(issues.kind), {"beta_image"})
        config = json.loads((output / "export_config.json").read_text())
        self.assertEqual(set(config["input_error_subjects"]), {"sub-05", "sub-06"})


if __name__ == "__main__":
    unittest.main()
