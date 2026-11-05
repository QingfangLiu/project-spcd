"""Synthetic integration checks; no participant beta images are needed."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("value_workflow", HERE / "run_searchlights.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="spcd value workflow ")
        cls.root = Path(cls.temporary.name)
        cls.base = cls.root / "data"
        cls.affine = np.diag([2., 2., 2., 1.])
        mask = cls.base / "masks/gm_0.1.nii"
        mask.parent.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), cls.affine), mask)
        cls.ratings = cls.root / "ratings.csv"
        pd.DataFrame([dict(sub=s, pre_odor1=3., pre_odor2=2., pre_clean=0.,
                           post_odor1=2.5, post_odor2=-1., post_clean=0.)
                      for s in ("sub-10", "sub-11", "sub-19", "sub-31", "sub-33", "sub-38")]
                     ).to_csv(cls.ratings, index=False)
        # No row for sub-36, matching its eligibility failure.
        for sub in ("sub-10", "sub-11", "sub-38"):
            cls.make_subject(sub)
        bad = cls.base / "GLM/postmeal_trialwise_2mm/sub-38/beta_run1_img001.nii"
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), np.nan), cls.affine), bad)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def make_subject(cls, sub):
        rng = np.random.default_rng(int(sub[-2:]))
        cues = [(1, 1), (3, 2), (5, 0), (2, 1), (4, 2), (6, 0),
                (7, 1), (9, 2), (11, 0), (8, 1), (10, 2), (12, 0)]
        for phase in workflow.PHASES:
            glm = cls.base / "GLM" / f"{phase}_trialwise_2mm" / sub
            events = cls.base / "nii" / sub / "func_PAID" / phase
            glm.mkdir(parents=True)
            events.mkdir(parents=True)
            for run in (1, 2, 3):
                records, buttons = [], []
                for trial in range(1, 25):
                    for position in (1, 2):
                        cue, odor = cues[(trial - 1 + position - 1) % 12]
                        name = f"beta_run{run}_img{2 * (trial - 1) + position:03d}.nii"
                        values = rng.normal(size=(2, 2, 2)) + odor * np.arange(8).reshape(2, 2, 2) / 8
                        nib.save(nib.Nifti1Image(values, cls.affine), glm / name)
                        records.append(dict(beta_file=name, trial=trial, img_index=position,
                                            cue_index=cue, odor_index=odor))
                    buttons.append(dict(trial=trial, trial_type="button",
                                        response_hand="left" if trial % 3 else "right"))
                pd.DataFrame(records).to_csv(glm / f"{sub}_{phase}_run-{run}_beta_index.csv", index=False)
                pd.DataFrame(buttons).to_csv(events / f"{sub}_task-{phase}_run-{run}_events.tsv", sep="\t", index=False)

    def cli(self, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / "run_searchlights.py"), *map(str, args)],
                              capture_output=True, text=True)

    def prepare(self, name, subjects=("sub-10", "sub-11")):
        output = self.root / name
        result = self.cli("prepare", "--base_dir", self.base, "--output_dir", output,
                          "--ratings_file", self.ratings, "--n_jobs", "1", "--subjects", *subjects)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return output

    def test_eligibility_and_refusal_to_overwrite(self):
        output = self.prepare("eligibility", ("sub-10", "sub-19", "sub-31", "sub-33", "sub-36", "sub-38"))
        config = json.loads((output / "analysis_config.json").read_text())
        self.assertEqual(config["subjects"], ["sub-10", "sub-38"])
        inclusion = pd.read_csv(output / "subject_inclusion.csv").set_index("sub")
        self.assertEqual(inclusion.loc["sub-31", "status"], "excluded_paper_rule")
        self.assertEqual(inclusion.loc["sub-36", "status"], "excluded_missing_ratings")
        self.assertFalse(config["center_within_run"])
        self.assertEqual(len((output / "jobs.swarm").read_text().splitlines()), 2)
        previous = (output / "analysis_config.json").read_bytes()
        repeated = self.cli("prepare", "--base_dir", self.base, "--output_dir", output)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(previous, (output / "analysis_config.json").read_bytes())

    def test_maps_reproduce_old_pipeline_and_group_gating(self):
        from rsa_imagewise.searchlight import searchlight_rsa_single_subject
        output = self.prepare("equivalence")
        result = self.cli("run", "--run_dir", output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = self.cli("check", "--run_dir", output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        ready = json.loads((output / "group_inputs.json").read_text())
        self.assertEqual(ready["subjects"], ["sub-10", "sub-11"])
        for subset in workflow.SUBSETS:
            self.assertEqual(len(ready["maps"][subset]), 2)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                expected = searchlight_rsa_single_subject(
                    sub="sub-10", base_dir=str(self.base), session_tag=workflow.PHASES,
                    radius=8., n_jobs=1, cue_subset=subset, targets=["odor_value"],
                    ratings_by_phase={p: workflow.load_subject_pleasantness_ratings(
                        "sub-10", p, str(self.ratings)) for p in workflow.PHASES})["odor_value"]
            actual = nib.load(ready["maps"][subset][0])
            np.testing.assert_allclose(actual.get_fdata(), expected.get_fdata(), atol=1e-12, rtol=0)
        # Running again must preserve every successful attempt.
        result = self.cli("run", "--run_dir", output, "--subjects", "sub-10")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list((output / "subjects/sub-10").glob("attempt-*"))), 1)
        # A changed map invalidates readiness rather than leaving a stale manifest.
        path = Path(ready["maps"]["all"][0])
        altered = nib.load(path).get_fdata() + .01
        nib.save(nib.Nifti1Image(altered, self.affine), path)
        self.assertNotEqual(self.cli("check", "--run_dir", output).returncode, 0)
        self.assertFalse((output / "group_inputs.json").exists())

    def test_input_failure_retry_preservation_and_no_partial_group(self):
        output = self.prepare("failures", ("sub-10", "sub-38"))
        result = self.cli("run", "--run_dir", output, "--subjects", "sub-38")
        self.assertNotEqual(result.returncode, 0)
        first = output / "subjects/sub-38/attempt-001"
        status = json.loads((first / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        qc = pd.read_csv(first / "input_coverage.csv")
        self.assertEqual(len(qc), 288)
        self.assertEqual(len(qc.loc[qc.status == "failed"]), 1)
        self.assertEqual(int(qc.n_nonfinite_voxels.sum()), 8)
        original = (first / "run.log").read_bytes()
        self.assertNotEqual(self.cli("run", "--run_dir", output, "--subjects", "sub-38").returncode, 0)
        self.assertEqual(original, (first / "run.log").read_bytes())
        self.assertTrue((output / "subjects/sub-38/attempt-002/status.json").exists())
        self.assertNotEqual(self.cli("check", "--run_dir", output).returncode, 0)
        self.assertFalse((output / "group_inputs.json").exists())
        report = pd.read_csv(output / "check_report.csv").set_index("sub")
        self.assertEqual(report.loc["sub-10", "reason"], "Not run")

    def test_missing_run_cannot_be_silently_skipped(self):
        path = self.base / "GLM/premeal_trialwise_2mm/sub-10/sub-10_premeal_run-1_beta_index.csv"
        contents = path.read_bytes()
        path.unlink()
        try:
            with tempfile.TemporaryDirectory(dir=self.root) as reports:
                with self.assertRaisesRegex(ValueError, "input problems"):
                    workflow.validate_inputs(self.base, "sub-10", nib.load(self.base / "masks/gm_0.1.nii"), Path(reports))
                problems = json.loads((Path(reports) / "input_problems.json").read_text())
                self.assertEqual(len(problems), 1)
                self.assertEqual(problems[0]["run"], 1)
        finally:
            path.write_bytes(contents)

    def test_prepared_inputs_cannot_change_silently(self):
        output = self.prepare("frozen")
        (output / "ratings.csv").write_text("modified\n")
        with self.assertRaisesRegex(ValueError, "Prepared input changed"):
            workflow.load_plan(output)

    def test_swarm_generator_reuses_cohort_and_initializes_every_job(self):
        conda_root = self.root / "Conda's installation"
        conda_sh = conda_root / "etc/profile.d/conda.sh"
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text('conda() { [ "$1" = activate ] || return 12; export CONDA_PREFIX="$2"; }\n')
        target_env = Path(sys.executable).parent.parent
        output = self.root / "biowulf swarm"
        result = subprocess.run([
            sys.executable, "-B", str(HERE / "make_swarm.py"),
            "--base_dir", str(self.base), "--output_dir", str(output),
            "--conda_root", str(conda_root), "--conda_env", str(target_env),
            "--ratings_file", str(self.ratings), "--subjects",
            "sub-10", "sub-11", "sub-19", "sub-31", "sub-33", "sub-36", "sub-38",
            "--n_jobs", "2", "--mem_gb", "20", "--time", "12:00:00",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        files = list(output.glob("*.swarm"))
        self.assertEqual([p.name for p in files], ["jobs.swarm"])
        commands = files[0].read_text().splitlines()
        plan = json.loads((output / "analysis_config.json").read_text())
        self.assertEqual(plan["subjects"], ["sub-10", "sub-11", "sub-38"])
        self.assertEqual(len(commands), 3)
        for sub, line in zip(plan["subjects"], commands):
            prefix, command = line.rsplit(" && ", 1)
            tokens = shlex.split(command)
            self.assertEqual(tokens[-2:], ["--subjects", sub])
            self.assertEqual(Path(tokens[0]).parent.parent, target_env)
            # Execute only the generated setup, replacing the analysis with an
            # environment probe. This also tests quoted paths containing spaces/apostrophes.
            probe = "import json, os; print(json.dumps({k:os.environ.get(k) for k in ['CONDA_PREFIX','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']}))"
            checked = subprocess.run(["bash", "-c", prefix + " && " + shlex.join(
                [sys.executable, "-B", "-c", probe])], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            env = json.loads(checked.stdout)
            self.assertEqual(env.pop("CONDA_PREFIX"), str(target_env))
            self.assertEqual(set(env.values()), {"1"})
        metadata = json.loads((output / "submission_config.json").read_text())
        submit = shlex.split(metadata["submission_command"])
        self.assertEqual(submit[submit.index("-t") + 1], "2")
        self.assertEqual(submit[submit.index("-g") + 1], "20")
        self.assertEqual(submit[submit.index("--time") + 1], "12:00:00")
        self.assertFalse(metadata["submitted"])
        self.assertFalse((output / "subjects").exists())
        self.assertTrue((output / "swarm_logs").is_dir())

    def test_swarm_stops_before_preparation_if_conda_activation_fails(self):
        conda_root = self.root / "broken Conda"
        conda_sh = conda_root / "etc/profile.d/conda.sh"
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text("conda() { return 17; }\n")
        output = self.root / "no invalid swarm"
        result = subprocess.run([
            sys.executable, "-B", str(HERE / "make_swarm.py"),
            "--base_dir", str(self.base), "--output_dir", str(output),
            "--conda_root", str(conda_root), "--conda_env", str(Path(sys.executable).parent.parent),
        ], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(output.exists())

    def test_swarm_rejects_zero_workers(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "make_swarm.py"),
                                 "--n_jobs", "0"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("n_jobs", result.stderr)


if __name__ == "__main__":
    unittest.main()
