"""Check the small preparation/Swarm adaptation against the original pipeline."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="value rsa ")
        cls.root = Path(cls.temp.name)
        cls.base = cls.root / "mini data"
        cls.affine = np.diag([2., 2., 2., 1.])
        (cls.base / "masks").mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), cls.affine), cls.base / "masks/gm_0.1.nii")
        cls.ratings = cls.root / "ratings.csv"
        pd.DataFrame([dict(sub=s, pre_odor1=3., pre_odor2=2., pre_clean=0.,
                           post_odor1=2.5, post_odor2=-1., post_clean=0.)
            for s in ("sub-10", "sub-11", "sub-19", "sub-31", "sub-33")]
            ).to_csv(cls.ratings, index=False)
        cues = [(1, 1), (3, 2), (5, 0), (2, 1), (4, 2), (6, 0),
                (7, 1), (9, 2), (11, 0), (8, 1), (10, 2), (12, 0)]
        for sub in ("sub-10", "sub-11"):
            rng = np.random.default_rng(int(sub[-2:]))
            for phase in ("premeal", "postmeal"):
                glm = cls.base / "GLM" / f"{phase}_trialwise_2mm" / sub
                events = cls.base / "nii" / sub / "func_PAID" / phase
                glm.mkdir(parents=True)
                events.mkdir(parents=True)
                for run in (1, 2, 3):
                    rows, buttons = [], []
                    for trial in range(1, 25):
                        for position in (1, 2):
                            cue, odor = cues[(trial - 1 + position - 1) % 12]
                            name = f"beta_run{run}_img{2 * (trial - 1) + position:03d}.nii"
                            nib.save(nib.Nifti1Image(rng.normal(size=(2, 2, 2)), cls.affine), glm / name)
                            rows.append(dict(beta_file=name, trial=trial, img_index=position,
                                             cue_index=cue, odor_index=odor))
                        buttons.append(dict(trial=trial, trial_type="button",
                                            response_hand="left" if trial % 3 else "right"))
                    pd.DataFrame(rows).to_csv(glm / f"{sub}_{phase}_run-{run}_beta_index.csv", index=False)
                    pd.DataFrame(buttons).to_csv(events / f"{sub}_task-{phase}_run-{run}_events.tsv", sep="\t", index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def cli(self, script, *args):
        return subprocess.run([sys.executable, "-B", str(HERE / script), *map(str, args)],
                              capture_output=True, text=True)

    def test_swarm_selection_and_direct_original_commands(self):
        output = self.root / "selection/value_rsa.swarm"
        result = self.cli("make_swarm.py", "--ratings_file", self.ratings, "--out", output,
                          "--subjects", "sub-10", "sub-11", "sub-19", "sub-31", "sub-33", "sub-36")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = output.read_text().splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual((output.parent / "subjects.txt").read_text().splitlines(), ["sub-10", "sub-11"])
        report = pd.read_csv(output.parent / "subject_selection.csv").set_index("sub")
        self.assertEqual(report.loc["sub-31", "status"], "excluded_paper_rule")
        self.assertEqual(report.loc["sub-36", "status"], "excluded_missing_ratings")
        seen = set()
        for line in lines:
            parsed = subprocess.run(["bash", "-n"], input=line, capture_output=True, text=True)
            self.assertEqual(parsed.returncode, 0, parsed.stderr)
            tokens = shlex.split(line.rsplit(" && ", 1)[1])
            self.assertEqual(tokens[:4], ["python", "-B", "-m", "rsa_imagewise.run_rsa"])
            options = dict(zip(tokens[4::2], tokens[5::2]))
            self.assertEqual(options["--session"], "both")
            self.assertEqual(options["--target"], "odor_value")
            self.assertEqual(options["--base_dir"], "/data/liuq13/SPCD_mini")
            self.assertEqual(options["--button_scope"], "second_image_only")
            self.assertNotIn("--center_within_run", tokens)
            seen.add((options["--sub"], options["--cue_subset"]))
        self.assertEqual(seen, {(s, c) for s in ("sub-10", "sub-11")
                               for c in ("all", "conditioned_only", "preconditioned_only")})
        self.assertIn("-t 4", result.stdout)

    def test_preparation_reuses_package_and_reports_missing_events(self):
        stage = self.root / "event stage"
        missing = self.base / "nii/sub-11/func_PAID/postmeal/sub-11_task-postmeal_run-3_events.tsv"
        contents = missing.read_bytes()
        missing.unlink()
        try:
            result = self.cli("prepare_rsa_subset.py", "--full_data_dir", self.base,
                              "--subset_dir", stage, "--ratings_file", self.ratings,
                              "--subjects", "sub-10", "sub-11", "sub-31", "sub-36")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            source = HERE.parent / "rsa_imagewise"
            for name in ("run_rsa.py", "searchlight.py", "models.py", "ratings.py"):
                self.assertEqual((stage / "rsa_imagewise" / name).read_bytes(), (source / name).read_bytes())
            self.assertEqual(len(list((stage / "nii").rglob("*.tsv"))), 11)
            failures = pd.read_csv(stage / "missing_events.csv")
            self.assertEqual(failures[["sub", "phase", "run"]].to_dict("records"),
                             [dict(sub="sub-11", phase="postmeal", run=3)])
            self.assertFalse((stage / "GLM").exists())
            self.assertFalse((stage / "masks").exists())
        finally:
            missing.write_bytes(contents)

    def test_staged_swarm_executes_all_three_original_searchlights(self):
        from rsa_imagewise.searchlight import searchlight_rsa_single_subject
        from rsa_imagewise.ratings import load_subject_pleasantness_ratings
        stage = self.root / "execution stage"
        result = self.cli("prepare_rsa_subset.py", "--full_data_dir", self.base,
                          "--subset_dir", stage, "--ratings_file", self.ratings, "--subjects", "sub-10")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        shutil.copytree(stage, self.base, dirs_exist_ok=True)  # Simulate the rsync merge.
        conda_root = self.root / "Conda's installation"
        conda_sh = conda_root / "etc/profile.d/conda.sh"
        conda_sh.parent.mkdir(parents=True)
        conda_sh.write_text('conda() { [ "$1" = activate ] || return 17; export PATH="$2/bin:$PATH"; }\n')
        output = stage / "value_rsa.swarm"
        result = self.cli("make_swarm.py", "--ratings_file", self.ratings, "--out", output,
                          "--subjects", "sub-10", "--base_dir", self.base, "--n_jobs", "1",
                          "--conda_root", conda_root, "--conda_env", Path(sys.executable).parent.parent)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for line in output.read_text().splitlines():
            result = subprocess.run(["bash", "-c", line], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            options = shlex.split(line.rsplit(" && ", 1)[1])
            subset = options[options.index("--cue_subset") + 1]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                expected = searchlight_rsa_single_subject(
                    sub="sub-10", base_dir=str(self.base), session_tag=("premeal", "postmeal"),
                    radius=8., n_jobs=1, cue_subset=subset, targets=["odor_value"],
                    ratings_by_phase={p: load_subject_pleasantness_ratings("sub-10", p, str(self.ratings))
                                      for p in ("premeal", "postmeal")})["odor_value"]
            actual = nib.load(self.base / "rsa_maps/probe_imagewise_value_updated/both" /
                              f"sub-10_model-odor_value_subset-{subset}.nii.gz")
            np.testing.assert_allclose(actual.get_fdata(), expected.get_fdata(), atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
