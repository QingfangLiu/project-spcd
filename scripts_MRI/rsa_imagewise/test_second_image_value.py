"""Run: python -B -m unittest discover -s scripts_MRI/rsa_imagewise -p 'test_*.py'."""

from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsa_imagewise.diagnostics import check_second_image_design
from rsa_imagewise.io_imagewise import _pair_trial_cues, build_imagewise_patterns_for_sessions
from rsa_imagewise.models import (
    VALUE_MODELS, analysis_subfolder, create_odor_value_rdm_by_phase,
    create_second_image_value_rdms, validate_analysis_options,
)
from rsa_imagewise.prepare_rsa_subset import copy_package_files
from rsa_imagewise.searchlight import searchlight_rsa_single_subject

PACKAGE = Path(__file__).resolve().parent


class ValueDefinitionsTest(unittest.TestCase):
    def test_signed_and_absolute_are_different_hypotheses(self):
        variables, rdms = create_second_image_value_rdms(
            ['A11_img1', 'A21_img1', 'C1_img1'],
            ['A21_img2', 'A11_img2', 'A11_img2'], ['premeal'] * 3,
            {'premeal': {'odor1': 2., 'odor2': 6., 'clean': 0.}}, normalize=False)
        np.testing.assert_array_equal(variables['cue_value'], [6, 2, 2])
        np.testing.assert_array_equal(variables['signed_difference'], [4, -4, 2])
        np.testing.assert_array_equal(variables['absolute_difference'], [4, 4, 2])
        self.assertEqual(rdms['cue_value'][0, 1], 4)
        self.assertEqual(rdms['signed_difference'][0, 1], 8)
        self.assertEqual(rdms['absolute_difference'][0, 1], 0)
        for rdm in rdms.values():
            np.testing.assert_array_equal(rdm, rdm.T)
            np.testing.assert_array_equal(np.diag(rdm), 0)

    def test_joint_normalization_before_trial_expansion(self):
        ratings = {'premeal': {'odor1': 2., 'odor2': 6., 'clean': 0.},
                   'postmeal': {'odor1': 2., 'odor2': -2., 'clean': 0.}}
        first = ['A11_img1', 'A11_img1', 'A11_img1']
        second = ['A21_img2'] * 3
        phases = ['premeal', 'postmeal', 'premeal']
        variables, rdms = create_second_image_value_rdms(first, second, phases, ratings)
        flat = np.array([2, 6, 0, 2, -2, 0], dtype=float)
        expected = (np.array([6., -2., 6.]) - flat.mean()) / flat.std()
        np.testing.assert_allclose(variables['cue_value'], expected)
        np.testing.assert_allclose(variables['signed_difference'], np.array([4, -4, 4]) / flat.std())
        np.testing.assert_allclose(rdms['cue_value'], np.abs(expected[:, None] - expected))
        # The existing cue-value definition is exactly the same special case.
        np.testing.assert_array_equal(rdms['cue_value'], create_odor_value_rdm_by_phase(second, phases, ratings))

    def test_bad_ratings_and_invalid_modes_fail(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            create_second_image_value_rdms(['A11_img1'], ['A21_img2'], ['premeal'],
                {'premeal': {'odor1': 2., 'odor2': np.nan, 'clean': 0.}})
        with self.assertRaisesRegex(ValueError, 'require'):
            validate_analysis_options('both', 'signed_difference')
        self.assertEqual(analysis_subfolder(), 'probe_imagewise')
        self.assertEqual(len({analysis_subfolder('second', name) for name in VALUE_MODELS}), 3)

    def test_pair_validation_does_not_rely_on_adjacent_rows(self):
        rows = pd.DataFrame([
            dict(trial=2, img_index=2, cue_index=1, odor_index=1),
            dict(trial=1, img_index=1, cue_index=1, odor_index=1),
            dict(trial=2, img_index=1, cue_index=3, odor_index=2),
            dict(trial=1, img_index=2, cue_index=3, odor_index=2)])
        pairs = _pair_trial_cues(rows, 'test')
        self.assertEqual(pairs[1], {1: 'A11', 2: 'A21'})
        self.assertEqual(pairs[2], {1: 'A21', 2: 'A11'})
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            _pair_trial_cues(rows.iloc[:-1], 'missing second')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            _pair_trial_cues(pd.concat([rows, rows.iloc[:1]]), 'duplicate')
        rows.loc[0, 'trial'] = np.nan
        with self.assertRaisesRegex(ValueError, 'Noninteger'):
            _pair_trial_cues(rows, 'missing trial')

    def test_constant_controls_dropped_and_collinearity_rejected(self):
        x = np.array([0., 1., 3., 2.])
        value = abs(x[:, None] - x)
        active, _, info = check_second_image_design(
            {'odor_value': value, 'button_presence': np.zeros((4, 4))}, [1, 1, 2, 2])
        self.assertEqual(list(active), ['odor_value'])
        self.assertEqual(info['dropped_constant_regressors'], ['button_presence'])
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            check_second_image_design({'odor_value': value, 'image_identity': value}, [1, 1, 2, 2])
        with self.assertRaisesRegex(ValueError, 'no variation'):
            check_second_image_design({'odor_value': np.zeros((4, 4))}, [1, 1, 2, 2])


class SecondImagePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='spcd-second-image-test-')
        cls.root = Path(cls.temp.name)
        cls.affine = np.diag([2., 2., 2., 1.])
        (cls.root / 'masks').mkdir()
        nib.save(nib.Nifti1Image(np.ones((3, 3, 3), np.uint8), cls.affine), cls.root/'masks/gm_0.1.nii')
        cls.ratings = {'premeal': {'odor1': 4., 'odor2': 3., 'clean': 0.},
                       'postmeal': {'odor1': 4., 'odor2': -1., 'clean': 0.}}
        rng = np.random.default_rng(834)
        cls.expected = {}
        for phase in cls.ratings:
            for run in (1, 2):
                folder = cls.root / 'GLM' / f'{phase}_trialwise_2mm' / 'sub-01'
                folder.mkdir(parents=True, exist_ok=True)
                rows, buttons = [], []
                trial = 0
                for cues in [(1, 3, 5), (7, 9, 11), (2, 4, 6), (8, 10, 12)]:
                    for a, b in itertools.combinations(cues, 2):
                        for first, second in [(a, b), (b, a)]:
                            trial += 1
                            for position, cue in enumerate((first, second), start=1):
                                odor = {1: 1, 2: 1, 3: 2, 4: 2, 5: 0, 6: 0,
                                        7: 1, 8: 1, 9: 2, 10: 2, 11: 0, 12: 0}[cue]
                                name = f'run-{run}_trial-{trial}_img-{position}.nii.gz'
                                data = rng.normal(size=(3, 3, 3)).astype(np.float32)
                                nib.save(nib.Nifti1Image(data, cls.affine), folder/name)
                                rows.append(dict(beta_file=name, trial=trial, img_index=position,
                                                 cue_index=cue, odor_index=odor))
                            cls.expected[(phase, run, trial)] = (first, second)
                            button = 'none' if trial % 7 == 0 else rng.choice(['left', 'right'])
                            buttons.append(dict(trial=trial, trial_type='button', response_hand=button))
                # CSV order deliberately intermixes trials and cue positions.
                pd.DataFrame(rows).sample(frac=1, random_state=run).to_csv(
                    folder/f'sub-01_{phase}_run-{run}_beta_index.csv', index=False)
                events = cls.root / 'nii/sub-01/func_PAID' / phase
                events.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(buttons).to_csv(events/f'sub-01_task-{phase}_run-{run}_events.tsv',
                                             sep='\t', index=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, **options):
        with redirect_stdout(io.StringIO()):
            return build_imagewise_patterns_for_sessions('sub-01', str(self.root),
                                                         ('premeal', 'postmeal'), **options)

    def test_loader_pairs_by_phase_run_trial_and_preserves_default(self):
        both = self.load()
        explicit = self.load(image_scope='both')
        self.assertEqual(len(both), 5)
        self.assertEqual(len(both[4]), 192)
        for a, b in zip(both[:4], explicit[:4]):
            np.testing.assert_array_equal(a, b)
        second = self.load(image_scope='second', return_trial_metadata=True)
        runs, buttons, labels, phases, images, metadata = second
        self.assertEqual(len(images), 96)
        self.assertEqual(set(runs), {1, 2, 101, 102})
        self.assertTrue(all(label.endswith('_img2') for label in labels))
        np.testing.assert_array_equal(buttons, metadata.button_label)
        np.testing.assert_array_equal(phases, metadata.session)
        for row in metadata.itertuples():
            from rsa_imagewise.io_imagewise import condition_map
            expected_first, expected_second = self.expected[(row.session, row.run_within_phase, row.trial)]
            # Labels must be attached to the actual trial, not CSV adjacency.
            lookup = {cue: label for (cue, _), label in condition_map.items()}
            self.assertEqual(row.first_label, lookup[expected_first] + '_img1')
            self.assertEqual(row.second_label, lookup[expected_second] + '_img2')
        for subset, prefixes in [('conditioned_only', ('B', 'D')), ('preconditioned_only', ('A', 'C'))]:
            loaded = self.load(image_scope='second', return_trial_metadata=True, cue_subset=subset)
            self.assertEqual(len(loaded[4]), 48)
            self.assertTrue(all(s.startswith(prefixes) for s in loaded[5].first_label))
            self.assertTrue(all(s.startswith(prefixes) for s in loaded[5].second_label))

    def test_actual_searchlights_match_direct_ols_for_all_three_versions(self):
        for value_model in VALUE_MODELS:
            with self.subTest(value_model=value_model), redirect_stdout(io.StringIO()):
                out = self.root/'outputs'/value_model
                maps = searchlight_rsa_single_subject(
                    sub='sub-01', base_dir=str(self.root), session_tag=('premeal', 'postmeal'),
                    radius=4., n_jobs=1, image_scope='second', value_model=value_model,
                    center_within_run=True, ratings_by_phase=self.ratings, output_dir=str(out))
                self.assertEqual(set(maps), {'odor_value'})
                diag = out/'diagnostics/sub-01/subset-all'
                metadata = json.loads((diag/'analysis_metadata.json').read_text())
                self.assertEqual(metadata['status'], 'complete')
                self.assertIn('button_side', metadata['included_regressors'])
                self.assertIn('button_presence', metadata['included_regressors'])
                self.assertEqual(metadata['n_between_run_pairs'], 3456)
                samples = pd.read_csv(diag/'samples.csv')
                loaded = self.load(image_scope='second', return_trial_metadata=True)
                patterns = np.array([im.get_fdata().ravel() for im in loaded[4]])
                for run in np.unique(loaded[0]):
                    selected = loaded[0] == run
                    patterns[selected] -= patterns[selected].mean(axis=0)
                neural = 1 - np.corrcoef(patterns)
                lower = np.tril_indices(len(patterns), k=-1)
                cross = loaded[0][lower[0]] != loaded[0][lower[1]]
                with np.load(diag/'model_rdms.npz') as rdms:
                    columns = []
                    for name in metadata['included_regressors']:
                        rdm = rdms[value_model if name == 'odor_value' else name]
                        columns.append(rdm[lower][cross])
                    design = np.column_stack([np.ones(cross.sum()), *columns])
                    beta = np.linalg.lstsq(design, neural[lower][cross], rcond=None)[0]
                    index = metadata['included_regressors'].index('odor_value') + 1
                actual = maps['odor_value'].get_fdata()[1, 1, 1]
                self.assertAlmostEqual(actual, beta[index], places=10)
                self.assertGreater(abs(actual), 1e-7)
                np.testing.assert_allclose(samples.signed_difference,
                                           samples.cue_value - samples.first_cue_value)
                self.assertTrue((out/'sub-01_model-odor_value_subset-all.nii.gz').is_file())

    def test_all_responded_drops_only_constant_button_presence(self):
        loaded = list(self.load(image_scope='second', return_trial_metadata=True))
        loaded[1] = np.array(['left', 'right'] * 48)
        loaded[5] = loaded[5].copy()
        loaded[5]['button_label'] = loaded[1]
        out = self.root/'all_responded'
        with patch('rsa_imagewise.searchlight.build_imagewise_patterns_for_sessions',
                   return_value=tuple(loaded)), redirect_stdout(io.StringIO()):
            searchlight_rsa_single_subject('sub-01', str(self.root), ('premeal', 'postmeal'),
                radius=4., n_jobs=1, image_scope='second', value_model='cue_value',
                ratings_by_phase=self.ratings, output_dir=str(out))
        meta = json.loads((out/'diagnostics/sub-01/subset-all/analysis_metadata.json').read_text())
        self.assertEqual(meta['dropped_constant_regressors'], ['button_presence'])
        self.assertIn('button_side', meta['included_regressors'])

    def test_cli_saves_to_variant_specific_default_folder(self):
        ratings = self.root/'cli_ratings.csv'
        pd.DataFrame([dict(sub='sub-01', pre_odor1=4., pre_odor2=3., pre_clean=0.,
                           post_odor1=4., post_odor2=-1., post_clean=0.)]).to_csv(ratings,index=False)
        result = subprocess.run([sys.executable, '-B', '-m', 'rsa_imagewise.run_rsa',
            '--sub', 'sub-01', '--session', 'both', '--base_dir', str(self.root),
            '--ratings_file', str(ratings), '--image_scope', 'second',
            '--value_model', 'absolute_difference', '--center_within_run',
            '--radius', '4', '--n_jobs', '1'], cwd=PACKAGE.parent,
            capture_output=True, text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        out=self.root/'rsa_maps/probe_imagewise_second_image/absolute_difference/both'
        self.assertTrue((out/'sub-01_model-odor_value_subset-all.nii.gz').is_file())
        meta=json.loads((out/'diagnostics/sub-01/subset-all/analysis_metadata.json').read_text())
        self.assertEqual(meta['requested_targets'], ['odor_value'])
        self.assertEqual(meta['value_model'], 'absolute_difference')

    def test_subset_designs_drop_cue_type(self):
        from rsa_imagewise.models import create_image_identity_rdm, create_button_side_rdm, create_button_presence_rdm
        for subset in ('conditioned_only', 'preconditioned_only'):
            runs, buttons, labels, phases, _, samples = self.load(
                image_scope='second', return_trial_metadata=True, cue_subset=subset)
            _, rdms = create_second_image_value_rdms(samples.first_label, labels, phases, self.ratings)
            for name in VALUE_MODELS:
                active, _, info = check_second_image_design({
                    'odor_value': rdms[name], 'image_identity': create_image_identity_rdm(labels),
                    'button_presence': create_button_presence_rdm(buttons),
                    'button_side': create_button_side_rdm(buttons)}, runs)
                self.assertNotIn('cue_type', active)
                self.assertEqual(info['design_rank'], info['design_columns'])

    def test_swarm_generates_three_versions_and_excludes_paper_subjects(self):
        ratings = self.root/'compiled ratings.csv'
        rows=[]
        for sub in ['sub-01', 'sub-10', 'sub-19', 'sub-31', 'sub-33', 'sub-36']:
            row={'sub': sub}
            row.update(dict(pre_odor1=4., pre_odor2=3., pre_clean=0.,
                            post_odor1=4., post_odor2=-1., post_clean=0.))
            if sub=='sub-36':
                row['post_clean']=np.nan
            rows.append(row)
        pd.DataFrame(rows).to_csv(ratings,index=False)
        output=self.root/'jobs.swarm'
        result=subprocess.run([sys.executable, '-B', str(PACKAGE/'make_swarm_imagewise.py'),
            '--base_dir', str(self.root), '--out', str(output), '--ratings_file', str(ratings),
            '--image_scope', 'second', '--sessions', 'both', '--center_within_run',
            '--subjects', *[r['sub'] for r in rows]], capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        jobs=[shlex.split(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(jobs),6)
        self.assertEqual({j[j.index('--sub')+1] for j in jobs},{'sub-01','sub-10'})
        self.assertEqual({j[j.index('--value_model')+1] for j in jobs},set(VALUE_MODELS))
        for job in jobs:
            self.assertEqual(job[job.index('--target')+1],'odor_value')
            self.assertIn(str(ratings),job)
            self.assertIn('--center_within_run',job)

    def test_subset_package_contains_imported_runtime_helpers(self):
        with redirect_stdout(io.StringIO()):
            copy_package_files(PACKAGE,self.root/'staging')
        for filename in ['ratings.py','diagnostics.py','models.py','run_rsa.py']:
            self.assertTrue((self.root/'staging/rsa_imagewise'/filename).is_file())


if __name__=='__main__':
    unittest.main()
