"""Independently verify native SPM12 SVC outputs, then add hashes and a report.

Run after run_second_image_value_svc.m, using Python with numpy/scipy/nibabel:
    python scripts_MRI/rsa_imagewise/verify_second_image_value_svc.py
Only reads source models; writes verified result metadata and the overview.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq
from scipy.special import gammaln
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'rsa_roi/probe_imagewise_value'
MODELS = ('cue_value', 'signed_difference', 'absolute_difference')
SUBSETS = ('all', 'conditioned_only', 'preconditioned_only')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resels(mask, fwhm):
    """Count vertices, edges, faces and cubes independently of SPM's MEX."""
    edges = [np.count_nonzero(np.take(mask, range(1, mask.shape[a]), a)
                             & np.take(mask, range(mask.shape[a]-1), a))
             for a in range(3)]

    def cells(axes):
        shape = tuple(n - (a in axes) for a, n in enumerate(mask.shape))
        present = np.ones(shape, dtype=bool)
        for shifts in itertools.product((0, 1), repeat=len(axes)):
            offsets = dict(zip(axes, shifts))
            slices = tuple(slice(offsets.get(a, 0), offsets.get(a, 0) + n)
                           for a, n in enumerate(shape))
            present &= mask[slices]
        return np.count_nonzero(present)

    xy, xz, yz = [cells(axes) for axes in ((0, 1), (0, 2), (1, 2))]
    cubes = cells((0, 1, 2))
    ex, ey, ez = edges
    rx, ry, rz = 1 / np.asarray(fwhm)
    return np.array([
        mask.sum() - sum(edges) + xy + xz + yz - cubes,
        (ex-xy-xz+cubes)*rx + (ey-xy-yz+cubes)*ry + (ez-xz-yz+cubes)*rz,
        (xy-cubes)*rx*ry + (xz-cubes)*rx*rz + (yz-cubes)*ry*rz,
        cubes*rx*ry*rz,
    ])


def corrected_p(height, df, R, voxels):
    """Single-field T peak EC probability with SPM's Bonferroni minimum."""
    a = 4*np.log(2)
    b = np.exp(gammaln((df+1)/2) - gammaln(df/2))
    c = (1+height**2/df)**((1-df)/2)
    ec = np.array([t.sf(height, df), np.sqrt(a)/(2*np.pi)*c,
                   a/(2*np.pi)**1.5*c*height/np.sqrt(df/2)*b,
                   a**1.5/(2*np.pi)**2*c*((df-1)*height**2/df-1)])
    ec = np.maximum(ec, np.finfo(float).eps)
    rf = -np.expm1(-(R @ ec + np.finfo(float).eps))
    return min(rf, voxels*t.sf(height, df))


def check_model(source, rows, masks, expected_subjects):
    spm = loadmat(source/'SPM.mat', simplify_cells=True)['SPM']
    df = spm['xX']['erdf']
    subjects = [re.search(r'sub-\d+', p).group() for p in spm['xY']['P']]
    assert subjects == expected_subjects and len(set(subjects)) == 30
    assert np.all(spm['xX']['X'] == 1) and np.size(spm['xX']['X']) == 30
    np.testing.assert_allclose(df, 29, atol=1e-10)
    assert spm['xCon']['STAT'] == 'T' and spm['xCon']['c'] == 1
    ti, mi = [nib.load(source/f) for f in ('spmT_0001.nii', 'mask.nii')]
    values, group = ti.get_fdata(), mi.get_fdata()
    np.testing.assert_allclose(ti.affine, mi.affine, atol=1e-10)
    assert sha256(source/'mask.nii') == sha256(ROOT/'temp_maps/spm_value_n30/mask.nii')
    # Verify supplied sync checks against the source README when available.
    readme = (source/'README.md').read_text()
    for filename, expected in re.findall(r'SHA-256 `([^`]+)`: `([a-f0-9]{64})`', readme):
        assert sha256(source/filename) == expected, f'Sync hash mismatch: {source/filename}'
    all_resels = {}
    for row in rows:
        roi_path = ROOT / masks[row['ROI']]
        image = nib.load(roi_path)
        roi = image.get_fdata()
        assert roi.shape == values.shape == group.shape
        np.testing.assert_allclose(image.affine, ti.affine, atol=1e-10)
        native = np.isfinite(roi) & (roi != 0)
        overlap = native & np.isfinite(group) & (group != 0) & np.isfinite(values)
        S = int(overlap.sum())
        R = resels(native, spm['xVol']['FWHM'])
        all_resels[row['ROI']] = R.tolist()
        # Fortran order matches SPM/MATLAB's tie-breaking at equal maxima.
        indices = np.flatnonzero(overlap.ravel(order='F'))
        peak_values = values.ravel(order='F')[indices]
        peak = peak_values.max()
        ijk = np.unravel_index(indices[peak_values.argmax()], values.shape, order='F')
        xyz = nib.affines.apply_affine(ti.affine, ijk)
        critical = brentq(lambda z: corrected_p(z, df, R, S)-0.05, 2, 15)
        expected = dict(native_roi_voxels=int(native.sum()), analysis_overlap_voxels=S,
                        peak_T=peak, MNI_x=xyz[0], MNI_y=xyz[1], MNI_z=xyz[2],
                        peak_p_FWE_SVC=corrected_p(peak, df, R, S),
                        peak_p_uncorrected=t.sf(peak, df), critical_T_FWE_05=critical,
                        resel_volume=R[3], voxels_above_CDT_001=int((peak_values>t.isf(.001,df)).sum()),
                        voxels_above_FWE_05=int((peak_values>critical).sum()))
        for field, value in expected.items():
            np.testing.assert_allclose(float(row[field]), value, rtol=1e-8, atol=1e-9,
                                       err_msg=f'{source.name} {row["ROI"]} {field}')
    return all_resels


def main():
    reference = json.loads((BASE/'searchlight_svc_n30_ofc_v1/analysis_metadata.json').read_text())
    masks, subjects = reference['roi_masks'], reference['participant_ids']
    with (BASE/'searchlight_svc_n30_ofc_v1/peak_svc_summary.csv').open() as f:
        check_model(ROOT/'temp_maps/spm_value_n30', list(csv.DictReader(f)), masks, subjects)
    checked = []
    for model, subset in itertools.product(MODELS, SUBSETS):
        suffix = model + ('' if subset == 'all' else '_'+subset)
        source = ROOT / ('temp_maps/spm_value_n30_second_image_'+suffix)
        target = BASE / ('searchlight_svc_n30_ofc_second_image_'+suffix+'_v1')
        with (target/'peak_svc_summary.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert [r['ROI'] for r in rows] == ['Medial_OFC', 'Lateral_OFC']
        R = check_model(source, rows, masks, subjects)
        meta = json.loads((target/'analysis_metadata.json').read_text())
        assert meta['participant_ids'] == subjects and meta['value_model'] == model and meta['cue_subset'] == subset
        for roi, counts in R.items():
            np.testing.assert_allclose(meta['roi_resels'][roi], counts, atol=1e-10)
        paths = [source/name for name in ('SPM.mat','spmT_0001.nii','mask.nii','RPV.nii','ResMS.nii')]
        paths += [ROOT/path for path in masks.values()]
        paths += [Path(__file__).resolve(), Path(__file__).with_name('run_second_image_value_svc.m')]
        meta['source_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
        check = ('Independently verified all summary fields with Python (nibabel/scipy), '
                 'including native ROI topology/resels, peak coordinates, FWE p and critical T; '
                 'source hashes match supplied README checks.')
        if check not in meta['checks']:
            meta['checks'].append(check)
        checked.append((target, meta, model, subset, rows))

    overview = BASE/'searchlight_svc_n30_ofc_second_image_v1'
    with (overview/'peak_svc_summary.csv').open() as f:
        aggregate = list(csv.DictReader(f))
    assert len(aggregate) == 18
    expected_rows = [dict(value_model=model, cue_subset=subset, **row)
                     for _, _, model, subset, rows in checked for row in rows]
    assert aggregate == expected_rows
    # Only finalize provenance and the report after every independent check passes.
    for target, meta, _, _, _ in checked:
        (target/'analysis_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    lines = [
        '# Second-cue image value RSA: OFC small-volume correction', '',
        'None of the nine analyses survives peak-level SVC at FWE p < .05 in '
        'either OFC ROI. The smallest corrected p is .124252 for the signed '
        'difference model with all cues in medial OFC: T(29) = 3.813151, '
        'MNI [4, 52, -22].', '',
        'Native SPM12 positive one-sample T contrasts, n = 30, df = 29; premeal and postmeal '
        'combined. The nine models use only the second cue image.', '',
        'Peak-level FWE correction is applied separately within the established medial '
        'and lateral OFC K2 gray-matter masks. This matches `searchlight_svc_n30_ofc_v1`: '
        'full native ROI resels with each group model\'s own residual smoothness, '
        'and the finite ROI/group-mask overlap as the voxel count. No correction '
        'is applied across the two ROIs or nine analyses.', '',
        '| Value model | Cue subset | Medial OFC p FWE | Lateral OFC p FWE |',
        '|---|---|---:|---:|',
    ]
    for _, _, model, subset, rows in checked:
        ps = [float(r['peak_p_FWE_SVC']) for r in rows]
        lines.append(f'| {model} | {subset} | {ps[0]:.6f} | {ps[1]:.6f} |')
    lines += ['', 'The combined CSV includes all 18 peaks, MNI coordinates, T values, '
              'uncorrected and corrected p values, critical T thresholds, ROI voxel '
              'counts, resel volumes, and counts above voxelwise p < .001 and SVC FWE < .05.', '',
              'Per-analysis results and metadata are saved in the sibling directories '
              '`searchlight_svc_n30_ofc_second_image_<model>[_<subset>]_v1/`. '
              'Metadata includes source checksums and full resel vectors.', '',
              'Validation: reproduced both existing n30 OFC SVC rows; independently '
              'checked every numeric summary field, native ROI resels, image grids, '
              'participants, synced source hashes, and identical group masks with '
              'Python. No subject-level maps were needed; source SPM models were read only.', '',
              'Reproduction: run `scripts_MRI/rsa_imagewise/run_second_image_value_svc.m` '
              'in MATLAB with native SPM12, then '
              '`python scripts_MRI/rsa_imagewise/verify_second_image_value_svc.py`.', '',
              'These separately fitted models do not test differences between value '
              'representations. The supplement reports the medial OFC all-cue results '
              'for first-cue outcome value and second-cue signed value difference. '
              'The main manuscript is unchanged.', '']
    (overview/'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    print('PASS: baseline plus all 18 second-image SVC rows independently verified.')


if __name__ == '__main__':
    main()
