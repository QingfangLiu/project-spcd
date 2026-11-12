"""Design checks and provenance for the second-image value analyses."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .estimator import SingleModelRSA
from .rsa_stats import vectorize_lower_triangle


def check_second_image_design(model_rdms, run_ids):
    """Drop constant controls and reject an unidentifiable regression."""
    pair_mask = SingleModelRSA._make_cross_run_pair_mask(run_ids)
    if not pair_mask.any():
        raise ValueError("Second-image RSA needs samples from at least two runs")
    active, dropped = {}, []
    for name, rdm in model_rdms.items():
        vector = vectorize_lower_triangle(rdm, pair_mask)
        if not np.isfinite(vector).all():
            raise ValueError(f"Nonfinite model RDM: {name}")
        if np.allclose(vector, vector[0], rtol=0, atol=1e-12):
            if name == "odor_value":
                raise ValueError("Value RDM has no variation on between-run pairs")
            dropped.append(name)
        else:
            active[name] = rdm
    vectors = np.column_stack([vectorize_lower_triangle(r, pair_mask)
                               for r in active.values()])
    design = np.column_stack([np.ones(len(vectors)), vectors])
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError("Second-image RSA design is rank deficient; value/control "
                         f"effects cannot be uniquely estimated: {list(active)}")
    target = vectorize_lower_triangle(active['odor_value'], pair_mask)
    controls = np.column_stack([np.ones(len(target))] + [
        vectorize_lower_triangle(r, pair_mask) for n, r in active.items()
        if n != 'odor_value'
    ])
    residual = target - controls @ np.linalg.lstsq(controls, target, rcond=None)[0]
    info = dict(n_between_run_pairs=int(pair_mask.sum()),
                included_regressors=list(active), dropped_constant_regressors=dropped,
                design_rank=rank, design_columns=int(design.shape[1]),
                design_condition_number=float(np.linalg.cond(design)),
                value_variance_fraction_after_controls=float(
                    residual @ residual / np.sum((target - target.mean()) ** 2)))
    return active, pair_mask, info


def write_second_image_diagnostics(output_dir, sub, cue_subset, value_model,
                                   center_within_run, samples, variables,
                                   diagnostic_rdms, pair_mask, design_info, targets):
    directory = Path(output_dir) / 'diagnostics' / sub / f'subset-{cue_subset}'
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / 'analysis_metadata.json'
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if (previous['value_model'] != value_model or
                previous['center_within_run'] != bool(center_within_run)):
            raise ValueError(f"Existing output has different analysis settings: {directory}")
    table = samples.copy()
    for name, values in variables.items():
        table[name] = values
    table.to_csv(directory / 'samples.csv', index=False)
    vectors = pd.DataFrame({n: vectorize_lower_triangle(r, pair_mask)
                            for n, r in diagnostic_rdms.items()})
    correlations = vectors.corr()
    correlations.to_csv(directory / 'model_rdm_correlations.csv')
    np.savez_compressed(directory / 'model_rdms.npz', **diagnostic_rdms,
                        between_run_pair_mask=pair_mask)
    package = Path(__file__).resolve().parent
    metadata = dict(
        subject=sub, image_scope='second', value_model=value_model,
        cue_subset=cue_subset, sessions=list(dict.fromkeys(samples.session)),
        n_samples=len(samples), center_within_run=bool(center_within_run),
        rating_normalization='Joint z-score of three outcome ratings per requested phase, before trial expansion',
        trial_variable_definitions={'cue_value': 'V2', 'signed_difference': 'V2 - V1',
                                    'absolute_difference': 'abs(V2 - V1)'},
        value_rdm='abs(trial_variable_i - trial_variable_j)',
        neural_rdm='1 - Pearson correlation',
        regression='OLS with intercept; selected value RDM plus included nuisance RDMs',
        phase_rdm_in_regression=False,
        diagnostic_only_rdms='The alternative value RDMs, first-image identity, and unordered outcome-pair identity are diagnostics, not additional regressors.',
        requested_targets=list(targets), status='prepared',
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(package.glob('*.py'))},
        **design_info,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f"Second-image design: {design_info}")
    print(f"Model RDM correlations on between-run pairs:\n{correlations.to_string()}")
    return metadata_path
