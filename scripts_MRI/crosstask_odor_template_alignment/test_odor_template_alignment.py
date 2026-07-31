#!/usr/bin/env python3
"""Synthetic tests for passive-odor to cue template alignment."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aggregate_roi_odor_template_alignment as aggregate
import run_roi_odor_template_alignment as alignment


def _patterns() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    odor1 = np.asarray([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    odor2 = -odor1
    patterns = np.vstack(
        [
            odor1 * 0.9,
            odor1 * 1.1,
            odor2 * 0.9,
            odor2 * 1.1,
            odor1 * 1.2,
            odor1 * 0.8,
            odor2 * 1.2,
            odor2 * 0.8,
        ]
    )
    labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    runs = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
    return patterns, labels, runs


class PrototypeGeometryTests(unittest.TestCase):
    def test_matching_advantage_equals_oriented_axis_projection(self) -> None:
        patterns, labels, runs = _patterns()
        normalized = alignment.decoding._runwise_zscore(patterns, runs)
        prototypes, run_prototypes = (
            alignment._build_run_balanced_prototypes(
                normalized, labels, runs
            )
        )
        trial_metrics, summary = alignment._score_against_prototypes(
            normalized, labels, runs, prototypes
        )

        np.testing.assert_allclose(
            trial_metrics["matching_correlation_advantage"],
            trial_metrics["oriented_axis_projection"],
        )
        self.assertGreater(summary["matching_correlation_advantage"], 0)
        self.assertEqual(
            summary["cue_nearest_prototype_balanced_accuracy"], 1.0
        )
        self.assertEqual(set(run_prototypes), {1, 2})

    def test_run_and_label_balancing_gives_cells_equal_weight(self) -> None:
        values = np.asarray([1.0, 1.0, 3.0, 5.0, 7.0])
        labels = np.asarray([0, 0, 1, 0, 1])
        runs = np.asarray([1, 1, 1, 2, 2])
        observed = alignment._run_and_label_balanced_mean(
            values, labels, runs
        )
        self.assertEqual(observed, np.mean([1.0, 3.0, 5.0, 7.0]))

    def test_passive_leave_one_run_out_prototype_qc(self) -> None:
        patterns, labels, runs = _patterns()
        normalized = alignment.decoding._runwise_zscore(patterns, runs)
        metrics = alignment._passive_leave_one_run_out_metrics(
            normalized, labels, runs
        )
        self.assertEqual(metrics["passive_looro_balanced_accuracy"], 1.0)
        self.assertGreater(
            metrics["passive_looro_matching_correlation_advantage"], 0
        )


class SubjectIntegrationTests(unittest.TestCase):
    def test_conditioned_and_preconditioned_are_scored_separately(self) -> None:
        passive, labels, runs = _patterns()
        voxel_indices = np.column_stack(
            [np.arange(passive.shape[1]), np.zeros((passive.shape[1], 2))]
        ).astype(int)

        def load_cues(_diagnostics, sub, cue_subset):
            cue = passive if cue_subset.startswith("pre") else -passive
            samples = pd.DataFrame(
                {
                    "sample_index": np.arange(len(cue)),
                    "sub": [sub] * len(cue),
                    "run": runs,
                    "trial": np.arange(1, len(cue) + 1),
                    "img_index": np.ones(len(cue), dtype=int),
                    "condition": np.where(labels == 0, "A11", "A21"),
                }
            )
            return cue, voxel_indices, samples, Path(f"{cue_subset}.npz")

        def load_passive(_base, _sub, _runs, _voxels):
            metadata = pd.DataFrame({"run": runs, "label": labels})
            return passive, labels, runs, metadata, np.eye(4)

        with (
            patch.object(
                alignment.decoding,
                "_load_cue_diagnostics",
                side_effect=load_cues,
            ),
            patch.object(
                alignment.decoding,
                "_cue_labels",
                side_effect=lambda _sub, _samples, _events: (
                    labels,
                    "synthetic",
                ),
            ),
            patch.object(
                alignment.decoding,
                "_load_passive_odor_patterns",
                side_effect=load_passive,
            ),
        ):
            rows, trials = alignment._analyze_subject(
                base_dir=Path("/unused"),
                diagnostics_dir=Path("/unused"),
                sub="sub-01",
                roi_name="synthetic_roi",
                hemisphere="bilateral",
                cue_subsets=alignment.CUE_SUBSETS,
                odor_runs=(1, 2),
                sub14_original_events_dir=Path("/unused"),
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["cue_subset"] for row in rows}, set(alignment.CUE_SUBSETS)
        )
        self.assertEqual(set(trials), set(alignment.CUE_SUBSETS))
        effects = {
            row["cue_subset"]: row["matching_correlation_advantage"]
            for row in rows
        }
        self.assertGreater(effects["preconditioned_odor_only"], 0)
        self.assertLess(effects["conditioned_odor_only"], 0)
        self.assertFalse(
            trials["preconditioned_odor_only"].equals(
                trials["conditioned_odor_only"]
            )
        )
        self.assertTrue(all(row["n_voxels_used"] == 6 for row in rows))


class AggregationTests(unittest.TestCase):
    def test_primary_group_test_is_directional_against_zero(self) -> None:
        summary = aggregate._one_sample_summary(
            np.asarray([0.01, 0.02, 0.03, 0.04]),
            metric="matching_correlation_advantage",
            null_value=0.0,
            cue_subset="preconditioned_odor_only",
            test="one_sample",
            contrast="primary",
            directional=True,
        )
        self.assertEqual(summary["alternative"], "greater")
        self.assertEqual(summary["n_above_null"], 4)
        self.assertGreater(summary["mean"], 0)
        self.assertLess(summary["p_sign_flip_mean"], 0.1)


if __name__ == "__main__":
    unittest.main()
