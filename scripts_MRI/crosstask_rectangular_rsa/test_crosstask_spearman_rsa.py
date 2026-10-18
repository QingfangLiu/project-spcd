#!/usr/bin/env python3
"""Tests for passive-odor to premeal-cue rectangular Spearman RSA."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aggregate_roi_crosstask_spearman_rsa as aggregate
import plot_ofc_k2_crosstask_spearman_rsa as plot_rsa
import run_roi_crosstask_spearman_rsa as subject_rsa


class RectangularRsaTests(unittest.TestCase):
    def test_cross_task_mask_selects_only_rectangle(self) -> None:
        mask = subject_rsa._cross_task_pair_mask(4, 3)
        self.assertEqual(mask.dtype, np.dtype(bool))
        self.assertEqual(mask.shape, (21,))
        self.assertEqual(int(mask.sum()), 12)

    def test_perfect_odor_geometry_has_spearman_one(self) -> None:
        odor_1 = np.array([2.0, 1.0, -1.0, -2.0])
        odor_2 = -odor_1
        passive = np.vstack([odor_1, odor_1, odor_2, odor_2])
        cue = np.vstack([odor_1, odor_2, odor_1, odor_2])
        passive_labels = np.array([0, 0, 1, 1])
        cue_labels = np.array([0, 1, 0, 1])

        result, _, _, pair_mask = (
            subject_rsa._compute_cross_task_rank_statistics(
                passive,
                passive_labels,
                cue,
                cue_labels,
            )
        )
        self.assertEqual(int(pair_mask.sum()), 16)
        self.assertAlmostEqual(
            result["cross_task_odor_identity_spearman_rho"],
            1.0,
        )
        self.assertAlmostEqual(
            result["cross_task_odor_identity_rank_biserial"],
            1.0,
        )

    def test_pair_table_preserves_cue_labels(self) -> None:
        passive_metadata = pd.DataFrame(
            {
                "run": [1, 1],
                "beta_trial_number": [1, 2],
                "trial_type": ["odor_1", "odor_2"],
                "label": [0, 1],
            }
        )
        cue_samples = pd.DataFrame(
            {
                "run": [1, 2],
                "trial": [3, 4],
                "condition": ["A11", "A21"],
            }
        )
        cue_labels = np.array([0, 1])
        labels = np.array([0, 1, 0, 1])
        identity_rdm = (labels[:, None] != labels[None, :]).astype(float)
        neural_rdm = np.arange(16, dtype=float).reshape(4, 4)
        table = subject_rsa._cross_task_pair_table(
            neural_rdm,
            identity_rdm,
            passive_metadata,
            cue_samples,
            cue_labels,
        )
        self.assertEqual(table["cue_label"].tolist(), [0, 0, 1, 1])
        self.assertEqual(
            table["odor_identity_dissimilarity"].tolist(),
            [0, 1, 1, 0],
        )

    def test_group_summary_contains_cue_and_paired_tests(self) -> None:
        rows = []
        for index in range(6):
            for cue, value in (
                ("conditioned_odor_only", 0.01 + index * 0.001),
                ("preconditioned_odor_only", 0.08 + index * 0.002),
            ):
                rows.append(
                    {
                        "sub": f"sub-{index + 1:02d}",
                        "cue_subset": cue,
                        "cross_task_odor_identity_spearman_rho": value,
                        "cross_task_odor_identity_rank_biserial": value,
                    }
                )
        summary = aggregate._summarize(pd.DataFrame(rows))
        self.assertEqual(len(summary), 6)
        self.assertIn(
            "preconditioned_minus_conditioned",
            set(summary["cue_subset"]),
        )

    def test_plot_panel_accepts_complete_subject_pairs(self) -> None:
        rows = []
        for index in range(6):
            for cue, value in (
                ("conditioned_odor_only", -0.02 + index * 0.005),
                ("preconditioned_odor_only", 0.03 + index * 0.008),
            ):
                rows.append(
                    {
                        "sub": f"sub-{index + 1:02d}",
                        "cue_subset": cue,
                        plot_rsa.METRIC: value,
                    }
                )
        figure, axis = plot_rsa.plt.subplots()
        try:
            plot_rsa._plot_panel(axis, pd.DataFrame(rows), "Synthetic ROI")
            self.assertEqual(len(axis.get_xticklabels()), 2)
        finally:
            plot_rsa.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
