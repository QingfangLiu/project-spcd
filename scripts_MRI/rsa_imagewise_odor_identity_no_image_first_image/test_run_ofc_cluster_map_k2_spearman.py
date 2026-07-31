#!/usr/bin/env python3
"""Tests for hemisphere-specific Kahnt OFC K=2 Spearman orchestration."""

from __future__ import annotations

import unittest

from run_ofc_cluster_map_k2 import CLUSTERS
from run_ofc_cluster_map_k2_spearman import _analysis_scope


class AnalysisScopeTest(unittest.TestCase):
    def test_bilateral_scope_preserves_existing_names(self) -> None:
        spec = CLUSTERS["cluster1"]
        scope = _analysis_scope(spec, "bilateral")

        self.assertEqual(scope["mask_filename"], spec.mask_filename)
        self.assertEqual(scope["roi_name"], spec.roi_name)
        self.assertEqual(scope["output_key"], spec.key)
        self.assertEqual(
            scope["plot_stem"],
            "OFC_K2_cluster1_medial_posterior_lateral_"
            "no_image_first_image_spearman_roi_rsa_paired",
        )

    def test_unilateral_scopes_have_distinct_names(self) -> None:
        spec = CLUSTERS["cluster2"]
        left = _analysis_scope(spec, "left")
        right = _analysis_scope(spec, "right")

        self.assertTrue(left["mask_filename"].endswith("_left_gm0p1.nii.gz"))
        self.assertTrue(
            right["mask_filename"].endswith("_right_gm0p1.nii.gz")
        )
        self.assertEqual(
            left["roi_name"], "OFC_K2_cluster2_central_anterior_lateral_left"
        )
        self.assertEqual(
            right["output_key"],
            "cluster2_central_anterior_lateral_right",
        )
        self.assertNotEqual(left["plot_stem"], right["plot_stem"])


if __name__ == "__main__":
    unittest.main()
