#!/usr/bin/env python3
"""Tests for first-image-only probe loading invariants."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import nilearn.image  # noqa: F401
except ModuleNotFoundError:
    nilearn = types.ModuleType("nilearn")
    nilearn_image = types.ModuleType("nilearn.image")

    def unavailable_nilearn(*args, **kwargs):
        raise RuntimeError("This test path should not load NIfTI data")

    for name in ("concat_imgs", "get_data", "load_img", "new_img_like"):
        setattr(nilearn_image, name, unavailable_nilearn)
    nilearn.image = nilearn_image
    sys.modules["nilearn"] = nilearn
    sys.modules["nilearn.image"] = nilearn_image

import probe_decoding_utils_first_image as first_image_utils


class FirstImageProbeLoaderTests(unittest.TestCase):
    def test_select_first_image_rows_rejects_missing_column(self):
        with self.assertRaisesRegex(ValueError, "missing img_index"):
            first_image_utils._select_first_image_rows(
                pd.DataFrame({"beta_file": ["beta.nii"]}),
                "index.csv",
            )

    def test_normalization_pool_contains_only_first_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sub = "sub-01"
            sub_dir = os.path.join(temp_dir, "GLM", "premeal_trialwise_2mm", sub)
            os.makedirs(sub_dir)
            csv_path = os.path.join(sub_dir, f"{sub}_premeal_run-1_beta_index.csv")
            pd.DataFrame(
                {
                    "beta_file": ["first_a.nii", "second_a.nii", "first_b.nii", "second_b.nii"],
                    "run": [1, 1, 1, 1],
                    "set": [1, 1, 2, 2],
                    "trial": [1, 1, 2, 2],
                    "img_index": [1, 2, 1, 2],
                    "trial_type": ["cue_img1", "cue_img2", "cue_img1", "cue_img2"],
                    "cue_index": [1, 2, 2, 1],
                    "odor_index": [1, 2, 2, 1],
                }
            ).to_csv(csv_path, index=False)

            captured = {}

            def fake_zscore(df_run_all, df_selected):
                captured["stats"] = df_run_all.copy()
                captured["selected"] = df_selected.copy()
                return [object()] * len(df_selected)

            with patch.object(
                first_image_utils,
                "_runwise_setwise_zscore",
                side_effect=fake_zscore,
            ):
                imgs, labels, meta = first_image_utils.load_betas_cue_first_image(
                    sub=sub,
                    run=1,
                    base_dir=temp_dir,
                    cue_type="AB",
                    n_classes=2,
                    normalize=True,
                )

            self.assertEqual(len(imgs), 2)
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(meta["img_index"].tolist(), [1, 1])
            self.assertEqual(captured["stats"]["img_index"].tolist(), [1, 1])
            self.assertEqual(captured["selected"]["img_index"].tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
