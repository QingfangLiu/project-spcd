#!/usr/bin/env python3
"""Tests for binary first-image cross-task loading invariants."""

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

import odor_training_utils_binary as odor_utils
import probe_decoding_utils_first_image as probe_utils


class BinaryFirstImageLoaderTests(unittest.TestCase):
    def test_cue_subset_indices_match_established_definitions(self):
        self.assertEqual(
            probe_utils.CUE_SUBSET_INDICES["conditioned_odor_only"],
            {2, 4, 8, 10},
        )
        self.assertEqual(
            probe_utils.CUE_SUBSET_INDICES["preconditioned_odor_only"],
            {1, 3, 7, 9},
        )

    def test_select_first_image_rows_rejects_missing_column(self):
        with self.assertRaisesRegex(ValueError, "missing img_index"):
            probe_utils._select_first_image_rows(
                pd.DataFrame({"beta_file": ["beta.nii"]}),
                "index.csv",
            )

    def test_normalization_pool_is_first_image_and_subset_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sub = "sub-01"
            sub_dir = os.path.join(temp_dir, "GLM", "premeal_trialwise_2mm", sub)
            os.makedirs(sub_dir)
            csv_path = os.path.join(sub_dir, f"{sub}_premeal_run-1_beta_index.csv")
            pd.DataFrame(
                {
                    "beta_file": [
                        "pre_odor1.nii",
                        "pre_odor2.nii",
                        "cond_odor1.nii",
                        "cond_odor2.nii",
                        "clean_air.nii",
                        "second_image.nii",
                    ],
                    "run": [1] * 6,
                    "set": [1, 1, 1, 1, 1, 1],
                    "trial": [1, 2, 3, 4, 5, 1],
                    "img_index": [1, 1, 1, 1, 1, 2],
                    "trial_type": ["cue_img1"] * 5 + ["cue_img2"],
                    "cue_index": [1, 3, 2, 4, 5, 1],
                    "odor_index": [1, 2, 1, 2, 0, 1],
                }
            ).to_csv(csv_path, index=False)

            captured = {}

            def fake_zscore(df_run_all, df_selected):
                captured["stats"] = df_run_all.copy()
                captured["selected"] = df_selected.copy()
                return [object()] * len(df_selected)

            with patch.object(
                probe_utils,
                "_runwise_setwise_zscore",
                side_effect=fake_zscore,
            ):
                imgs, labels, meta = probe_utils.load_betas_cue_first_image(
                    sub=sub,
                    run=1,
                    base_dir=temp_dir,
                    cue_subset="preconditioned_odor_only",
                    normalize=True,
                )

            self.assertEqual(len(imgs), 2)
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(meta["cue_index"].tolist(), [1, 3])
            self.assertEqual(meta["img_index"].tolist(), [1, 1])
            self.assertEqual(captured["stats"]["cue_index"].tolist(), [1, 3])
            self.assertEqual(captured["selected"]["cue_index"].tolist(), [1, 3])

    def test_sub14_uses_original_premeal_odor_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            events_dir = os.path.join(temp_dir, "original_events")
            os.makedirs(events_dir)
            pd.DataFrame(
                {
                    "trial_type": ["cue_img1", "cue_img2", "cue_img1"],
                    "run": [1, 1, 1],
                    "trial": [1, 1, 2],
                    "img_index": [1, 2, 1],
                    "odor_index": [1, 2, 2],
                }
            ).to_csv(
                os.path.join(
                    events_dir,
                    "sub-14_task-premeal_run-1_events_orig.tsv",
                ),
                sep="\t",
                index=False,
            )
            selected = pd.DataFrame(
                {
                    "run": [1, 1],
                    "trial": [1, 2],
                    "img_index": [1, 1],
                    "odor_index": [2, 1],
                }
            )

            labeled = probe_utils._apply_sub14_original_labels(
                selected,
                sub="sub-14",
                base_dir=temp_dir,
                session_tag="premeal",
                original_events_dir=events_dir,
            )

            self.assertEqual(labeled["odor_index"].tolist(), [1, 2])
            self.assertEqual(
                labeled["label_source"].tolist(),
                ["original_premeal_events", "original_premeal_events"],
            )

    def test_binary_odor_rows_preserve_full_event_beta_numbers(self):
        events = pd.DataFrame(
            {
                "trial_type": ["odor_1", "clean_air", "odor_2", "clean_air"]
            }
        )
        selected = odor_utils._binary_odor_event_rows(events, "events.tsv")
        self.assertEqual(selected["beta_trial_number"].tolist(), [1, 3])
        self.assertEqual(selected["y"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
