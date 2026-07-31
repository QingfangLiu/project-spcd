#!/usr/bin/env python3
"""Tests for the Kahnt OFC K=2 hemisphere mask split."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from prepare_ofc_cluster_map_k2_hemisphere_masks import (
    prepare_hemisphere_masks,
)
from prepare_ofc_cluster_map_k2_masks import CLUSTERS


class PrepareHemisphereMasksTest(unittest.TestCase):
    def _write_bilateral_masks(self, mask_dir: Path) -> None:
        affine = np.array(
            [
                [2.0, 0.0, 0.0, -2.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        for cluster in CLUSTERS:
            data = np.ones((3, 2, 2), dtype=np.uint8)
            nib.save(
                nib.Nifti1Image(data, affine),
                str(mask_dir / cluster["filename"]),
            )

    def test_splits_by_world_x_and_excludes_midline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            mask_dir = project_root / "masks"
            mask_dir.mkdir()
            self._write_bilateral_masks(mask_dir)

            rows = prepare_hemisphere_masks(
                mask_dir=mask_dir,
                project_root=project_root,
            )

            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertEqual(row["n_voxels_bilateral"], 12)
                self.assertEqual(row["n_voxels_hemisphere"], 4)
                self.assertEqual(row["n_voxels_midline_excluded"], 4)
                output = project_root / row["output_mask"]
                data = np.asanyarray(nib.load(str(output)).dataobj)
                self.assertEqual(int((data > 0).sum()), 4)
                x_index = 0 if row["hemisphere"] == "left" else 2
                self.assertTrue(np.all(data[x_index, :, :] == 1))

            manifest = mask_dir / "hemisphere_manifest.csv"
            with manifest.open(newline="", encoding="utf-8") as stream:
                manifest_rows = list(csv.DictReader(stream))
            self.assertEqual(len(manifest_rows), 4)
            self.assertTrue(
                all(row["output_sha256"] for row in manifest_rows)
            )

    def test_dry_run_does_not_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            mask_dir = project_root / "masks"
            mask_dir.mkdir()
            self._write_bilateral_masks(mask_dir)

            prepare_hemisphere_masks(
                mask_dir=mask_dir,
                project_root=project_root,
                dry_run=True,
            )

            self.assertFalse((mask_dir / "hemisphere_manifest.csv").exists())
            self.assertEqual(len(list(mask_dir.glob("*left*"))), 0)
            self.assertEqual(len(list(mask_dir.glob("*right*"))), 0)


if __name__ == "__main__":
    unittest.main()
