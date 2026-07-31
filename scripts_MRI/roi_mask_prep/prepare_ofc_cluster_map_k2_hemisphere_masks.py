#!/usr/bin/env python3
"""Split prepared bilateral Kahnt OFC K=2 masks at MNI x=0."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

if __package__ in (None, ""):
    from prepare_ofc_cluster_map_k2_masks import (
        CLUSTERS,
        _binary_image,
        _default_project_root,
        _display_path,
        _format_orientation,
        _format_shape,
        _format_xyz,
        _format_zooms,
        _sha256,
    )
else:
    from .prepare_ofc_cluster_map_k2_masks import (
        CLUSTERS,
        _binary_image,
        _default_project_root,
        _display_path,
        _format_orientation,
        _format_shape,
        _format_xyz,
        _format_zooms,
        _sha256,
    )


HEMISPHERES = ("left", "right")
HEMISPHERE_RULES = {
    "left": "MNI x < 0 mm",
    "right": "MNI x > 0 mm",
}


def hemisphere_mask_filename(filename: str, hemisphere: str) -> str:
    if hemisphere not in HEMISPHERES:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")
    suffix = "_gm0p1.nii.gz"
    if not filename.endswith(suffix):
        raise ValueError(
            f"Expected mask filename ending in {suffix!r}; got {filename!r}"
        )
    return f"{filename[:-len(suffix)]}_{hemisphere}{suffix}"


def _output_paths(mask_dir: Path) -> list[Path]:
    paths = [
        mask_dir / hemisphere_mask_filename(cluster["filename"], hemisphere)
        for cluster in CLUSTERS
        for hemisphere in HEMISPHERES
    ]
    return [*paths, mask_dir / "hemisphere_manifest.csv"]


def _preflight_outputs(mask_dir: Path, force: bool) -> None:
    existing = [path for path in _output_paths(mask_dir) if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Hemisphere outputs already exist. Use --force to replace them:\n"
            f"{formatted}"
        )


def prepare_hemisphere_masks(
    *,
    mask_dir: Path,
    project_root: Path,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    mask_dir = mask_dir.expanduser().resolve()
    project_root = project_root.expanduser().resolve()
    _preflight_outputs(mask_dir, force=(force or dry_run))

    rows: list[dict] = []
    for cluster in CLUSTERS:
        source_mask = mask_dir / str(cluster["filename"])
        if not source_mask.is_file():
            raise FileNotFoundError(
                f"Prepared bilateral mask not found: {source_mask}"
            )

        source_img = nib.load(str(source_mask))
        if len(source_img.shape) != 3:
            raise ValueError(f"Mask must be three-dimensional: {source_mask}")
        source_data = np.asanyarray(source_img.dataobj, dtype=float)
        if np.any(~np.isfinite(source_data)):
            raise ValueError(f"Mask contains nonfinite values: {source_mask}")
        bilateral_mask = source_data > 0
        if not np.any(bilateral_mask):
            raise ValueError(f"Mask has no positive voxels: {source_mask}")

        voxel_indices = np.column_stack(np.where(bilateral_mask))
        world_xyz = nib.affines.apply_affine(source_img.affine, voxel_indices)
        world_x = world_xyz[:, 0]
        midline = np.isclose(world_x, 0.0, rtol=0.0, atol=1e-6)
        n_bilateral = int(len(voxel_indices))
        n_midline = int(midline.sum())

        for hemisphere in HEMISPHERES:
            selected = world_x < 0 if hemisphere == "left" else world_x > 0
            selected &= ~midline
            hemisphere_indices = voxel_indices[selected]
            n_hemisphere = int(len(hemisphere_indices))
            if n_hemisphere == 0:
                raise ValueError(
                    f"{cluster['slug']} has no {hemisphere}-hemisphere voxels."
                )

            hemisphere_mask = np.zeros(source_img.shape, dtype=bool)
            hemisphere_mask[tuple(hemisphere_indices.T)] = True
            centroid_xyz = world_xyz[selected].mean(axis=0)
            output_mask = mask_dir / hemisphere_mask_filename(
                str(cluster["filename"]), hemisphere
            )

            print(
                f"Cluster {cluster['cluster_id']}: {cluster['name']} "
                f"({hemisphere})\n"
                f"  bilateral voxels:  {n_bilateral}\n"
                f"  hemisphere voxels: {n_hemisphere}\n"
                f"  midline excluded:   {n_midline}\n"
                f"  centroid MNI (mm):  {_format_xyz(centroid_xyz)}\n"
                f"  output:              {output_mask}"
            )

            if not dry_run:
                mask_dir.mkdir(parents=True, exist_ok=True)
                nib.save(
                    _binary_image(hemisphere_mask, source_img),
                    str(output_mask),
                )

            rows.append(
                {
                    "cluster_id": int(cluster["cluster_id"]),
                    "cluster_slug": cluster["slug"],
                    "cluster_name": cluster["name"],
                    "hemisphere": hemisphere,
                    "hemisphere_rule": HEMISPHERE_RULES[hemisphere],
                    "midline_rule": "MNI x = 0 mm excluded",
                    "source_bilateral_mask": _display_path(
                        source_mask, project_root
                    ),
                    "source_bilateral_sha256": _sha256(source_mask),
                    "output_mask": _display_path(output_mask, project_root),
                    "shape": _format_shape(source_img.shape),
                    "zooms_mm": _format_zooms(source_img),
                    "orientation": _format_orientation(source_img),
                    "n_voxels_bilateral": n_bilateral,
                    "n_voxels_hemisphere": n_hemisphere,
                    "n_voxels_midline_excluded": n_midline,
                    "centroid_mni_xyz_mm": _format_xyz(centroid_xyz),
                    "output_sha256": "",
                }
            )

    if not dry_run:
        for row in rows:
            output_path = mask_dir / Path(str(row["output_mask"])).name
            row["output_sha256"] = _sha256(output_path)

        manifest_path = mask_dir / "hemisphere_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0].keys()),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved manifest: {manifest_path}")
    else:
        print("Dry run: no hemisphere masks or manifest were written.")

    return rows


def main() -> None:
    project_root = _default_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Split prepared bilateral Kahnt OFC K=2 masks into left and "
            "right masks using MNI x coordinates."
        )
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=project_root,
        help="Project root. Default: infer from this script location.",
    )
    parser.add_argument(
        "--mask_dir",
        type=Path,
        default=project_root / "masks" / "OFC_clusterMap_K2_gm0p1",
        help="Directory containing the prepared bilateral masks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing hemisphere masks and manifest.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and report split details without writing files.",
    )
    args = parser.parse_args()

    prepare_hemisphere_masks(
        mask_dir=args.mask_dir,
        project_root=args.project_root,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
