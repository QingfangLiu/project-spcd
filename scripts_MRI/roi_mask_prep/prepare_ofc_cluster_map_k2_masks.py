#!/usr/bin/env python3
"""Prepare separate 2 mm gray-matter masks from the Kahnt OFC K=2 map.

The source map was downloaded from NeuroVault collection 17528 and contains
two scaled, nonzero label values.  This script maps the lower nonzero value to
published cluster 1 and the higher value to published cluster 2, resamples the
label image to the requested gray-matter reference with nearest-neighbor
interpolation, intersects each cluster with that gray-matter mask, and writes
two binary NIfTI files plus a provenance manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


NEUROVAULT_COLLECTION = "https://identifiers.org/neurovault.collection:17528"
NEUROVAULT_URL = "https://neurovault.org/collections/17528/"
PAPER_DOI = "10.1523/JNEUROSCI.0257-12.2012"

CLUSTERS = (
    {
        "cluster_id": 1,
        "slug": "cluster1_medial_posterior_lateral",
        "name": "medial/posterior-lateral OFC",
        "filename": (
            "mask-OFC_K2_cluster1_medial_posterior_lateral_gm0p1.nii.gz"
        ),
    },
    {
        "cluster_id": 2,
        "slug": "cluster2_central_anterior_lateral",
        "name": "central/anterior-lateral OFC",
        "filename": (
            "mask-OFC_K2_cluster2_central_anterior_lateral_gm0p1.nii.gz"
        ),
    },
)


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _format_shape(shape) -> str:
    return "x".join(str(int(value)) for value in shape[:3])


def _format_zooms(img) -> str:
    return "x".join(f"{float(value):g}" for value in img.header.get_zooms()[:3])


def _format_orientation(img) -> str:
    return "".join(nib.aff2axcodes(img.affine))


def _format_xyz(values) -> str:
    return ";".join(f"{float(value):.3f}" for value in values)


def _positive_label_values(data: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(data[np.isfinite(data)], dtype=float)
    positive = np.unique(finite[finite > 0])
    if len(positive) != 2:
        raise ValueError(
            "Expected exactly two positive values in the K=2 source map; "
            f"found {len(positive)}: {positive.tolist()}"
        )
    return float(positive[0]), float(positive[1])


def _build_integer_label_image(
    source_img,
    source_data: np.ndarray,
    source_values: tuple[float, float],
):
    labels = np.zeros(source_data.shape, dtype=np.uint8)
    positive_mask = np.isfinite(source_data) & (source_data > 0)

    for cluster_id, source_value in enumerate(source_values, start=1):
        cluster_mask = np.isclose(
            source_data,
            source_value,
            rtol=1e-6,
            atol=1e-6,
        )
        labels[cluster_mask] = cluster_id

    unassigned = positive_mask & (labels == 0)
    if np.any(unassigned):
        raise ValueError(
            f"{int(unassigned.sum())} positive source voxels were not assigned "
            "to either K=2 cluster."
        )

    label_img = nib.Nifti1Image(labels, source_img.affine)
    qform, qform_code = source_img.get_qform(coded=True)
    sform, sform_code = source_img.get_sform(coded=True)
    if qform is not None:
        label_img.set_qform(qform, int(qform_code))
    if sform is not None:
        label_img.set_sform(sform, int(sform_code))
    return label_img


def _binary_image(mask: np.ndarray, reference_img):
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)

    out_img = nib.Nifti1Image(
        np.asarray(mask, dtype=np.uint8),
        reference_img.affine,
        header=header,
    )
    qform, qform_code = reference_img.get_qform(coded=True)
    sform, sform_code = reference_img.get_sform(coded=True)
    if qform is not None:
        out_img.set_qform(qform, int(qform_code))
    if sform is not None:
        out_img.set_sform(sform, int(sform_code))
    return out_img


def _preflight_outputs(output_dir: Path, force: bool) -> list[Path]:
    output_paths = [output_dir / cluster["filename"] for cluster in CLUSTERS]
    output_paths.append(output_dir / "manifest.csv")
    existing = [path for path in output_paths if path.exists()]
    if existing and not force:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "Output files already exist. Use --force to replace them:\n"
            f"{formatted}"
        )
    return output_paths


def prepare_masks(
    *,
    source_mask: Path,
    reference_mask: Path,
    output_dir: Path,
    project_root: Path,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    source_mask = source_mask.expanduser().resolve()
    reference_mask = reference_mask.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    project_root = project_root.expanduser().resolve()

    if not source_mask.is_file():
        raise FileNotFoundError(f"Source cluster map not found: {source_mask}")
    if not reference_mask.is_file():
        raise FileNotFoundError(
            f"Gray-matter reference mask not found: {reference_mask}"
        )

    _preflight_outputs(output_dir, force=(force or dry_run))

    source_img = nib.load(str(source_mask))
    reference_img = nib.load(str(reference_mask))
    if len(source_img.shape) != 3 or len(reference_img.shape) != 3:
        raise ValueError(
            "Source and reference masks must both be three-dimensional."
        )

    source_data = np.asanyarray(source_img.dataobj, dtype=float)
    reference_data = np.asanyarray(reference_img.dataobj, dtype=float)
    if np.any(~np.isfinite(source_data)):
        raise ValueError("Source K=2 cluster map contains nonfinite values.")

    gm_mask = np.isfinite(reference_data) & (reference_data > 0)
    if not np.any(gm_mask):
        raise ValueError("Gray-matter reference mask has no positive voxels.")

    source_values = _positive_label_values(source_data)
    integer_label_img = _build_integer_label_image(
        source_img,
        source_data,
        source_values,
    )
    resampled_img = resample_from_to(
        integer_label_img,
        reference_img,
        order=0,
        mode="constant",
        cval=0,
    )
    resampled_labels = np.rint(
        np.asanyarray(resampled_img.dataobj, dtype=float)
    ).astype(np.uint8)

    unexpected = sorted(
        set(np.unique(resampled_labels).tolist()) - {0, 1, 2}
    )
    if unexpected:
        raise ValueError(
            f"Unexpected labels after nearest-neighbor resampling: {unexpected}"
        )

    source_sha256 = _sha256(source_mask)
    reference_sha256 = _sha256(reference_mask)
    rows = []

    for cluster, source_value in zip(CLUSTERS, source_values):
        cluster_id = int(cluster["cluster_id"])
        resampled_mask = resampled_labels == cluster_id
        final_mask = resampled_mask & gm_mask
        n_before_gm = int(resampled_mask.sum())
        n_after_gm = int(final_mask.sum())
        if n_after_gm == 0:
            raise ValueError(
                f"Cluster {cluster_id} has no voxels after GM intersection."
            )

        voxel_indices = np.column_stack(np.where(final_mask))
        world_xyz = nib.affines.apply_affine(
            reference_img.affine,
            voxel_indices,
        )
        centroid_xyz = world_xyz.mean(axis=0)
        output_mask = output_dir / str(cluster["filename"])

        print(
            f"Cluster {cluster_id}: {cluster['name']}\n"
            f"  source value:       {source_value:.9g}\n"
            f"  resampled voxels:   {n_before_gm}\n"
            f"  GM-intersected:     {n_after_gm}\n"
            f"  removed by GM:      {n_before_gm - n_after_gm}\n"
            f"  centroid MNI (mm):  {_format_xyz(centroid_xyz)}\n"
            f"  output:              {output_mask}"
        )

        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            nib.save(
                _binary_image(final_mask, reference_img),
                str(output_mask),
            )

        row = {
            "cluster_id": cluster_id,
            "cluster_slug": cluster["slug"],
            "cluster_name": cluster["name"],
            "source_label_value": f"{source_value:.12g}",
            "source_mask": _display_path(source_mask, project_root),
            "source_sha256": source_sha256,
            "source_shape": _format_shape(source_img.shape),
            "source_zooms_mm": _format_zooms(source_img),
            "source_orientation": _format_orientation(source_img),
            "reference_mask": _display_path(reference_mask, project_root),
            "reference_sha256": reference_sha256,
            "output_mask": _display_path(output_mask, project_root),
            "output_shape": _format_shape(reference_img.shape),
            "output_zooms_mm": _format_zooms(reference_img),
            "output_orientation": _format_orientation(reference_img),
            "n_voxels_resampled_before_gm": n_before_gm,
            "n_voxels_after_gm": n_after_gm,
            "n_voxels_removed_by_gm": n_before_gm - n_after_gm,
            "centroid_mni_xyz_mm": _format_xyz(centroid_xyz),
            "interpolation": "nearest-neighbor",
            "gm_rule": "reference_mask > 0",
            "neurovault_collection": NEUROVAULT_COLLECTION,
            "neurovault_url": NEUROVAULT_URL,
            "paper_doi": PAPER_DOI,
            "output_sha256": "",
        }
        rows.append(row)

    if not dry_run:
        for row in rows:
            output_path = project_root / row["output_mask"]
            if not output_path.is_file():
                output_path = Path(row["output_mask"])
            row["output_sha256"] = _sha256(output_path)

        manifest_path = output_dir / "manifest.csv"
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
        print("Dry run: no masks or manifest were written.")

    return rows


def main() -> None:
    project_root = _default_project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Split the Kahnt OFC K=2 cluster map and resample each cluster "
            "to a 2 mm gray-matter reference mask."
        )
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=project_root,
        help="Project root. Default: infer from this script location.",
    )
    parser.add_argument(
        "--source_mask",
        type=Path,
        default=project_root / "masks" / "OFC_clusterMap_K2.nii.gz",
        help="Downloaded NeuroVault K=2 label map.",
    )
    parser.add_argument(
        "--reference_mask",
        type=Path,
        default=project_root / "masks" / "gm_0.1.nii",
        help="Target-grid gray-matter mask.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "masks" / "OFC_clusterMap_K2_gm0p1",
        help="Output directory for the two binary masks and manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing prepared masks and manifest.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Compute and report preparation details without writing files.",
    )
    args = parser.parse_args()

    prepare_masks(
        source_mask=args.source_mask,
        reference_mask=args.reference_mask,
        output_dir=args.output_dir,
        project_root=args.project_root,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
