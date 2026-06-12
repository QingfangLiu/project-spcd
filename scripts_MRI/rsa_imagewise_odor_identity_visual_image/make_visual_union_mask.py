#!/usr/bin/env python3
"""Create a GM-intersected union mask from character/place visual masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_MASK_DIR = Path("masks") / "independent_image_decoding_masks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a binary union mask from xjView-cleaned character and place "
            "independent image-decoding masks, intersected with a gray-matter mask."
        )
    )
    parser.add_argument(
        "--char_mask",
        default=str(DEFAULT_MASK_DIR / "char_p001_visual_only.hdr"),
        help="Binary/nonzero character decoding visual-only mask from xjView.",
    )
    parser.add_argument(
        "--place_mask",
        default=str(DEFAULT_MASK_DIR / "place_p001_visual_only.hdr"),
        help="Binary/nonzero place decoding visual-only mask from xjView.",
    )
    parser.add_argument(
        "--gm_mask",
        default=str(Path("masks") / "gm_0.1.nii"),
        help="Gray-matter mask used to intersect the union mask.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_MASK_DIR / "visual_union_p001_visual_only_gm0p1.nii"),
        help="Output path for the binary GM-intersected union mask.",
    )
    return parser.parse_args()


def load_mask(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    mask = np.isfinite(data) & (data > 0)
    return img, mask


def assert_same_space(reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if reference.shape[:3] != other.shape[:3]:
        raise ValueError(
            f"{name} shape does not match reference: "
            f"{other.shape[:3]} vs {reference.shape[:3]}"
        )
    if not np.allclose(reference.affine, other.affine, atol=1e-5):
        raise ValueError(f"{name} affine does not match reference affine.")


def main() -> None:
    args = parse_args()

    char_path = Path(args.char_mask)
    place_path = Path(args.place_mask)
    gm_path = Path(args.gm_mask)
    out_path = Path(args.out)

    char_img, char_mask = load_mask(char_path)
    place_img, place_mask = load_mask(place_path)
    gm_img, gm_mask = load_mask(gm_path)

    assert_same_space(char_img, place_img, "place_mask")
    assert_same_space(char_img, gm_img, "gm_mask")

    union_mask = char_mask | place_mask
    final_mask = union_mask & gm_mask

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = final_mask.astype(np.uint8)
    out_img = nib.Nifti1Image(out_data, affine=char_img.affine, header=char_img.header)
    out_img.set_data_dtype(np.uint8)
    out_img.header["descrip"] = "char/place image-decoding union intersected with gm_0.1"
    nib.save(out_img, str(out_path))

    print("Visual union mask written")
    print(f"  character mask: {char_path}")
    print(f"  place mask:     {place_path}")
    print(f"  GM mask:        {gm_path}")
    print(f"  output:         {out_path}")
    print("")
    print("Voxel counts")
    print(f"  character > 0:          {int(char_mask.sum())}")
    print(f"  place > 0:              {int(place_mask.sum())}")
    print(f"  overlap:                {int((char_mask & place_mask).sum())}")
    print(f"  union before GM:        {int(union_mask.sum())}")
    print(f"  GM mask > 0:            {int(gm_mask.sum())}")
    print(f"  union intersected GM:   {int(final_mask.sum())}")


if __name__ == "__main__":
    main()
