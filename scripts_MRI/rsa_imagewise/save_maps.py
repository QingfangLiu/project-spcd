"""Save RSA maps."""

from __future__ import annotations

import os


def save_rsa_maps(
    sub: str,
    rsa_maps: dict,
    output_dir: str,
    radius: float,
    button_scope: str,
):
    os.makedirs(output_dir, exist_ok=True)
    for model_name, rsa_map in rsa_maps.items():
        fname = "_".join([
            sub,
            f"model-{model_name}",
            f"buttonscope-{button_scope}",
            f"radius-{radius}mm",
        ]) + ".nii.gz"
        out = os.path.join(output_dir, fname)
        rsa_map.to_filename(out)
        print(f"Saved: {out}")
