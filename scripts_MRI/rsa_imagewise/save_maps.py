"""Save RSA maps."""

from __future__ import annotations

import os


def save_single_rsa_map(
    sub: str,
    model_name: str,
    rsa_map,
    output_dir: str,
    radius: float,
    button_scope: str,
):
    os.makedirs(output_dir, exist_ok=True)
    fname = "_".join([
        sub,
        f"model-{model_name}",
        f"buttonscope-{button_scope}",
        f"radius-{radius}mm",
    ]) + ".nii.gz"
    out = os.path.join(output_dir, fname)
    rsa_map.to_filename(out)
    print(f"Saved: {out}")


def save_rsa_maps(
    sub: str,
    rsa_maps: dict,
    output_dir: str,
    radius: float,
    button_scope: str,
):
    for model_name, rsa_map in rsa_maps.items():
        save_single_rsa_map(
            sub=sub,
            model_name=model_name,
            rsa_map=rsa_map,
            output_dir=output_dir,
            radius=radius,
            button_scope=button_scope,
        )
