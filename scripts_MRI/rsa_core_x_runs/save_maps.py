
"""
Output utilities for RSA maps.

This module defines standardized routines for saving subject-level
RSA result maps to disk, including filename construction that
encodes analysis parameters such as model, metric, and radius.
"""

import os

def save_rsa_maps(sub: str,
                  rsa_maps: dict,
                  output_dir: str,
                  session_tag: str,
                  radius: float = 8.0,
                  rsa_method: str = "regression",
                  cue_subset: str = "all",
                  use_pair_mask: bool = False,
                  mode: str = "both",
                  stat_type: str = "beta",
                  ):
    
    """
    Save RSA score maps for one subject.
    
    Parameters
    ----------
    sub : str
        Subject ID
    rsa_maps : dict
        Dictionary of rsa maps (NIfTI images)
    output_dir : str
        Output directory
    session_tag : str
        Session tag (e.g., 'premeal')
    radius : float
        Searchlight radius for filename
    rsa_method : str
        "regression", "spearman", or "kendall"
    cue_subset : str
        Which cues were included: 'all', 'conditioned', or 'preconditioned'
    use_pair_mask : bool
        Whether odor-only pair mask was applied
    mode: str
        "both", "img1only", "img2only" 
    stat_type : str
        Input statistic type from GLM: 'beta' or 'tmap'
    """

    os.makedirs(output_dir, exist_ok=True)

    mask_label = "mask-odor" if use_pair_mask else None
    subset_label = None if cue_subset == "all" else f"subset-{cue_subset}"

    for model_name, rsa_map in rsa_maps.items():

        name_parts = [
            f"{sub}_session-{session_tag}",
            f"input-{input_stat}",
            f"mode-{mode}",
            model_name,
            rsa_method,
            f"radius-{int(radius)}mm",
        ]

        if subset_label is not None:
            name_parts.append(subset_label)

        if mask_label is not None:
            name_parts.append(mask_label)

        output_file = os.path.join(
            output_dir,
            "_".join(name_parts) + ".nii.gz"
        )
        rsa_map.to_filename(output_file)
        print(f"Saved: {output_file}")
