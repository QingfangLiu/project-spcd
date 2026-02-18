
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
                  neural_metric: str = "crossnobis",
                  cue_subset: str = "all",
                  use_pair_mask: bool = False,
                  mode: str = "both",
                  stat_type: str = "beta",
                  use_resms_whitening: bool = False,
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

    # filename-only labels per new agreement
    pairmask_label = f"pairmask-{int(bool(use_pair_mask))}"          # 0/1
    subset_label   = f"subset-{cue_subset}" if cue_subset != "all" else None
    radius_label   = f"radius-{int(radius)}mm"                       # keep units explicit
    whiten_label   = f"whiten-{int(bool(use_resms_whitening))}"      # 0/1 (explicit even if constant)

    for model_name, rsa_map in rsa_maps.items():

        # stable ordering in filename:
        # sub, model, subset (optional), pairmask, radius, whiten
        name_parts = [
            f"{sub}",  # keep existing sub string (e.g., "sub-01")
            f"model-{model_name}",
        ]

        if subset_label is not None:
            name_parts.append(subset_label)

        name_parts.extend([pairmask_label, radius_label, whiten_label])

        output_file = os.path.join(output_dir, "_".join(name_parts) + ".nii.gz")

        # "_".join(list)
        # concatenate list elements into one string, inserting "_" between elements

        rsa_map.to_filename(output_file)
        print(f"Saved: {output_file}")

