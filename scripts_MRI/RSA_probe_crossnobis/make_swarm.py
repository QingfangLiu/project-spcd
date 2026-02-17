#!/usr/bin/env python3
"""
Generate a Biowulf swarm file for cuewise GLM cross-run searchlight RSA.

Usage:
    python make_swarm.py

Then submit (example):
    swarm -f <swarm_file> -g 16 -t 1 --logdir logs_rsa

This version writes each swarm line in the "minimal way" for conda on Biowulf:
    source <conda_root>/etc/profile.d/conda.sh; conda activate <env>; ...; python -m <module> ...

Notes
-----
- Subjects are discovered from:
    <base_dir>/GLM/{session}_{glm_variant}_2mm/sub-*
  using the first session in `sessions`.
- `--models` is written as multiple tokens to match argparse nargs="+" in run_rsa.py.
"""

import os
from typing import Iterable, List, Optional, Tuple, Union


def _ensure_models_list(models: Union[str, Iterable[str]]) -> List[str]:
    """Normalize models input into a non-empty list of strings."""
    if isinstance(models, str):
        parts = [p.strip() for p in models.split() if p.strip()]
        if not parts:
            raise ValueError("Empty models string provided.")
        return parts
    parts = [str(m).strip() for m in models if str(m).strip()]
    if not parts:
        raise ValueError("Empty models iterable provided.")
    return parts


def generate_swarm_file(
    base_dir: str = "/data/liuq13",
    module_name: str = "RSA_probe_crossnobis.run_rsa",
    output_file: Optional[str] = None,
    sessions: Tuple[str, ...] = ("premeal", "postmeal"),
    radius_mm: float = 8.0,
    n_jobs: int = 4,  # passed to your python code as --n_jobs (independent of swarm -t)
    rsa_method: str = "spearman",
    mask_path: Optional[str] = None,
    cue_subset: str = "all",
    models: Union[str, Iterable[str]] = ("identity",),
    use_pair_mask: bool = False,
    stat_type: str = "beta",       # "beta" or "tmap"
    mode: str = "both",            # "both", "img1only", "img2only"
    neural_metric: str = "crossnobis",
    glm_variant: str = "cuewise",  # "cuewise" or "cuewiseLSS"
    session_folder_template: str = "{session}_{glm_variant}_2mm",  # for subject discovery

    # NEW (aligned to current run_rsa.py)
    use_resms_whitening: bool = False,
    whitening_eps: float = 1e-6,

    # Conda activation (Biowulf swarm-safe; minimal way)
    conda_root: str = "/data/liuq13/conda",
    conda_env_path: str = "/data/liuq13/conda/envs/project_rsa",
    cap_threads: bool = True,
    pythonpath: Optional[str] = None,  # e.g., "/data/liuq13/SPCD_RSA_cuewise_subset"
) -> str:
    """Create a swarm file and return its path."""

    # Defaults
    if mask_path is None:
        mask_path = os.path.join(base_dir, "masks", "gm_0.1.nii")  # matches run_rsa.py default

    models_list = _ensure_models_list(models)

    # Auto-generate output filename if not provided
    if output_file is None:
        mask_tag = "_mask-odor" if use_pair_mask else ""
        whiten_tag = "_whiten-resms" if use_resms_whitening else ""
        models_tag = "-".join(models_list)  # ["identity","value"] -> "identity-value"
        output_file = (
            f"rsa_{glm_variant}_{stat_type}_{rsa_method}_cue_subset_{cue_subset}_"
            f"{models_tag}{mask_tag}_mode_{mode}{whiten_tag}.swarm"
        )

    # Discover subjects from one session folder (based on glm_variant)
    if len(sessions) == 0:
        raise ValueError("sessions must contain at least one session name.")
    discover_folder = session_folder_template.format(session=sessions[0], glm_variant=glm_variant)
    glm_path = os.path.join(base_dir, "GLM", discover_folder)

    if not os.path.isdir(glm_path):
        raise FileNotFoundError(f"GLM folder not found: {glm_path}")

    sub_list = sorted(
        [
            d
            for d in os.listdir(glm_path)
            if d.startswith("sub-") and os.path.isdir(os.path.join(glm_path, d))
        ]
    )
    if len(sub_list) == 0:
        raise RuntimeError(f"No subjects found in: {glm_path}")

    # Conda init script
    conda_sh = os.path.join(conda_root, "etc", "profile.d", "conda.sh")
    if not os.path.isfile(conda_sh):
        raise FileNotFoundError(
            f"conda.sh not found at: {conda_sh}. "
            f"Check conda_root={conda_root} (expected your conda install root)."
        )

    # Write swarm file
    with open(output_file, "w") as f:
        for sub in sub_list:
            for session in sessions:
                # prefix: conda activation + optional thread caps + optional PYTHONPATH
                prefix_parts = [
                    f"source {conda_sh}",
                    f"conda activate {conda_env_path}",
                ]

                if cap_threads:
                    prefix_parts += [
                        "export OMP_NUM_THREADS=1",
                        "export MKL_NUM_THREADS=1",
                        "export OPENBLAS_NUM_THREADS=1",
                        "export NUMEXPR_NUM_THREADS=1",
                    ]

                if pythonpath:
                    prefix_parts.append(f"export PYTHONPATH={pythonpath}:$PYTHONPATH")

                # main python command (aligned to run_rsa.py CLI)
                cmd_parts = [
                    "python", "-m", module_name,
                    "--sub", sub,
                    "--session", session,
                    "--base_dir", base_dir,
                    "--mask", mask_path,
                    "--radius", str(radius_mm),
                    "--n_jobs", str(n_jobs),
                    "--rsa_method", rsa_method,
                    "--cue_subset", cue_subset,
                    "--models", *models_list,
                    "--stat_type", stat_type,
                    "--glm_variant", glm_variant,
                    "--mode", mode,
                    "--neural_metric", neural_metric,
                ]

                if use_pair_mask:
                    cmd_parts.append("--use_pair_mask")

                if use_resms_whitening:
                    cmd_parts.append("--use_resms_whitening")
                    # keep eps explicit for provenance/reproducibility
                    cmd_parts += ["--whitening_eps", str(whitening_eps)]

                # Join into one swarm line using ';' so each part runs in the same shell
                full_line = "; ".join(prefix_parts + [" ".join(cmd_parts)])
                f.write(full_line + "\n")

    total_jobs = len(sub_list) * len(sessions)
    print(f"✅ Swarm file saved to: {output_file}")
    print(f"📌 {total_jobs} jobs written for {len(sub_list)} subjects × {len(sessions)} sessions")
    print()
    print("Parameters:")
    print(f"  base_dir:           {base_dir}")
    print(f"  glm_path:           {glm_path}")
    print(f"  mask:               {mask_path}")
    print(f"  stat_type:          {stat_type}")
    print(f"  glm_variant:        {glm_variant}")
    print(f"  mode:               {mode}")
    print(f"  neural_metric:      {neural_metric}")
    print(f"  radius:             {radius_mm} mm")
    print(f"  n_jobs:             {n_jobs}")
    print(f"  rsa_method:         {rsa_method}")
    print(f"  cue_subset:         {cue_subset}")
    print(f"  models:             {models_list}")
    print(f"  use_pair_mask:      {use_pair_mask}")
    print(f"  use_resms_whitening:{use_resms_whitening}")
    if use_resms_whitening:
        print(f"  whitening_eps:      {whitening_eps}")
    print()
    print("Conda (swarm prefix):")
    print(f"  conda_root:         {conda_root}")
    print(f"  conda_env:          {conda_env_path}")
    if pythonpath:
        print(f"  PYTHONPATH:         {pythonpath}")
    print()
    # Recommendation: when cap_threads=True, swarm -t 1 is usually safest
    rec_t = 1 if cap_threads else max(1, n_jobs)
    print("Submit with (example):")
    print(f"  swarm -f {output_file} -g 16 -t {rec_t} --logdir logs_rsa")
    return output_file


if __name__ == "__main__":
    generate_swarm_file()
