# make_searchlight_swarm_withinset_crosscue.py
# Supports:
#   - withinset_set2set   (pooled across runs)
#   - withinset_strict    (run-matched)

import os
import argparse

VALID_MODES = [
    "withinset_set2set",
    "withinset_strict",
]

def generate_swarm_file(
    base_dir: str,
    script_name: str = "run_searchlight_crosscue.py",
    output_file: str = None,
    radius_mm: float = 8.0,
    n_jobs: int = -1,
    session_folder: str = "premeal_trialwise_2mm",
    session_tag: str = "premeal",
    n_classes: int = 2,
    mode: str = "withinset_set2set",
    mask_img: str = None,
    disable_monitor: bool = False,
):
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got: {mode}")

    glm_path = os.path.join(base_dir, "GLM", session_folder)
    if not os.path.isdir(glm_path):
        raise FileNotFoundError(f"GLM folder not found: {glm_path}")

    sub_list = sorted([d for d in os.listdir(glm_path) if d.startswith("sub-")])
    if len(sub_list) == 0:
        raise RuntimeError(f"No subjects found under: {glm_path}")

    if mask_img is None:
        mask_img = os.path.join(base_dir, "masks", "gm_0.1.nii")

    if output_file is None:
        output_file = (
            f"decode_{mode}_classes-{n_classes}_"
            f"session-{session_tag}.swarm"
        )

    with open(output_file, "w") as f:
        for sub in sub_list:
            cmd = (
                f"python {script_name} "
                f"--sub {sub} "
                f"--base_dir {base_dir} "
                f"--mask_img {mask_img} "
                f"--n_classes {n_classes} "
                f"--radius_mm {radius_mm} "
                f"--n_jobs {n_jobs} "
                f"--session_folder {session_folder} "
                f"--session_tag {session_tag} "
                f"--mode {mode}"
            )
            if disable_monitor:
                cmd += " --no_monitor"
            f.write(cmd + "\n")

    print(f"✅ Swarm file written: {output_file}")
    print(f"📌 Subjects: {len(sub_list)}")
    print(f"🧠 mode={mode} | classes={n_classes}")
    print(f"📁 session={session_tag} | folder={session_folder}")
    print(f"🎭 mask={mask_img}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="/data/liuq13/SPCD_subset")
    ap.add_argument("--script_name", default="run_searchlight_crosscue.py")
    ap.add_argument("--output_file", default=None)
    ap.add_argument("--radius_mm", type=float, default=8.0)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--session_folder", default="premeal_trialwise_2mm")
    ap.add_argument("--session_tag", default="premeal")
    ap.add_argument("--n_classes", type=int, choices=[2, 3], default=2)
    ap.add_argument("--mode", choices=VALID_MODES, default="withinset_set2set")
    ap.add_argument("--mask_img", default=None)
    ap.add_argument("--disable_monitor", action="store_true")
    args = ap.parse_args()

    generate_swarm_file(
        base_dir=args.base_dir,
        script_name=args.script_name,
        output_file=args.output_file,
        radius_mm=args.radius_mm,
        n_jobs=args.n_jobs,
        session_folder=args.session_folder,
        session_tag=args.session_tag,
        n_classes=args.n_classes,
        mode=args.mode,
        mask_img=args.mask_img,
        disable_monitor=args.disable_monitor,
    )
