#!/usr/bin/env python3
"""Compile a one-page poster, render its preview, and bundle local TeX inputs."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


def executable(name):
    found = shutil.which(name)
    if not found and name in {"xelatex", "kpsewhich"}:
        candidate = Path("/Library/TeX/texbin") / name
        if candidate.is_file():
            found = str(candidate)
    if not found:
        raise RuntimeError(f"Required program is missing: {name}")
    return found


def run(command, cwd):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, errors="replace",
        timeout=180, check=False,
    )
    return result.returncode, result.stdout + result.stderr


def local_inputs(recorder, source, project_root, build_dir, kpsewhich):
    """Use TeX's actual input list rather than an includegraphics regex."""
    inputs = {source}
    for line in recorder.read_text(errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        path = Path(line[6:])
        if not path.is_absolute():
            path = source.parent / path
        path = path.resolve()
        if not path.is_file() or path.is_relative_to(build_dir):
            continue
        if path.is_relative_to(project_root):
            inputs.add(path)
        elif path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".eps"}:
            # Beamer loads distribution-owned PDF icons even with navigation hidden.
            status, installed = run([kpsewhich, path.name], source.parent)
            if status or Path(installed.strip()).resolve() != path:
                raise RuntimeError(f"Figure lies outside --project-root: {path}. Choose a root containing it.")
    return sorted(inputs)


def build(args):
    source = args.source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".tex":
        raise RuntimeError("Provide an existing .tex source file.")
    project_root = (args.project_root or source.parent).expanduser().resolve()
    if not source.is_relative_to(project_root):
        raise RuntimeError("The source must be inside --project-root.")
    if args.preview_width <= 0:
        raise RuntimeError("--preview-width must be positive.")
    xelatex, kpsewhich, pdfinfo, pdftoppm = (
        executable(n) for n in ("xelatex", "kpsewhich", "pdfinfo", "pdftoppm")
    )
    output = (args.output_dir or source.parent).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"{source.stem}_build.log"
    transcript = []

    with tempfile.TemporaryDirectory(prefix="sfn-poster-build-") as temporary:
        build_dir = Path(temporary).resolve()
        for number in (1, 2):
            status, text = run([
                xelatex, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
                "-recorder", f"-output-directory={build_dir}", source.name,
            ], source.parent)
            transcript.append(f"PASS {number}\n{text}")
            log_path.write_text("\n".join(transcript))
            if status:
                raise RuntimeError(f"XeLaTeX pass {number} failed. See {log_path}\n{text[-4000:]}")

        tex_log = (build_dir / f"{source.stem}.log").read_text(errors="replace")
        if re.search(r"Overfull \\[hv]box", tex_log):
            raise RuntimeError(f"Overfull content detected; adjust the layout. See {log_path}")
        if "Rerun to get" in tex_log or "undefined references" in tex_log:
            raise RuntimeError(f"Unresolved references remain after two passes. See {log_path}")
        pdf = build_dir / f"{source.stem}.pdf"
        status, info = run([pdfinfo, str(pdf)], source.parent)
        page_count = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
        if status or not page_count or int(page_count.group(1)) != 1:
            raise RuntimeError(f"Expected one poster page. pdfinfo reported:\n{info}")

        preview_stem = build_dir / f"{source.stem}_preview"
        status, render_log = run([
            pdftoppm, "-f", "1", "-singlefile", "-scale-to-x", str(args.preview_width),
            "-scale-to-y", "-1", "-png", str(pdf), str(preview_stem),
        ], source.parent)
        preview = preview_stem.with_suffix(".png")
        if status or not preview.is_file():
            raise RuntimeError(f"Preview rendering failed:\n{render_log}")

        inputs = local_inputs(
            build_dir / f"{source.stem}.fls", source, project_root, build_dir, kpsewhich,
        )
        archive = build_dir / f"{source.stem}_source.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in inputs:
                bundle.write(path, path.relative_to(project_root).as_posix())
        with zipfile.ZipFile(archive) as bundle:
            if bundle.testzip() is not None:
                raise RuntimeError("Source bundle integrity check failed.")
            for path in inputs:
                if bundle.read(path.relative_to(project_root).as_posix()) != path.read_bytes():
                    raise RuntimeError(f"Source changed during packaging: {path}")

        for artifact in (pdf, preview, archive):
            destination = output / artifact.name
            shutil.copy2(artifact, destination)
            print(destination)
        print(f"Build log: {log_path}")
        print(f"Compiled twice; one page; no overfull boxes; bundled {len(inputs)} local inputs.")
        print("Inspect the preview and verify the scientific content before delivery.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Poster .tex file")
    parser.add_argument("--project-root", type=Path, help="Root containing local source assets")
    parser.add_argument("--output-dir", type=Path, help="Destination (default: source folder)")
    parser.add_argument("--preview-width", type=int, default=3200, help="PNG width in pixels")
    args = parser.parse_args()
    try:
        build(args)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, zipfile.BadZipFile) as error:
        print(f"Poster build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
