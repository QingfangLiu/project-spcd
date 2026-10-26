#!/usr/bin/env python3
"""Refresh plain versions of the PNAS manuscript and supplement and build PDFs.

Run from the repository root:
    python3 writing/word_export/build_simple.py
    python3 writing/word_export/build_simple.py --target supp
    python3 writing/word_export/build_simple.py --target all

Sources are writing/main/main.tex and writing/supp/supp.tex. Generated .tex,
.conversion.json, and .pdf files live beside this script. Matching PNG figures
and the bibliography stay in their shared locations; no asset copies are created.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from convert_pnas import main as convert_main


def build_target(target, directory, executable, environment):
    source = directory.parent / target / (target + ".tex")
    output = directory / (target + ".tex")
    if convert_main([str(source), "-o", str(output), "--document-kind", target,
                     "--asset-mode", "link", "--figure-format", "png", "--overwrite"]):
        return 1
    return build_pdf(output, executable, environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("main", "supp", "all"), default="main",
                        help="Refresh the main manuscript (default), supplement, or both.")
    args = parser.parse_args(argv)
    directory = Path(__file__).resolve().parent
    executable = shutil.which("latexmk")
    if executable is None:
        mactex = Path("/Library/TeX/texbin/latexmk")
        if mactex.is_file():
            executable = str(mactex)
    if executable is None:
        print("latexmk was not found. Add your TeX installation's bin directory to PATH.", file=sys.stderr)
        return 1
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).parent) + os.pathsep + environment.get("PATH", "")
    if shutil.which("pdflatex", path=environment["PATH"]) is None:
        print("pdflatex was not found beside latexmk or on PATH.", file=sys.stderr)
        return 1
    targets = ("main", "supp") if args.target == "all" else (args.target,)
    for target in targets:
        if build_target(target, directory, executable, environment):
            return 1
    return 0


def build_pdf(output, executable, environment):
    report_path = output.with_suffix(".conversion.json")
    report = json.loads(report_path.read_text())
    result = subprocess.run(
        [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", "-synctex=1", output.name],
        cwd=output.parent, env=environment, capture_output=True, text=True, errors="replace",
    )
    log_path = output.with_suffix(".log")
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    unresolved = any(message in log for message in (
        "There were undefined citations", "There were undefined references",
    ))
    succeeded = result.returncode == 0 and output.with_suffix(".pdf").is_file() and not unresolved
    report["pdf_build"] = {
        "status": "success" if succeeded else "failed",
        "engine": "pdflatex via latexmk",
        "unresolved_references": unresolved,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not succeeded:
        print((result.stdout + "\n" + result.stderr)[-6000:], file=sys.stderr)
        print(f"Build failed or references remain unresolved. See {log_path}", file=sys.stderr)
        return 1
    print(f"Built {output.with_suffix('.pdf')}\nCitations and references resolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
