#!/usr/bin/env python3
"""Refresh the plain manuscript from the PNAS source and build its PDF.

Run from the repository root:
    python3 writing/word_export/build_simple.py

The source is writing/main/main.tex. The generated files are main.tex,
main.conversion.json, and main.pdf beside this script. Matching PNG figures and
the bibliography stay in their shared locations; no asset copies are created.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from convert_pnas import main as convert_main


def main():
    directory = Path(__file__).resolve().parent
    source = directory.parent / "main" / "main.tex"
    output = directory / "main.tex"
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
    if convert_main([str(source), "-o", str(output), "--asset-mode", "link",
                     "--figure-format", "png", "--overwrite"]):
        return 1
    report_path = output.with_suffix(".conversion.json")
    report = json.loads(report_path.read_text())
    result = subprocess.run(
        [executable, "-pdf", "-interaction=nonstopmode", "-halt-on-error",
         "-file-line-error", "-synctex=1", output.name],
        cwd=directory, env=environment, capture_output=True, text=True, errors="replace",
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
