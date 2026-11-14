#!/usr/bin/env python3
"""Convert a pnas-new manuscript to ordinary LaTeX; Python standard library only.

This is a targeted source translator, not a general TeX interpreter. Balanced
groups are parsed explicitly. Unhandled front matter and external TeX inputs
are rejected rather than silently dropped. The source file is never modified.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


class ConversionError(ValueError):
    pass


def strip_comments(text):
    """Remove TeX comments, respecting escaped percent signs and TeX newlines."""
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            out.append(text[i:i + 2])
            i += 2
        elif text[i] == "%":
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def skip_space(text, pos):
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def group(text, pos, opener="{"):
    pos = skip_space(text, pos)
    if pos >= len(text) or text[pos] != opener:
        raise ConversionError(f"Expected {opener!r} near {text[pos:pos + 70]!r}")
    start = pos + 1
    closer = "}" if opener == "{" else "]"
    depth = 1
    pos += 1
    while pos < len(text):
        ch = text[pos]
        if ch == "\\":
            pos += 2
            continue
        if opener == "[" and ch == "{":
            _, pos = group(text, pos)
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:pos], pos + 1
        pos += 1
    raise ConversionError(f"Unclosed {opener!r} at character {start - 1}")


@dataclass
class Command:
    name: str
    start: int
    end: int
    args: tuple
    option: str = ""


ARG_COUNTS = {
    "documentclass": 1, "title": 1, "author": 1, "affil": 1,
    "articletype": 1, "templatetype": 1, "leadauthor": 1,
    "significancestatement": 1, "authorcontributions": 1,
    "authordeclaration": 1, "correspondingauthor": 1, "keywords": 1,
    "dates": 1, "doi": 1, "matmethods": 1, "dataavail": 1, "acknow": 1,
    "showmatmethods": 1, "showacknow": 1, "thispagestyle": 1,
    "ifthenelse": 3, "bibliographystyle": 1,
    "usepackage": 1, "renewcommand": 2, "DeclareRobustCommand": 2,
}
OPTIONAL = {"documentclass", "author", "affil", "bibsplit", "usepackage"}
LAYOUT = {
    "articletype", "templatetype", "leadauthor", "dates", "doi",
    "thispagestyle", "Firstpage", "Endparasplit", "Parasplit", "bibsplit",
    "showmatmethods", "showacknow", "bibliographystyle",
}
FRONT = {
    "title", "author", "affil", "significancestatement", "authorcontributions",
    "authordeclaration", "correspondingauthor", "keywords",
}
SECTIONS = {
    "matmethods": "Materials and methods", "dataavail": "Data availability",
    "acknow": "Acknowledgments",
}
REJECT = {"input", "include", "includeonly", "endinput", "verbatim", "verb"}
CITATION_COMMANDS = {"cite", "citep", "citet", "citealp", "citeyearpar", "citeyear", "citeauthor"}


def commands(text):
    pos = 0
    while pos < len(text):
        start = text.find("\\", pos)
        if start < 0:
            return
        match = re.match(r"\\([A-Za-z]+|[^\n])", text[start:])
        if not match:
            pos = start + 1
            continue
        name = match[1]
        end = start + len(match[0])
        option = ""
        if name in OPTIONAL and text[skip_space(text, end):].startswith("["):
            option, end = group(text, end, "[")
        args = []
        for _ in range(ARG_COUNTS.get(name, 0)):
            value, end = group(text, end)
            args.append(value)
        yield Command(name, start, end, tuple(args), option)
        pos = end


def replace_spans(text, replacements):
    for start, end, value in sorted(replacements, reverse=True):
        text = text[:start] + value + text[end:]
    return text


def nested_commands(text):
    for cmd in commands(text):
        yield cmd
        for argument in cmd.args:
            yield from nested_commands(argument)


def plain_space(text):
    return re.sub(r"\s+", " ", text).strip()


def canonical_citations(text):
    """Ignore citation presentation when checking that source content survived."""
    return re.sub(r"\\(?:" + "|".join(sorted(CITATION_COMMANDS)) + r")\b",
                  lambda _: r"\cite", text)


def author_year_citations(text):
    """Change citation commands only; preserve keys, notes, and surrounding prose."""
    edits = []
    counts = {"parenthetical": 0, "narrative_year": 0}
    for cmd in commands(text):
        if cmd.name != "cite":
            continue
        end = cmd.end
        has_notes = False
        while text[skip_space(text, end):].startswith("["):
            has_notes = True
            _, end = group(text, end, "[")
        keys, _ = group(text, end)
        narrative = (not has_notes and "," not in keys
                     and re.search(r"\bet\s+al\.(?:\s|\\\s|~)*$", text[:cmd.start]))
        name = "citeyearpar" if narrative else "citep"
        counts["narrative_year" if narrative else "parenthetical"] += 1
        edits.append((cmd.start, cmd.end, "\\" + name))
    return replace_spans(text, edits), counts


def content_items(text):
    """Fingerprint semantic objects that must survive the layout translation."""
    items = {"citations": [], "labels": [], "references": [], "equations": [],
             "tables": [], "captions": [], "graphics": [], "bibliography": [],
             "highlights": []}
    for cmd in commands(text):
        names = {"cite": "citations", "label": "labels", "ref": "references",
                 "autoref": "references", "eqref": "references",
                 "caption": "captions", "includegraphics": "graphics",
                 "bibliography": "bibliography", "hl": "highlights"}
        names.update({name: "citations" for name in CITATION_COMMANDS})
        if cmd.name in names:
            end = cmd.end
            while text[skip_space(text, end):].startswith("["):
                _, end = group(text, end, "[")
            value, _ = group(text, end)
            items[names[cmd.name]].append(plain_space(value))
        elif cmd.name in SECTIONS:
            nested = content_items(cmd.args[0])
            for key in items:
                items[key].extend(nested[key])
    for env, key in [("equation", "equations"), ("tabular", "tables")]:
        items[key] = [plain_space(x) for x in re.findall(
            r"\\begin\{" + env + r"\*?\}(.*?)\\end\{" + env + r"\*?\}",
            text, flags=re.S)]
    return items


def flatten_floats(text):
    """Keep source order; let long captions break as ordinary paragraphs."""
    pattern = re.compile(r"\\begin\{(figure\*?|table\*?)\}(?:\s*\[[^\]]*\])?(.*?)"
                         r"\\end\{\1\}", re.S)

    def block(match):
        kind = match[1].rstrip("*")
        body = match[2]
        if len(re.findall(r"\\caption\b", body)) != 1:
            raise ConversionError(f"Each {kind} must have exactly one caption.")
        edits = []
        for cmd in commands(body):
            if cmd.name == "centering":
                edits.append((cmd.start, cmd.end, ""))
            elif cmd.name == "caption":
                caption, end = group(body, cmd.end)
                edits.append((cmd.start, end, "\\par\\noindent\\textbf{"
                              + kind.title() + "~\\the" + kind + ". }"
                              + caption + "\\par\n"))
            elif cmd.name == "includegraphics":
                end = cmd.end
                if body[skip_space(body, end):].startswith("["):
                    _, end = group(body, end, "[")
                _, end = group(body, end)
                edits.append((cmd.start, end, "\\begin{center}\n"
                              + body[cmd.start:end] + "\n\\end{center}"))
        body = replace_spans(body, edits)
        if kind == "table":
            # Font and padding changes only; all original table cells survive.
            body = body.replace(r"\begin{tabular}",
                                r"\small\setlength{\tabcolsep}{4pt}\begin{tabular}")
        return ("\n\\par\\bigskip\\begingroup\n\\refstepcounter{" + kind + "}\n"
                + body + "\n\\par\\endgroup\\bigskip\n")

    return pattern.sub(block, text)


def reflow_highlighted_citations(text):
    """Allow boxed author-year citation lists to wrap between references."""
    highlighted = []
    for cmd in commands(text):
        if cmd.name == "hl":
            _, end = group(text, cmd.end)
            highlighted.append((cmd.end, end))
    edits = []
    for cmd in commands(text):
        if cmd.name != "mbox":
            continue
        boxed, end = group(text, cmd.end)
        if not any(start <= cmd.start and end <= stop for start, stop in highlighted):
            continue
        boxed = boxed.strip()
        citation = next(commands(boxed), None)
        if (citation is None or citation.start != 0
                or citation.name not in {"citep", "citealp"}):
            continue
        if not boxed[skip_space(boxed, citation.end):].startswith("{"):
            continue  # Preserve optional citation notes verbatim.
        keys, citation_end = group(boxed, citation.end)
        if boxed[citation_end:].strip():
            continue
        keys = [key.strip() for key in keys.split(",")]
        if len(keys) < 2 or not all(re.fullmatch(r"[\w:.-]+", key) for key in keys):
            continue
        value = "; ".join(r"\mbox{\citealp{" + key + "}}" for key in keys)
        if citation.name == "citep":
            value = "(" + value + ")"
        edits.append((cmd.start, end, value))
    output = replace_spans(text, edits)
    def cited_keys(value):
        return Counter(key.strip() for keys in content_items(value)["citations"]
                       for key in keys.split(","))
    if cited_keys(text) != cited_keys(output):
        raise ConversionError("Citation keys changed while wrapping highlighted citations.")
    return output, len(edits)


def convert(source, template, document_kind="main"):
    if document_kind not in {"main", "supp"}:
        raise ConversionError(f"Unknown document kind: {document_kind}")
    supplement = document_kind == "supp"
    clean = strip_comments(source)
    if not re.search(r"\\documentclass(?:\[[^\]]*\])?\{pnas-new\}", clean):
        raise ConversionError("Expected a pnas-new documentclass.")
    if clean.count(r"\begin{document}") != 1 or clean.count(r"\end{document}") != 1:
        raise ConversionError("Expected exactly one document environment.")
    if clean.split(r"\end{document}", 1)[1].strip():
        raise ConversionError("Content after end{document} requires manual review.")
    if re.search(r"\\begin\{(?:verbatim|lstlisting|minted)\}", source):
        raise ConversionError("Verbatim content requires manual review.")
    found = list(commands(clean))
    for cmd in nested_commands(clean):
        if cmd.name in REJECT:
            raise ConversionError(f"External or verbatim content: \\{cmd.name}.")
    preamble, document = clean.split(r"\begin{document}", 1)
    body = document.split(r"\end{document}", 1)[0]
    preamble_edits = [(c.start, c.end, "") for c in commands(preamble)
                      if c.name == "documentclass" or c.name in LAYOUT or c.name in FRONT]
    if supplement:
        if not any(c.name == "templatetype" and c.args == ("pnassupportinginfo",) for c in found):
            raise ConversionError("Expected the pnassupportinginfo template for a supplement.")
    supported_packages = {"soul", "xcolor"}
    if supplement:
        supported_packages |= {"graphicx", "amsmath", "placeins"}
    for cmd in commands(preamble):
        # Review-marking packages are handled by the plain template. Supplement
        # float packages and numbering retain their existing treatment.
        if (cmd.name == "usepackage" and not cmd.option
                and set(cmd.args[0].replace(" ", "").split(",")) <= supported_packages):
            preamble_edits.append((cmd.start, cmd.end, ""))
        elif (supplement and cmd.name == "renewcommand" and len(cmd.args) == 2
              and any(tuple(plain_space(x) for x in cmd.args) == ("\\the" + kind, "S\\arabic{" + kind + "}")
                      for kind in ("figure", "table"))):
            preamble_edits.append((cmd.start, cmd.end, ""))
    if replace_spans(preamble, preamble_edits).strip():
        raise ConversionError("Custom preamble content requires manual review; it was not discarded.")
    metadata = {key: [] for key in FRONT}
    for cmd in found:
        if cmd.name in FRONT:
            metadata[cmd.name].append(cmd)
    for key in FRONT - {"author", "affil"}:
        if len(metadata[key]) > 1:
            raise ConversionError(f"Multiple \\{key} commands require manual review.")
    if not metadata["title"]:
        raise ConversionError("No manuscript title found.")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    abstract_count = len(re.findall(r"\\begin\{abstract\}", body))
    if abstract_count > 1 or (not supplement and not abstract_match):
        raise ConversionError("Expected exactly one abstract.")
    abstract = abstract_match[1].strip() if abstract_match else ""
    edits = [(abstract_match.start(), abstract_match.end(), "")] if abstract_match else []
    removed = [{"command": c.name, "argument": list(c.args), "option": c.option}
               for c in found if c.name in LAYOUT]
    for cmd in commands(body):
        if abstract_match and abstract_match.start() <= cmd.start < abstract_match.end():
            continue
        value = None
        if cmd.name in FRONT or cmd.name == "maketitle":
            value = ""
        elif cmd.name in LAYOUT:
            value = ""
        elif supplement and cmd.name in {"SItext", "FloatBarrier"}:
            value = ""
        elif cmd.name == "ifthenelse":
            compact = re.sub(r"\s+", "", body[cmd.start:cmd.end])
            expected = (r"\ifthenelse{\boolean{shortarticle}}{\ifthenelse{\boolean{singlecolumn}}"
                        r"{\abscontentformatted}{\abscontent}}{}")
            if compact != expected:
                raise ConversionError("Unrecognized conditional requires manual review.")
            value = ""
        elif cmd.name in SECTIONS:
            value = "\n\\section*{" + SECTIONS[cmd.name] + "}\n" + cmd.args[0] + "\n"
        if value is not None:
            edits.append((cmd.start, cmd.end, value))
    body = replace_spans(body, edits).strip()
    # Confirm all ordinary body spans survived exactly before layout changes.
    cursor = 0
    for start, end, _ in sorted(edits):
        span = document.split(r"\end{document}", 1)[0][cursor:start]
        if span.strip() and span.strip() not in body:
            raise ConversionError("A manuscript text span did not survive conversion.")
        cursor = end
    tail = document.split(r"\end{document}", 1)[0][cursor:].strip()
    if tail and tail not in body:
        raise ConversionError("Final manuscript text did not survive conversion.")
    front = [r"\begin{center}", r"{\Large\bfseries " + metadata["title"][0].args[0]
             + r"\par}", r"\medskip"]
    if supplement:
        front.insert(1, r"{\large\bfseries Supporting Information\par}\medskip")
    author_lines = []
    for cmd in metadata["author"]:
        author_lines.append(cmd.args[0] + (r"\textsuperscript{" + cmd.option + "}" if cmd.option else ""))
    front.extend([", ".join(author_lines) + r"\par", r"\medskip"])
    for cmd in metadata["affil"]:
        front.append((r"\textsuperscript{" + cmd.option + "}" if cmd.option else "")
                     + cmd.args[0] + r"\par")
    front.append(r"\end{center}")
    if abstract_match:
        front.append("\n\\section*{Abstract}\n" + abstract)
    for key, heading in [("significancestatement", "Significance statement")]:
        if metadata[key]:
            front.append("\n\\section*{" + heading + "}\n" + metadata[key][0].args[0])
    if metadata["keywords"]:
        front.append("\n\\noindent\\textbf{Keywords:} " + metadata["keywords"][0].args[0])
    for key, heading in [("correspondingauthor", "Correspondence"),
                         ("authorcontributions", "Author contributions"),
                         ("authordeclaration", "Competing interests")]:
        if metadata[key]:
            front.append("\n\\noindent\\textbf{" + heading + ":} " + metadata[key][0].args[0])
    before_layout = "\n\n".join(front) + "\n\n" + body
    for cmd in found:
        if cmd.name in SECTIONS and cmd.args[0].strip() not in before_layout:
            raise ConversionError(f"Content in \\{cmd.name} did not survive conversion.")
    expected = content_items(clean)
    actual = content_items(before_layout)
    if expected != actual:
        raise ConversionError("Content inventory mismatch before layout conversion.")
    body = flatten_floats(body)
    if r"\bibliography{" in body:
        body = body.replace(r"\bibliography{", r"\bibliographystyle{plainnat}" + "\n"
                            + r"\bibliography{", 1)
    if not supplement and not re.match(r"\\section\*?\{", body.lstrip()):
        body = "\\section*{Introduction}\n\n" + body
    if supplement:
        body = (r"\renewcommand{\thefigure}{S\arabic{figure}}" + "\n"
                + r"\renewcommand{\thetable}{S\arabic{table}}" + "\n\n" + body)
    output = template
    for name, content in [("front", "\n\n".join(front)), ("body", body)]:
        pattern = r"%<simple-" + name + r">.*?%</simple-" + name + r">"
        if len(re.findall(pattern, output, re.S)) != 1:
            raise ConversionError(f"Template needs one simple-{name} marker pair.")
        output = re.sub(pattern, lambda _: content, output, flags=re.S)
    output = re.sub(r"[ \t]+$", "", output, flags=re.M)
    output, citation_rewrites = author_year_citations(output)
    # Verify relocated metadata, abstract, wrapper contents, and captions verbatim
    # apart from whitespace and citation presentation. Figures and tables now
    # use ordinary paragraphs; citation keys and notes must remain unchanged.
    normalized = plain_space(canonical_citations(output))
    protected = [abstract]
    protected += [c.args[0] for c in found if c.name in FRONT or c.name in set(SECTIONS) - {"matmethods"}]
    protected += expected["captions"]
    for content in protected:
        if plain_space(canonical_citations(content)) not in normalized:
            raise ConversionError("Protected content did not survive conversion.")
    for key in ["equations", "tables", "citations", "labels", "references", "graphics", "bibliography", "highlights"]:
        actual_values = content_items(output)[key]
        if Counter(map(canonical_citations, actual_values)) != Counter(map(canonical_citations, expected[key])):
            raise ConversionError(f"Content check failed: {key}.")
    # Apply this layout-only change after verifying complete source passages;
    # the helper separately verifies every citation key and its multiplicity.
    output, highlight_reflows = reflow_highlighted_citations(output)
    report = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "document_kind": document_kind,
              "checks": {key: {"count": len(value), "preserved": True}
                         for key, value in expected.items()},
              "metadata_and_abstract_preserved": True,
              "ordinary_body_spans_preserved": True,
              "removed_layout_metadata": removed,
              "citation_style": "author-year",
              "bibliography_style": "plainnat",
              "citation_rewrites": citation_rewrites,
              "highlighted_citation_reflows": highlight_reflows,
              "layout": "12pt article, one column, 1-inch margins; nonfloating figures and tables; author-year plainnat bibliography",
              "note": "Checks verify source content, not PDF appearance or reference resolution."}
    return output, report


def assets_for(output):
    inventory = content_items(output)
    graphics = inventory["graphics"]
    bibs = [x.strip() + ".bib" for group_ in inventory["bibliography"]
            for x in group_.split(",")]
    return [(x, "figure") for x in graphics] + [(x, "bibliography") for x in bibs]


def link_asset_paths(output, paths):
    """Rewrite only graphics/bibliography arguments to selected asset paths."""
    edits = []
    for cmd in commands(output):
        if cmd.name not in {"includegraphics", "bibliography"}:
            continue
        end = cmd.end
        if cmd.name == "includegraphics":
            while output[skip_space(output, end):].startswith("["):
                _, end = group(output, end, "[")
        start = skip_space(output, end)
        value, end = group(output, start)
        if cmd.name == "includegraphics":
            linked = paths[value, "figure"]
        else:
            linked = ",".join(
                str(Path(paths[name.strip() + ".bib", "bibliography"]).with_suffix(""))
                for name in value.split(",")
            )
        edits.append((start + 1, end - 1, linked))
    return replace_spans(output, edits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=Path(__file__).with_name("simple_template.tex"))
    parser.add_argument("--document-kind", choices=("main", "supp"), default="main",
                        help="Convert a main manuscript (default) or PNAS supporting information.")
    parser.add_argument("--asset-dir", type=Path, action="append", default=[],
                        help="Additional folder to search for assets, including by filename; repeatable.")
    parser.add_argument("--asset-mode", choices=("copy", "link"), default="copy",
                        help="Copy assets into the output project (default), or link existing files without copying. Link mode supports parent-relative paths and requires every asset to exist.")
    parser.add_argument("--figure-format", choices=("source", "png"), default="source",
                        help="Keep source figure formats (default), or require matching PNG files for Word import.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output and report.")
    args = parser.parse_args(argv)
    try:
        source_path = args.source.resolve()
        output_path = args.output.resolve()
        report_path = output_path.with_suffix(".conversion.json")
        if output_path in {source_path, args.template.resolve()}:
            raise ConversionError("Output must differ from the input and template.")
        if not args.overwrite and (output_path.exists() or report_path.exists()):
            raise ConversionError("Output/report already exists; use --overwrite to replace it.")
        output, report = convert(source_path.read_text(encoding="utf-8-sig"),
                                 args.template.read_text(encoding="utf-8"), args.document_kind)
        report["source"] = str(source_path)
        report["output"] = str(output_path)
        report["asset_mode"] = args.asset_mode
        report["figure_format"] = args.figure_format
        asset_results = []
        copy_plan = []
        linked_paths = {}
        for relative, kind in assets_for(output):
            rel = Path(relative)
            if args.asset_mode == "copy" and (rel.is_absolute() or ".." in rel.parts):
                raise ConversionError(f"Asset must use a project-relative path: {relative}")
            candidates = [source_path.parent / rel]
            for root in args.asset_dir:
                candidates.extend([root / rel, root / rel.name])
            matches = {p.resolve() for p in candidates if p.is_file()}
            if len(matches) > 1:
                digests = {hashlib.sha256(p.read_bytes()).hexdigest() for p in matches}
                if len(digests) > 1:
                    raise ConversionError(f"Conflicting asset copies: {relative}")
            origin = next(iter(sorted(matches)), None)
            if kind == "figure" and args.figure_format == "png":
                if origin is None or not origin.with_suffix(".png").is_file():
                    raise ConversionError(f"Matching PNG figure not found: {relative}")
                origin = origin.with_suffix(".png")
                rel = rel.with_suffix(".png")
            destination = output_path.parent / rel
            if args.asset_mode == "link":
                if origin is None:
                    raise ConversionError(f"Shared asset not found: {relative}")
                linked = Path(os.path.relpath(origin, output_path.parent)).as_posix()
                linked_paths[relative, kind] = linked
                asset_results.append({"path": linked, "source_path": relative,
                                      "kind": kind, "status": "linked", "origin": str(origin)})
                continue
            if destination.exists():
                if origin and destination.resolve() != origin and destination.read_bytes() != origin.read_bytes():
                    raise ConversionError(f"Existing asset differs; resolve manually: {destination}")
                state = "present"
            elif origin:
                state = "copied"
                copy_plan.append((origin, destination))
            else:
                state = "missing"
            linked_paths[relative, kind] = rel.as_posix()
            asset_results.append({"path": rel.as_posix(), "source_path": relative,
                                  "kind": kind, "status": state,
                                  "origin": str(origin) if origin else None})
        if args.asset_mode == "link" or args.figure_format == "png":
            output = link_asset_paths(output, linked_paths)
            report["asset_path_rewrites"] = [
                {"from": item["source_path"], "to": item["path"], "kind": item["kind"]}
                for item in asset_results
            ]
        report["assets"] = asset_results
        report["compilation_ready"] = all(x["status"] != "missing" for x in asset_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for origin, destination in copy_plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
        output_path.write_text(output, encoding="utf-8")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Created {output_path}\nContent checks passed. Report: {report_path}")
        for item in asset_results:
            print(f"{item['status'].upper()}: {item['path']}")
        if not report["compilation_ready"]:
            print("Add the missing assets before compiling.")
        return 0
    except (ConversionError, OSError) as exc:
        print(f"Conversion stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
