# Simple manuscript PDF for opening in Word

Edit **`writing/main/main.tex`** as the authoritative manuscript. This folder
generates a plain, single-column version from that source for PDF-to-Word
conversion. Do not edit the generated `writing/latex_simple/main.tex` directly:
the next refresh replaces it. The PNAS source is never modified by conversion.

## Refresh and build locally

From the repository root, run:

```sh
python3 writing/latex_simple/build_simple.py
```

Or run from any folder using the script's full path:

```sh
python3 /Users/liuq13/project-spcd/writing/latex_simple/build_simple.py
```

This regenerates `main.tex` and `main.conversion.json` here, then builds
**`writing/latex_simple/main.pdf`**, running BibTeX and the necessary LaTeX passes
through `latexmk`. It requires Python 3 and a TeX installation with `latexmk` and
`pdflatex`; no Python packages are needed. MacTeX at `/Library/TeX/texbin` is
detected even when it is not on your shell's PATH.

Open the resulting PDF in Word using your usual PDF-to-Word workflow. After any
manuscript revision, rerun the command. This is an on-demand refresh, not an
automatic background sync. Typesetting the generated `main.tex` in TeXShop alone
does not refresh it from the PNAS source; run the script first.

Figures are read directly from the repository's shared `fig/` folder, and the
bibliography from `writing/main/sn-bibliography.bib`. No figure or bibliography
copies are created. Keep the repository folder structure intact. The supplement
is not part of this conversion.

LaTeX creates temporary files alongside the PDF during a build (`.aux`, `.bbl`,
`.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, and `.synctex.gz`). These can be
removed after a successful build and will be regenerated when needed. Keep
`main.pdf`, `main.tex`, and `main.conversion.json` as the current output.

## Layout and content

- Standard `article`, 12-point Latin Modern text, one column, one-inch margins,
  a ragged right edge, and page numbers. No PNAS class is needed.
- Authors, affiliations, abstract, significance statement, keywords,
  correspondence, contributions, and competing interests appear as ordinary
  text blocks, retaining the source wording.
- An Introduction heading precedes the unheaded introduction. PNAS methods,
  data availability, and acknowledgment wrappers become ordinary sections.
  PNAS-specific display controls and running-footer metadata are removed.
- Body text, equations, citation keys, labels, cross-references, table cells,
  and captions are preserved. Asset paths are rebased to the shared files.
- Figures and tables appear in source order as ordinary blocks with numbered
  captions. Long captions can continue across pages. Figure artwork remains
  artwork, rather than editable Word shapes.
- References use `unsrt`: numbered in first-citation order. Their typography
  differs from the PNAS submission style.

The conversion report records the source hash, preservation checks, shared asset
paths, and build status. Source checks verify content preservation; the build
also checks for unresolved citations and references. Equations, tables, and page
breaks may still need cleanup after Word imports the PDF.

## Converter options and checks

To regenerate the plain source without building its PDF:

```sh
python3 writing/latex_simple/convert_pnas.py writing/main/main.tex \
  -o writing/latex_simple/main.tex --asset-mode link --overwrite
```

`link` mode requires all assets to exist and permits parent-relative paths.
The default `copy` mode remains available for a separate, self-contained project
whose source uses project-relative asset paths without `..` components. Extra
asset search folders can be passed with `--asset-dir`. Conflicting asset copies
stop conversion. Existing output is protected unless `--overwrite` is supplied.

The converter targets the `pnas-new` structure used by this manuscript, rather
than arbitrary LaTeX. External `\input`/`\include` files, custom preambles,
unfamiliar conditional blocks, and verbatim content stop conversion for review.
Other retained body commands may need additional packages if introduced later.
`simple_template.tex` controls the plain layout and also serves as a standalone
example.

Run the regression checks:

```sh
cd writing/latex_simple
python3 -m unittest -v test_convert_pnas.py
```
