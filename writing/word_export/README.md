# Simple manuscript and supplement PDFs for opening in Word

Edit **`writing/main/main.tex`** and **`writing/supp/supp.tex`** as the authoritative
manuscript and supplement. This folder generates plain, single-column versions
for PDF-to-Word conversion. Do not edit the generated `main.tex` or `supp.tex`
here directly: the next refresh replaces them. The sources and PDFs under
`writing/main/` and `writing/supp/` continue to use the original PDF figures;
conversion never modifies those files. Only the Word exports use PNG figures.

## Refresh and build locally

From the repository root, run:

```sh
python3 writing/word_export/build_simple.py
```

To refresh the supplement or both documents:

```sh
python3 writing/word_export/build_simple.py --target supp
python3 writing/word_export/build_simple.py --target all
```

The default target remains `main`. The supplement target reads
`writing/supp/supp.tex` and generates `supp.tex`, `supp.conversion.json`, and
**`writing/word_export/supp.pdf`** here, using the same layout and build checks.

Or run from any folder using the script's full path:

```sh
python3 /Users/liuq13/project-spcd/writing/word_export/build_simple.py
```

This regenerates `main.tex` and `main.conversion.json` here, then builds
**`writing/word_export/main.pdf`**, running BibTeX and the necessary LaTeX passes
through `latexmk`. It requires Python 3 and a TeX installation with `latexmk` and
`pdflatex`; no Python packages are needed. MacTeX at `/Library/TeX/texbin` is
detected even when it is not on your shell's PATH.

Open the resulting PDF in Word using your usual PDF-to-Word workflow. After any
manuscript revision, rerun the command. This is an on-demand refresh, not an
automatic background sync. Typesetting the generated `main.tex` in TeXShop alone
does not refresh it from the PNAS source; run the script first.

Figures use matching PNG files directly from the shared `fig/` (main) and
`supp_fig/` (supplement) folders. Both bibliographies come from
`writing/main/sn-bibliography.bib`. No figure or bibliography copies are created.
PNG artwork keeps each figure as a single raster image in
the PDF, reducing fragmentation when Word imports vector graphics. Refresh
stops if a matching PNG is missing. Keep the repository folder structure intact.

LaTeX creates temporary files alongside the PDF during a build (`.aux`, `.bbl`,
`.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, and `.synctex.gz`). These can be
removed after a successful build and will be regenerated when needed. Keep
`main.pdf`, `main.tex`, and `main.conversion.json`, and the corresponding `supp.*`
files, as the current outputs.

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
  captions. Long captions can continue across pages. Figures use PNG artwork;
  their labels and panels are embedded together in each image.
- References use `unsrt`: numbered in first-citation order. Their typography
  differs from the PNAS submission style.
- The supplement retains its methods and results headings, figure page breaks,
  and S-prefixed figure and table numbering. It has a plain title block without
  the PNAS cover page and does not require an abstract or add an Introduction.

The conversion report records the source hash, preservation checks, shared asset
paths, and build status. Source checks verify content preservation; the build
also checks for unresolved citations and references. Equations, tables, and page
breaks may still need cleanup after Word imports the PDF.

## Converter options and checks

To regenerate the plain source without building its PDF:

```sh
python3 writing/word_export/convert_pnas.py writing/main/main.tex \
  -o writing/word_export/main.tex --asset-mode link --figure-format png --overwrite
python3 writing/word_export/convert_pnas.py writing/supp/supp.tex \
  -o writing/word_export/supp.tex --document-kind supp \
  --asset-mode link --figure-format png --overwrite
```

`link` mode requires all assets to exist and permits parent-relative paths.
The default `copy` mode remains available for a separate, self-contained project
whose source uses project-relative asset paths without `..` components. Extra
asset search folders can be passed with `--asset-dir`. Conflicting asset copies
stop conversion. Existing output is protected unless `--overwrite` is supplied.
The converter itself keeps source figure formats unless `--figure-format png`
is supplied; the refresh/build command always supplies it. PNG files must have
the same basename and folder as the resolved source figures. Regenerate those
PNGs when updating the artwork; this command does not rasterize PDF figures.

The converter targets the `pnas-new` structures used by these documents, rather
than arbitrary LaTeX. Supplement conversion accepts the existing `graphicx`,
`amsmath`, and `placeins` packages and S-prefixed figure/table counter setup.
External `\input`/`\include` files, other custom preambles,
unfamiliar conditional blocks, and verbatim content stop conversion for review.
Other retained body commands may need additional packages if introduced later.
`simple_template.tex` controls the plain layout and also serves as a standalone
example.

Run the regression checks:

```sh
cd writing/word_export
python3 -m unittest -v test_convert_pnas.py
```
