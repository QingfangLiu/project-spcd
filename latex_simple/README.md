# Simple LaTeX manuscript for Overleaf and PDF-to-Word conversion

This folder contains a plain template, a Python converter, and a converted copy
of the supplied PNAS manuscript. The original manuscript is untouched. No Pandoc
or Python packages are required.

## Use the converted manuscript

1. Upload `simple_latex_bundle.zip` to a new Overleaf project, or upload this
   folder's `main.tex` and `fig/` folder.
2. Add the two missing original assets:
   - `fig/adaptive_identity_allocation.pdf`
   - `sn-bibliography.bib` (in the project root)
3. Set `main.tex` as the main document and choose **pdfLaTeX**. XeLaTeX and
   LuaLaTeX are also supported by the template's font setup.
4. Recompile. Allow Overleaf to run BibTeX and resolve references; recompile
   again if citations or figure references still show question marks.
5. Download the PDF and use your usual PDF-to-Word conversion workflow.

The three included figure PDFs were copied from this repository's `figures/`
folder by matching filenames. If those figures differ from your current
Overleaf originals, replace them with your original figure PDFs.

The missing fourth figure and bibliography were not invented or replaced.
Consequently, the delivered manuscript needs those files before it can compile
fully. `main.conversion.json` records the missing assets and preservation checks.
The separate `simple_template.tex` is a compilable blank/example manuscript.

## Convert future revisions

Run this from the folder containing `convert_pnas.py` and `simple_template.tex`:

```sh
python3 convert_pnas.py /path/to/pnas-project/main.tex -o /path/to/simple-project/main.tex
```

The script copies referenced figure and bibliography files from beside the
input manuscript, preserving relative paths such as `fig/fig_design.pdf`.
An additional asset directory can be supplied:

```sh
python3 convert_pnas.py /path/to/main.tex -o /path/to/simple-project/main.tex --asset-dir /path/to/figures
```

For this manuscript in this repository, run from the repository root:

```sh
python3 latex_simple/convert_pnas.py /Users/liuq13/Downloads/main.tex -o latex_simple/main.tex --asset-dir figures --overwrite
```

Existing output is protected unless `--overwrite` is supplied. Existing asset
files with different contents are never overwritten automatically. Each run
creates a `.conversion.json` report next to the output. A successful conversion
can still report missing assets; check `compilation_ready` in the report.
That flag checks file availability, not compilation or bibliography completeness.

## What the converter does

- Uses standard `article`, 12-point Latin Modern text, one column, one-inch
  margins, a ragged right edge, and page numbers. No PNAS class is needed.
- Prints authors, affiliation markers, affiliations, abstract, significance
  statement, keywords, correspondence, contributions, and competing interests
  as ordinary text blocks. It preserves all supplied wording.
- Adds an Introduction heading before the unheaded introductory text.
- Turns PNAS methods, data availability, and acknowledgment wrappers into
  ordinary sections. Removes PNAS display commands, column split controls,
  article classification, running-footer metadata, compilation date, and DOI
  layout metadata; removed values are recorded where applicable in the report.
- Keeps equations, mathematical notation, citation keys, labels, cross-reference
  commands, table cells, graphics paths, captions, and body text.
- Places figures and tables in source order as ordinary blocks, with numbered
  captions. Long captions can break across pages. Labels and `\autoref` continue
  to use LaTeX's figure/table counters. Table text uses a slightly smaller size
  and reduced cell padding; its contents are unchanged.
- Uses the standard `unsrt` bibliography style: numbered references in order of
  first citation. Reference typography will differ from PNAS's style.

The script parses nested braces instead of using a single replacement pattern.
It checks semantic objects and ordinary text spans before writing the output.
These source checks do not establish PDF appearance, validate the science, or
guarantee a particular Word converter's behavior. Equations and table structure
may still need cleanup after PDF-to-Word conversion. Figure artwork remains
figure artwork, rather than editable Word shapes.

## Scope and checks

The converter targets the `pnas-new` structure used in the supplied manuscript.
It is not a general TeX interpreter. External `\input`/`\include` files, custom
preambles, unfamiliar conditional blocks, and verbatim content stop conversion
for manual review. Other body commands are retained and may need their own
packages if introduced in future revisions. Keep the original PNAS project as
your submission source; rerun conversion whenever it changes.

Run the regression checks from this folder:

```sh
python3 -m unittest -v test_convert_pnas.py
```

Validation completed: nine regression checks passed. The blank template compiled
successfully using the desktop compiler (XeLaTeX). A separate temporary copy of
the full manuscript also compiled, with explicit notices in place of the
unavailable fourth figure and bibliography. Those notices are not in the
delivered manuscript. Final reference resolution and appearance should be
checked in Overleaf after you add the missing originals.
