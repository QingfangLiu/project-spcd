---
name: sfn-poster-style
description: Create or update scientific conference posters in the user's SfN style, with a three-column landscape layout, teal headings, outlined panels, complete figures, and editable LaTeX plus PDF and preview outputs. Use for requests for this poster style, including future studies; preserve a different style when the user requests one.
---

# SfN Poster Style

Use the established visual system while adapting scientific content and panel sizes to the current study. This skill is saved under `skills/sfn-poster-style/` for review; it is not registered for automatic discovery or installed as a personal template. To use it from this location, ask Codex to read this file explicitly. Moving it to `.agents/skills/` for automatic repository discovery is a separate action requiring the user's request.

## Reference and starting point

- Inspect [assets/reference.png](assets/reference.png) for the visual target, recorded after the November 9, 2026 refresh.
- Read [references/style-guide.md](references/style-guide.md) for colors, typography, coordinates, and layout decisions.
- Start a new poster from [assets/poster-template.tex](assets/poster-template.tex). Copy it into the requested output project before editing. Its figure placeholders are scaffolding, not finished scientific content.
- [assets/reference-source.zip](assets/reference-source.zip) retains the original source and five complete figure PDFs. It is a historical style reference; its statistics, title, authors, funding, and figures are not defaults for new studies.
- For an existing poster, edit its current source in place; do not replace completed content with the blank template.

## Ground the scientific content

Read applicable repository instructions before edits, including any approval gate. Existing authorization applies only within its approved scope. Respect requested titles, conference dimensions, author order, and submitted presentation details.

Find the current manuscript, figure sources, and relevant result summaries. When checking whether a poster is current, inspect recent changes and the compiled poster and its source bundle: a current figure on disk does not update an already compiled PDF. Treat uncommitted results as provisional unless the user has adopted them. Do not rerun analyses just to make a poster.

Before writing, map each result panel to its current source figure and record the relevant sample size, effect, test direction, and correction. Use only supported numbers. Distinguish model simulations from empirical results, and avoid interpreting a significant effect in one condition and a nonsignificant effect in another as a tested difference. Carry over current acknowledgements rather than funding from the reference snapshot.

## Compose and arrange

The default is 72 by 46.25 inches with three columns, a pale gray title band and footer, teal section headings above outlined panels, and dark text on white. Preserve the style, not this study's five-figure count or section names.

Use a short introduction, experimental design, principal results, and conclusions. Arrange results in a reading order that matches the study. Under result figures, use a brief **Analysis** explanation followed by a **Finding** with decisive numbers. Explain model symbols next to a model figure when useful. Include limitations needed to interpret the claims.

Place complete vector figure PDFs once each, proportionally scaled. Do not crop panels, distort figures, or change internal figure layouts merely to fit. If a supplied figure is unsuitable, explain the constraint and adjust panel space or obtain authorization to revise it. Prefer shortening prose or rearranging panels to shrinking text below the established hierarchy.

The template's `\place` and `\posterfigure` macros reject content taller than its allotted space or extending outside the page. Set maximum heights to the actual available space; do not increase a limit past a neighboring panel merely to suppress an error. They do not detect every overlap, so visual inspection remains necessary. For another canvas size, reflow coordinates and text; changing paper dimensions alone does not resize the layout.

Keep a new poster self-contained: put its figure assets in a `figures/` subfolder and use relative paths. In this repository the established poster uses `../../fig/`; preserve that arrangement when updating it.

## Build and inspect

Use the helper from this skill directory (replace `SKILL_DIR` with its resolved location):

```bash
python3 "$SKILL_DIR/scripts/build_poster.py" /absolute/path/poster/poster.tex
```

For sources with assets outside their own folder, set an explicit project root so the archive can preserve relative paths:

```bash
python3 "$SKILL_DIR/scripts/build_poster.py" docs/SfN2026/sfn2026_poster.tex --project-root .
```

The helper requires Python 3.9+, XeLaTeX, and Poppler (`pdfinfo`, `pdftoppm`). It compiles twice without shell escape, rejects overfull boxes and multi-page output, renders a 3200-pixel preview, and bundles project-local inputs recorded by XeLaTeX. Outputs default to the source folder: `<stem>.pdf`, `<stem>_preview.png`, `<stem>_source.zip`, and `<stem>_build.log`. Auxiliary files stay in a temporary directory. Use `--output-dir` for another destination. It does not run bibliography tools; resolve bibliography generation separately if introducing one.

Inspect the full rendered preview and readable close-ups of changed or dense panels. Verify title and author spacing, heading clearance, figure labels, complete panel bounds, footers, and readable text. After any source or figure change, rebuild all three deliverables. Before delivery, check the PDF's actual numerical labels and funding text against current sources and confirm the archive contains the same figure versions used in the PDF. Remove all placeholder text and figure slots from a finished poster. Report compilation and visual checks accurately; a successful build alone is not visual verification.

For the repo's current poster, use its existing filenames. Deliver links to the PDF, preview, and source bundle, with a brief account of material changes. Do not commit unless the user asks, and follow repository commit rules.
