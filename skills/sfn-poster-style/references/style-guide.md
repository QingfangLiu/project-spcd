# Visual system and layout

Use these defaults from the approved reference, not fixed scientific content requirements.

## Canvas and colors

The reference is a single landscape page, **72 x 46.25 inches** (182.88 x 117.475 cm). Coordinates are in inches from the top-left. Columns begin around x=1, 25, and 48.5, with widths of roughly 22, 22.5, and 22.5 inches. Narrow gutters connect the three columns visually.

| Role | Hex color |
| --- | --- |
| Body text | `#243641` |
| Secondary metadata | `#58676F` |
| Headings and bullets | `#247C80` |
| Title and footer bands | `#F1F4F4` |
| Panel outlines | `#78999B` |

Use white panel backgrounds and 6 pt outlines. Section headings sit above their boxes. Align top borders in the first row and bottom borders in the last row across columns where practical. Avoid shadows, gradients, decorative illustrations, and separate colored title bars for every panel.

## Typography

The reference uses Arial and Latin Modern Math. The blank template uses Arial when available, otherwise TeX Gyre Heros; do not download proprietary fonts. Recheck line wrapping after font substitution.

| Element | Size / line spacing (pt) |
| --- | --- |
| Main title | 99 / 116, bold |
| Authors | 40 / 49 |
| Affiliations | 27 / 34, muted |
| Main section headings | 48 / 56, bold teal |
| Short side headings | 44 / 52, bold teal |
| Body text | 30 / 38 |
| Introduction and model explanation | 32 / 42 |

Center the title near y=0.65. Metadata sits near y=2.30: contact left, presentation number right, authors/affiliations centrally. Keep long titles and affiliations clear of the side blocks; measure the render rather than assuming a line count. Start content headings around y=4.70. The footer band begins at y=43.00, with headings near y=43.20 and panels at y=44.05.

Use concise paragraphs and teal bullets. Lead conclusions with short bold claims. Explain model symbols and reasoning instead of supplying equations alone. Avoid shrinking an entire dense poster to fit a fixed amount of prose.

## Reference arrangement

The original study uses five main figures. The blank template retains its seven content panels to demonstrate the style; rename, remove, or resize panels for another study.

| Panel | Outline bounds: left, top, right, bottom (in) | Content strategy |
| --- | --- | --- |
| Introduction | 0.60, 5.65, 24.40, 13.61 | Motivation bullets |
| Experimental design | 0.60, 15.06, 24.40, 33.86 | Full design figure, compact methods |
| Conclusions | 0.60, 35.31, 24.40, 42.70 | Claims and interpretive limits |
| Primary results | 24.65, 5.65, 47.90, 26.08 | Tall figure, Analysis / Finding |
| Second results | 24.65, 27.43, 47.90, 42.70 | Full figure and supporting prose |
| Third results | 48.15, 5.65, 71.40, 17.24 | Short wide figure and explanation |
| Model / final results | 48.15, 18.73, 71.40, 42.70 | Figure, equations, simulation and empirical findings |
| References | 0.60, 44.05, 27.60, 46.10 | Selected citations in two columns |
| Acknowledgements | 28.60, 44.05, 71.40, 46.10 | Current funding and institutional text |

The template shows empty frames when an expected figure PDF is missing, sized to its planning slot. Real figures retain their native aspect ratio and can occupy less height; adjust nearby text after inserting them. Never deliver a placeholder as a result.

## Source freshness and portability

The reference and source archive are snapshots of `docs/SfN2026/sfn2026_poster.*` after the November 9, 2026 update. The historical poster has simulated rho=-0.324 and empirical rho=-0.317; these identify the snapshot, not values to reuse for another study.

Current project sources are `writing/main/main.tex`, `writing/supp/supp.tex`, and main figures under `fig/`. Resolve these relative to the repository root, not the skill folder. New studies use their own sources. Check newer changes before copying scientific content.

For another aspect ratio or size, establish the new canvas first and re-plan panel bounds, title band, and footer. Keep white space and hierarchy; select type sizes for the final viewing distance. Do not stretch the page or retain inch coordinates outside the new canvas.
