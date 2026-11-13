#!/usr/bin/env python3
"""Build manuscript Figure 6, showing conditioning odor-identity RSA.

Run from any directory:
    python /path/to/project-spcd/fig/make_fig6_conditioning_rsa.py

Outputs: fig/fig6_conditioning_rsa.pdf, fig/fig6_conditioning_rsa.png,
fig/fig6_conditioning_rsa_caption.txt.
The PDF is 11.4 cm wide and the PNG is rendered from it at 600 dpi. Reuse
Figure X's validated RSA data, schematic, and plotting helpers. Preserve the
original panels' dimensions on a narrower working canvas, then scale the
complete figure uniformly to its final width; never stretch either axis.
Keep labels at least 6 pt at final size. The original figure and manuscript
are not modified. Requires Figure X's dependencies plus PyMuPDF.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fig import make_figx as original

import fitz
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.transforms import Bbox


FINAL_WIDTH_CM = 11.4
WORKING_WIDTH_IN = 9.0
ORIGINAL_WIDTH_IN = 13.4
MIN_FINAL_FONT_PT = 6.0
PNG_DPI = 600


def caption(n: int) -> str:
    return (
        "Figure 6. Lateral OFC represents reward identity during conditioning.\n\n"
        "(A) Odor expectation during conditioning. Dashed branches link conditioned "
        "cues to their expected odors, shown in dashed boxes; solid boxes indicate "
        "delivered odors. Odor images are illustrative and were not displayed during "
        "the task. Odor delivery was modeled separately in the single-trial GLMs. "
        "(B) Odor-identity RSA. Spearman correlations between neural and expected "
        "odor-identity RDMs were computed from cue and prediction-period estimates "
        "(0-5 s; EXT5s), using correct food-odor trials across all available "
        "conditioning runs and restricting comparisons to different runs. "
        "Analyses used the same bilateral OFC regions as Figure 3. Points represent "
        f"individual participants (n = {n}), and open diamonds indicate means with "
        "t-based 95% confidence intervals. The dashed line indicates zero RSA "
        "correlation. Asterisks denote uncorrected p < 0.05 from one-sided "
        "one-sample t tests against zero; n.s., not significant.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output_prefix", type=Path,
                        help="Default: <project_root>/fig/fig6_conditioning_rsa")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    prefix = (args.output_prefix or project / "fig" / "fig6_conditioning_rsa").expanduser().resolve()

    rsa, wide, _, baseline, summaries = original.load_panel(project, "rsa")
    if not rsa["runs"].isin(["1,2,3", "1,2"]).all():
        raise ValueError("Unexpected available-run pattern in the conditioning inputs.")
    n = len(wide)
    offsets = original.manuscript._participant_offsets(wide.index.tolist(), width=.12)
    original.configure_matplotlib()

    # Convert only normalized horizontal positions; panel sizes stay unchanged
    # in inches relative to the original Figure X working canvas.
    horizontal = ORIGINAL_WIDTH_IN / WORKING_WIDTH_IN
    fig = plt.figure(figsize=(WORKING_WIDTH_IN, 5.0))
    schematic = fig.add_axes([.015 * horizontal, .18, .345 * horizontal, .65])
    original.draw_conditioning_schematic(schematic, project)
    axis = fig.add_axes([.418 * horizontal, .28, .239 * horizontal, .55])
    original.draw_panel(axis, wide, baseline, summaries, offsets, ylabel="Spearman ρ")
    for x, letter, title in [(.02, "A", "Odor expectation\nduring conditioning"),
                             (.388, "B", "Odor-identity RSA")]:
        fig.text(x * horizontal, .965, letter, fontsize=original.PANEL_SIZE,
                 fontweight="bold", va="top")
        fig.text((x + .03) * horizontal, .965, title,
                 fontsize=original.TITLE_SIZE, fontweight="bold",
                 va="top", linespacing=1.2)

    scale = FINAL_WIDTH_CM / (WORKING_WIDTH_IN * 2.54)
    # Drawing creates the tick labels before enforcing their final font size.
    fig.canvas.draw()
    for label in fig.findobj(match=Text):
        label.set_fontsize(max(label.get_fontsize(), MIN_FINAL_FONT_PT / scale))
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    lower = max(0.0, bounds.y0 - .08)
    export_bounds = Bbox.from_extents(0, lower, WORKING_WIDTH_IN, fig.get_figheight())

    metadata = {
        "Title": "Figure 6. Conditioning odor-identity RSA",
        "Subject": f"EXT5s all-available-run cross-run RSA; n={n}; width=11.4 cm",
        "Creator": "project-spcd/fig/make_fig6_conditioning_rsa.py",
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with BytesIO() as buffer:
        fig.savefig(buffer, format="pdf", dpi=PNG_DPI * scale,
                    bbox_inches=export_bounds, pad_inches=0, metadata=metadata)
        with fitz.open(stream=buffer.getvalue(), filetype="pdf") as source:
            width_pt = FINAL_WIDTH_CM / 2.54 * 72
            height_pt = source[0].rect.height * scale
            with fitz.open() as output:
                page = output.new_page(width=width_pt, height=height_pt)
                page.show_pdf_page(page.rect, source, 0, keep_proportion=True)
                output.set_metadata(source.metadata)
                output.save(prefix.with_suffix(".pdf"), garbage=4, deflate=True)
                page.get_pixmap(dpi=PNG_DPI, alpha=False).save(prefix.with_suffix(".png"))
    plt.close(fig)

    caption_path = prefix.parent / f"{prefix.name}_caption.txt"
    caption_path.write_text(caption(n), encoding="utf-8")
    for roi, summary in summaries.items():
        print("RSA", original.manuscript.ROI_SHORT[roi], summary)
    print(f"Final size: {FINAL_WIDTH_CM:.2f} x {height_pt / 72 * 2.54:.2f} cm; PNG: {PNG_DPI} dpi")
    print(f"Saved {prefix}.pdf, {prefix}.png, and {caption_path}")


if __name__ == "__main__":
    main()
