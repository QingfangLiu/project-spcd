# Python Figure 1 assets

The separate Figure 1 generator is `scripts_behavior/make_design_figure.py`.
Run `python3 scripts_behavior/make_design_figure.py` from the project root.
It exports `fig/fig_design_python.pdf`, `.png`, and `.svg` and imports the
shared `figure_style.py`. The main manuscript uses this Python-generated PDF;
the original Adobe `.ai` and `.pdf` remain preserved as alternatives.

Six higher-resolution stimulus images are loaded directly from `cues/`:
Charmander (A1), Stitch (A2), Tigger (C), waterfall_small (B1), museum (B2),
and city_road_night (D). These match the examples in the original figure.

`sweet_odor.png` and `savory_odor.png` preserve the strawberry and potato-chip
illustrations embedded in the original `fig/fig_design.pdf`. They were extracted
losslessly with their alpha masks (PDF image/mask objects 27/32 and 26/31).
They illustrate example odor identities, rather than a fixed assignment across
participants. Colors follow the behavior figure: blue for A1/B1, orange for
A2/B2, and gray for C/D.

## Conditioning timing

The **4 s** conditioning delay in the figure and caption is the effective
interval from the end of prediction to odor arrival at the nose. It comprises
a **2 s programmed delay**, followed by approximately **2 s transit through
the delivery tubing** after odor release. The Methods explain this distinction;
the root README retains the programmed timeline and notes the effective delay.
The tubing transit time was clarified by the experimenter. Trial displays are
schematic; the 4 s label does not describe the duration of the software delay
screen alone.

The prediction display centers the cue above the three outcome abbreviations
as described in the manuscript. The preconditioning timeline has four displays:
first cue, gap, associate choice, and text-only accuracy feedback. The feedback
screen illustrates the alternative messages "Correct" or "Incorrect", without a
cue image. Feedback lasts 2 s, verified from `text_fb.started` and
`text_fb.stopped` timestamps in the raw preconditioning CSVs for subjects 04, 35,
and 36 (training and refresher). Those 2 s are included within the 3-8 s post-choice
interval, followed by fixation for the remainder. ITI information for all three
trial types is provided in the main manuscript's Figure 1 caption, rather than
the trial-demo labels. Each trial-demo cue image uses
the same size: 42.5% of screen width, including paired choices and prediction.
Display widths
are schematic, rather than proportional to time. Original `.ai` and `.pdf` files
are preserved, and the generator rejects their output stem.
