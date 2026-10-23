#!/usr/bin/env python3
"""Rebuild supplementary paired hunger/satiety/desire-to-eat ratings.

Run from any directory:
    python3 /path/to/project-spcd/scripts_behavior/make_hunger_satiety_figure.py

Source: analyze_hungry.ipynb, cells 2--6. Read beh_raw/sub-*/ratings/
hunger_ratings.csv with no header. Indices 0--2 encode pre-meal hunger,
satiety, and desire to eat; indices 3--5 encode the same post-meal questions.
Use main-figure exclusions (19, 31, 33); report missing files. Each included
participant must have exactly one rating for every phase/question cell.
Paired tests align by subject and follow the original two-sided asymptotic
Wilcoxon convention, without continuity or multiple-comparison correction.

Shared manuscript styling uses paired gray trajectories and open-diamond
means +/- t-based 95% CIs, as in the main figures. The original notebook
also used 95% CIs (seaborn bootstrap), despite the old supplement caption
saying SEM. Figures go to supp_fig/; source observations and
full precision statistics go to beh_processed_subs/. Originals are retained.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_font_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import (
    NOTE_SIZE, REFERENCE_COLOR, SPINE_COLOR, SUMMARY_COLOR,
    add_panel_header, configure_matplotlib, style_axis,
)
from scripts_behavior.make_behavior_figure import DEFAULT_SKIP_SUBS


TITLES = {1: "Hunger", 2: "Satiety", 3: "Desire to eat"}


def build_source_data(project_root: Path, skip_subs: tuple[str, ...]) -> pd.DataFrame:
    subjects = sorted(p.name for p in (project_root / "beh_processed_subs").glob("sub-*")
                      if p.is_dir() and p.name not in skip_subs)
    frames = []
    for subject in subjects:
        path = project_root / "beh_raw" / subject / "ratings/hunger_ratings.csv"
        if not path.is_file():
            print(f"Missing hunger/satiety ratings: {subject} (omitted from all three panels)")
            continue
        frame = pd.read_csv(path, header=None, names=["rating_index", "rating"])
        if len(frame) != 6 or set(frame["rating_index"]) != set(range(6)):
            raise ValueError(f"Expected unique rating indices 0--5 in {path}")
        if not np.isfinite(frame["rating"].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite ratings in {path}")
        if not frame["rating"].between(-4, 4).all():
            raise ValueError(f"Ratings outside the -4 to 4 scale in {path}")
        frame["question"] = (frame["rating_index"] % 3 + 1).astype(int)
        frame["measure"] = frame["question"].map(TITLES)
        frame["phase"] = np.where(frame["rating_index"] < 3, "pre", "post")
        frame["subject"] = subject
        frame["source_file"] = str(path.relative_to(project_root))
        frames.append(frame)
    if len(frames) < 3:
        raise ValueError("At least three participants with complete ratings are required")
    return pd.concat(frames, ignore_index=True)[
        ["subject", "question", "measure", "phase", "rating", "source_file"]
    ].sort_values(["subject", "question", "phase"]).reset_index(drop=True)


def summarize(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for question, title in TITLES.items():
        wide = source.loc[source["question"].eq(question)].pivot(
            index="subject", columns="phase", values="rating",
        )[["pre", "post"]]
        if not np.isfinite(wide.to_numpy()).all():
            raise ValueError(f"Incomplete paired ratings for {title}")
        n = len(wide)
        difference = wide["post"] - wide["pre"]
        if (difference == 0).all():
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(wide["pre"], wide["post"], alternative="two-sided",
                                  method="approx", correction=False, zero_method="wilcox")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        row = {"question": question, "measure": title, "n": n,
               "mean_change_post_minus_pre": float(difference.mean()),
               "wilcoxon_W": statistic, "p_two_sided_uncorrected": p_value}
        for phase in ("pre", "post"):
            mean, sem = float(wide[phase].mean()), float(stats.sem(wide[phase]))
            half_width = float(stats.t.ppf(0.975, n - 1) * sem)
            row.update({f"mean_{phase}": mean, f"sem_{phase}": sem,
                        f"ci95_lower_{phase}": mean - half_width,
                        f"ci95_upper_{phase}": mean + half_width})
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not np.isfinite(summary.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Nonfinite hunger/satiety statistics")
    return summary


def make_figure(source: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.4), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.79, wspace=0.30)
    for ax, (question, title), letter in zip(axes, TITLES.items(), "ABC"):
        style_axis(ax)
        add_panel_header(ax, letter, title)
        wide = source.loc[source["question"].eq(question)].pivot(
            index="subject", columns="phase", values="rating",
        )[["pre", "post"]]
        for values in wide.to_numpy():
            ax.plot([0, 1], values, color=SPINE_COLOR, alpha=0.40, linewidth=0.9, zorder=2)
        row = summary.loc[summary["question"].eq(question)].iloc[0]
        means = np.array([row["mean_pre"], row["mean_post"]])
        lower = np.array([row["ci95_lower_pre"], row["ci95_lower_post"]])
        upper = np.array([row["ci95_upper_pre"], row["ci95_upper_post"]])
        ax.errorbar([0, 1], means, yerr=[means - lower, upper - means], fmt="D-",
                    color=SUMMARY_COLOR, markerfacecolor="white", markeredgewidth=1.6,
                    markersize=7, linewidth=2, capsize=4, zorder=3)
        ax.axhline(0, color=REFERENCE_COLOR, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
        ax.set(xticks=[0, 1], xticklabels=["Pre-meal", "Post-meal"],
               xlim=(-0.25, 1.25), ylim=(-4.6, 4.8), yticks=[-4, -2, 0, 2, 4])
        p = row["p_two_sided_uncorrected"]
        label = "$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"
        ax.text(0.5, 0.98, label, transform=ax.transAxes, ha="center", va="top", fontsize=NOTE_SIZE)
    fig.supylabel("Rating", x=0.012, fontsize=16)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Figure directory (default: <project>/supp_fig). CSVs go to beh_processed_subs.")
    parser.add_argument("--skip-subs", nargs="*", default=list(DEFAULT_SKIP_SUBS))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    root = args.project_root.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve() if args.output_dir else root / "supp_fig"
    source = build_source_data(root, tuple(args.skip_subs))
    summary = summarize(source)
    print(f"Excluded: {', '.join(args.skip_subs)}; included n = {source.subject.nunique()}")
    print(summary.to_string(index=False))
    source.to_csv(root / "beh_processed_subs/hunger_satiety_source_data.csv", index=False)
    summary.to_csv(root / "beh_processed_subs/hunger_satiety_statistics.csv", index=False)
    figure = make_figure(source, summary)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = destination / f"hungry_ratings.{extension}"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        print(f"Saved: {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
