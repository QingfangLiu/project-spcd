#!/usr/bin/env python3
"""Rebuild supplementary recognition memory accuracy from raw trial CSVs.

Run from any directory:
    python3 /path/to/project-spcd/scripts_behavior/make_recognition_memory_figure.py

Source: analyze_recognition_memory.ipynb, cells 2--8. Normalize quoted
left/right key names; omit missing responses/RTs; average accuracy within
each of six pairs, then average the two pairs in each outcome group with
equal weight. Match the main figure exclusions (19, 31, 33). All plotted
participants must have both pairs in all three groups. The two-sided
Friedman test compares the three paired participant-level accuracies.

The figure uses the shared manuscript style and mean +/- t-based 95% CIs,
replacing the original notebook's SEM bars. Jitter is fixed, independent of
accuracy, and shared across groups. Original notebook outputs are preserved.
PDF/PNG files go to supp_fig/; source pair-level observations
and statistics go to beh_processed_subs/. The joint Friedman result is
repeated alongside each group summary in the statistics CSV.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import (
    MUTED_TEXT_COLOR, NOTE_SIZE, REFERENCE_COLOR,
    SUMMARY_COLOR, configure_matplotlib, style_axis,
)
from scripts_behavior.make_behavior_figure import DEFAULT_SKIP_SUBS, PAIR_GROUPS


GROUPS = ("A1-B1", "A2-B2", "C-D")


def normalize_response(values: pd.Series) -> pd.Series:
    return (values.astype("string").str.strip().str.strip("'").str.strip('"').str.lower()
            .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "na": pd.NA}))


def build_source_data(project_root: Path, skip_subs: tuple[str, ...]) -> pd.DataFrame:
    """Load and summarize each retained participant's six cue pairs."""
    subjects = sorted(p.name for p in (project_root / "beh_processed_subs").glob("sub-*")
                      if p.is_dir() and p.name not in skip_subs)
    if len(subjects) < 3:
        raise ValueError("At least three retained participants are required")
    frames = []
    for subject in subjects:
        paths = sorted((project_root / "beh_raw" / subject).glob("*RecognitionTrials.csv"))
        if len(paths) != 1:
            raise ValueError(f"Expected one recognition trial file for {subject}, found {len(paths)}")
        trials = pd.read_csv(paths[0])
        correct = normalize_response(trials["correct"])
        response = normalize_response(trials["key_resp.keys_raw"])
        for values in (correct, response):
            if not values.dropna().isin(["left", "right"]).all():
                raise ValueError(f"Unexpected response keys in {paths[0]}")
        trials["accuracy"] = (correct == response).astype(float)
        trials.loc[correct.isna() | response.isna(), "accuracy"] = np.nan
        trials = trials.dropna(subset=["accuracy", "key_resp.rt_mean"])
        pair = pd.to_numeric(trials["pair"], errors="raise")
        if not pair.isin(PAIR_GROUPS).all():
            raise ValueError(f"Unexpected cue-pair codes in {paths[0]}")
        trials["pair"] = pair.astype(int)
        summary = trials.groupby("pair", as_index=False).agg(
            mean_accuracy=("accuracy", "mean"), n_valid_trials=("accuracy", "count"),
        )
        if set(summary["pair"]) != set(PAIR_GROUPS):
            raise ValueError(f"Incomplete cue pairs for {subject}")
        summary["pair_group"] = summary["pair"].map(PAIR_GROUPS)
        summary["subject"] = subject
        summary["source_file"] = str(paths[0].relative_to(project_root))
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)[
        ["subject", "pair", "pair_group", "mean_accuracy", "n_valid_trials", "source_file"]
    ].sort_values(["subject", "pair"]).reset_index(drop=True)


def summarize(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Each pair receives equal weight even when valid trial counts differ.
    participants = source.groupby(["subject", "pair_group"], as_index=False)["mean_accuracy"].mean()
    wide = participants.pivot(index="subject", columns="pair_group", values="mean_accuracy")[list(GROUPS)]
    if not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Recognition accuracy is incomplete or nonfinite")
    test = stats.friedmanchisquare(*(wide[group].to_numpy() for group in GROUPS))
    summary = participants.groupby("pair_group")["mean_accuracy"].agg(
        n="count", mean="mean", sd="std", sem="sem",
    ).loc[list(GROUPS)]
    half_width = stats.t.ppf(0.975, summary["n"] - 1) * summary["sem"]
    summary["ci95_lower"] = summary["mean"] - half_width
    summary["ci95_upper"] = summary["mean"] + half_width
    summary["friedman_chi2"] = float(test.statistic)
    summary["friedman_df"] = 2
    summary["friedman_p"] = float(test.pvalue)
    if not np.isfinite(summary.to_numpy()).all():
        raise ValueError("Nonfinite recognition summaries or Friedman test")
    return participants, summary.reset_index()


def make_figure(participants: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    fig.subplots_adjust(left=0.16, right=0.975, bottom=0.28, top=0.83)
    style_axis(ax)
    subjects = sorted(participants["subject"].unique())
    offsets = dict(zip(subjects, np.random.default_rng(20261023).permutation(
        np.linspace(-0.13, 0.13, len(subjects)),
    )))
    positions = participants["pair_group"].map({group: i for i, group in enumerate(GROUPS)})
    ax.scatter(positions + participants["subject"].map(offsets), participants["mean_accuracy"],
               color=MUTED_TEXT_COLOR, alpha=0.4, s=34, linewidths=0, zorder=2)
    ax.errorbar(np.arange(3), summary["mean"],
                yerr=[summary["mean"] - summary["ci95_lower"], summary["ci95_upper"] - summary["mean"]],
                fmt="D", color=SUMMARY_COLOR, markersize=7, markerfacecolor="white",
                markeredgewidth=1.6, linewidth=2, capsize=4, zorder=3)
    ax.axhline(0.5, color=REFERENCE_COLOR, linestyle=(0, (3, 3)), linewidth=1.2, zorder=1)
    ax.set(xticks=np.arange(3), xticklabels=["Non-sated\n(A1-B1)", "Sated\n(A2-B2)", "Clean air\n(C-D)"],
           ylabel="Accuracy", xlim=(-0.45, 2.45), ylim=(0, max(1.08, summary["ci95_upper"].max() + 0.04)),
           yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.set_title("Recognition memory", loc="left", pad=16)
    fig.text(0.975, 0.965, f"n = {len(subjects)} participants", ha="right", va="top",
             fontsize=NOTE_SIZE, color=MUTED_TEXT_COLOR)
    fig.legend(handles=[
        Line2D([], [], linestyle="none", marker="o", color=MUTED_TEXT_COLOR,
               alpha=0.5, label="Participants"),
        Line2D([], [], linestyle="none", marker="D", color=SUMMARY_COLOR,
               markerfacecolor="white", markeredgewidth=1.6, label="Mean (95% CI)"),
    ], loc="lower center", bbox_to_anchor=(0.56, 0.105), ncol=2, frameon=False,
       columnspacing=1, handletextpad=0.5)
    row = summary.iloc[0]
    fig.text(0.56, 0.045, f"Friedman $\\chi^2$(2) = {row['friedman_chi2']:.2f}, $p$ = {row['friedman_p']:.3f}",
             ha="center", fontsize=12, color=MUTED_TEXT_COLOR)
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
    participants, summary = summarize(source)
    print(f"Excluded: {', '.join(args.skip_subs)}; included n = {participants.subject.nunique()}")
    print(summary.to_string(index=False))
    source.to_csv(root / "beh_processed_subs/recognition_memory_source_data.csv", index=False)
    summary.to_csv(root / "beh_processed_subs/recognition_memory_statistics.csv", index=False)
    figure = make_figure(participants, summary)
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        path = destination / f"recognition_accuracy.{extension}"
        figure.savefig(path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        print(f"Saved: {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
