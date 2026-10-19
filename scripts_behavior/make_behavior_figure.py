#!/usr/bin/env python3
"""Generate the five-panel behavior figure directly from processed subject CSVs.

Run from any directory:
    python3 /path/to/project-spcd/scripts_behavior/make_behavior_figure.py

The calculations follow analyze_precond.ipynb (A), analyze_conditioning.ipynb
(B), analyze_odor_ratings.ipynb (C), and analyze_probe.ipynb (D/E). Subjects
19, 31, and 33 are excluded before aggregation. Missing pleasantness files
are reported separately; they do not remove subjects from other panels.

A displays mean +/- SD; its inset and all other panels display mean +/- SEM.
Day 2 preconditioning uses the notebook's final-run correction (run 7 takes
precedence over run 6; both are recoded to run 1). The preconditioning inset
averages the two pairs within each pair group with equal weight. Conditioning
and probe analyses retain the notebooks' missing-response/RT filtering rules.
Probe changes are paired post-minus-pre proportions, with A2/B2 choices
oriented against A1/B1 and A3/B3 relabeled C/D. The generator does not read
the Illustrator file, existing panel PDFs, or cached group-level summaries.

Outcome colors are local to this figure: blue = non-sated (A1/B1), orange =
sated (A2/B2), gray = clean air (C/D). They do not encode learning route.
Panels D/E use neutral points and descriptive comparison labels, retaining
the analysis cue codes underneath. Negative changes indicate reduced choice
of the cue named before ">", independently of stimulus presentation order.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_font_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_style import (
    INSET_LABEL_SIZE, INSET_TICK_SIZE, MUTED_TEXT_COLOR, REFERENCE_COLOR,
    SPINE_COLOR, SUMMARY_COLOR, add_panel_header, configure_matplotlib, style_axis,
)


DEFAULT_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")
PAIR_GROUPS = {1: "A1-B1", 4: "A1-B1", 2: "A2-B2", 5: "A2-B2", 3: "C-D", 6: "C-D"}
CONDITIONED_CHOICES = ("B2>B1", "B2>D", "B1>D")
PRECONDITIONED_CHOICES = ("A2>A1", "A2>C", "A1>C")
# Odor-number codes match pleasantness_summary.csv and the cue-odor mappings.
OUTCOME_COLORS = {0: "#9AA3AA", 1: "#2878B5", 2: "#D97721"}
COMPARISON_LABELS = ("Sated\n> non-sated", "Sated\n> air", "Non-sated\n> air")


def _load_subjects(
    root: Path, subjects: list[str], filename: str, *, allow_missing: bool = False
) -> pd.DataFrame:
    frames = []
    for subject in subjects:
        path = root / subject / filename
        if not path.is_file():
            if allow_missing:
                print(f"Missing {filename}: {subject} (omitted from panel C only)")
                continue
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path).assign(subject=subject))
    if not frames:
        raise ValueError(f"No data found for {filename}")
    return pd.concat(frames, ignore_index=True)


def _panel_table(
    frame: pd.DataFrame, category: str, value: str,
    categories: tuple, subjects: list[str],
) -> pd.DataFrame:
    """Require one finite observation for every subject/category cell."""
    table = frame[["subject", category, value]].rename(
        columns={category: "category", value: "value"}
    ).copy()
    if table.duplicated(["subject", "category"]).any():
        raise ValueError(f"Duplicate subject/category observations for {category}")
    expected = {(subject, cell) for subject in subjects for cell in categories}
    observed = set(zip(table["subject"], table["category"]))
    if expected != observed:
        raise ValueError(
            f"Unexpected cells for {category}: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    if not np.isfinite(table["value"].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite values for {category}")
    return table.sort_values(["subject", "category"]).reset_index(drop=True)


def build_panel_data(project_root: Path, skip_subs: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    root = project_root / "beh_processed_subs"
    subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir() and p.name not in skip_subs)
    if len(subjects) < 2:
        raise ValueError(f"Fewer than two retained subjects found in {root}")

    # A: retain the original Day 2 final-run selection and pair-group weighting.
    day1 = _load_subjects(root, subjects, "preconditioning_day1_dat.csv")
    day2 = _load_subjects(root, subjects, "preconditioning_day2_dat.csv")
    has_run7 = day2.loc[day2["run"].eq(7), "subject"].unique()
    day2 = day2.loc[~(day2["subject"].isin(has_run7) & day2["run"].eq(6))].copy()
    day2.loc[day2["run"].isin([6, 7]), "run"] = 1
    final_run = day2.loc[day2["run"].eq(1)]
    day1_acc = day1.groupby(["subject", "run"], as_index=False)["acc"].mean()
    day2_acc = final_run.groupby("subject", as_index=False)["acc"].mean().assign(run=4)
    across_runs = pd.concat([day1_acc, day2_acc], ignore_index=True)
    pair_acc = final_run.groupby(["subject", "pair"], as_index=False)["acc"].mean()
    pair_acc["pair_group"] = pair_acc["pair"].map(PAIR_GROUPS)
    if pair_acc["pair_group"].isna().any():
        raise ValueError("Unrecognized Day 2 preconditioning pair")
    pair_acc = pair_acc.groupby(["subject", "pair_group"], as_index=False)["acc"].mean()

    # B: the original inset additionally drops trials with missing RT.
    conditioning = _load_subjects(root, subjects, "odor_pred_dat.csv")
    for column in ("Run", "acc", "rt", "odor_num"):
        conditioning[column] = pd.to_numeric(conditioning[column], errors="coerce")
    conditioning = conditioning.dropna(subset=["Run", "acc", "odor_num"])
    conditioning_runs = conditioning.groupby(["subject", "Run"], as_index=False)["acc"].mean()
    conditioning_end = conditioning.dropna(subset=["rt"])
    conditioning_end = conditioning_end.loc[conditioning_end["Run"].eq(3)]
    conditioning_end = conditioning_end.groupby(["subject", "odor_num"], as_index=False)["acc"].mean()
    conditioning_end["cue"] = conditioning_end["odor_num"].map({1: "B1", 2: "B2", 0: "D"})

    # C: use subject-level pre/post means and verify the stored change column.
    ratings = _load_subjects(root, subjects, "pleasantness_summary.csv", allow_missing=True)
    changes = ratings["post_mean"] - ratings["pre_mean"]
    if not np.allclose(changes, ratings["change"], rtol=0, atol=1e-10):
        raise ValueError("Pleasantness change disagrees with post_mean - pre_mean")
    ratings["change"] = changes
    rating_subjects = sorted(ratings["subject"].unique())

    # D/E: aggregate valid trials within subject/phase before pairing phases.
    phases = []
    for phase in ("pre", "post"):
        trials = _load_subjects(root, subjects, f"{phase}_meal_dat.csv")
        trials = trials.dropna(subset=["choice_class", "acc", "rt"])
        means = trials.groupby(["subject", "choice_class"], as_index=False)["acc"].mean()
        means["choice_class"] = (
            means["choice_class"].str.replace("A3", "C", regex=False)
            .str.replace("B3", "D", regex=False)
        )
        reverse = means["choice_class"].isin(["A1>A2", "B1>B2"])
        means.loc[reverse, "acc"] = 1 - means.loc[reverse, "acc"]
        means["choice_class"] = means["choice_class"].replace({"A1>A2": "A2>A1", "B1>B2": "B2>B1"})
        phases.append(means.assign(phase=phase))
    probe = pd.concat(phases).pivot(index=["subject", "choice_class"], columns="phase", values="acc")
    probe["change"] = probe["post"] - probe["pre"]
    probe = probe.reset_index()

    data = {
        "A": _panel_table(across_runs, "run", "acc", (1, 2, 3, 4), subjects),
        "A inset": _panel_table(pair_acc, "pair_group", "acc", ("A1-B1", "A2-B2", "C-D"), subjects),
        "B": _panel_table(conditioning_runs, "Run", "acc", (1, 2, 3), subjects),
        "B inset": _panel_table(conditioning_end, "cue", "acc", ("B1", "B2", "D"), subjects),
        "C": _panel_table(ratings, "odor_num", "change", (2, 1, 0), rating_subjects),
    }
    for letter, choices in (("D", CONDITIONED_CHOICES), ("E", PRECONDITIONED_CHOICES)):
        selected = probe.loc[probe["choice_class"].isin(choices)]
        data[letter] = _panel_table(selected, "choice_class", "change", choices, subjects)
    return data


def _summary(data: pd.DataFrame, order: tuple) -> pd.DataFrame:
    return data.groupby("category")["value"].agg(["count", "mean", "std", "sem"]).loc[list(order)]


def _bar_inset(
    ax: plt.Axes, data: pd.DataFrame, order: tuple, title: str,
    *, outcome_order: tuple[int, ...],
) -> None:
    summary = _summary(data, order)
    x = np.arange(len(order))
    ax.bar(x, summary["mean"], yerr=summary["sem"], width=0.6,
           color=[OUTCOME_COLORS[odor] for odor in outcome_order],
           capsize=2.5, error_kw={"elinewidth": 1})
    ax.set(xticks=x, xticklabels=order, ylim=(0, 1.07), yticks=[0, 0.5, 1], ylabel="Accuracy")
    ax.set_title(title, loc="left", fontsize=INSET_LABEL_SIZE, pad=8)
    style_axis(ax, inset=True)


def _plot_changes(
    ax: plt.Axes, data: pd.DataFrame, order: tuple, labels: tuple,
    title: str, ylabel: str, ylim: tuple[float, float], *, legend_loc: str = "upper left",
    colors: dict | None = None,
) -> None:
    summary = _summary(data, order)
    positions = data["category"].map({label: i for i, label in enumerate(order)}).to_numpy(dtype=float)
    jitter = np.random.default_rng(0).normal(0, 0.08, size=len(data))
    point_colors = data["category"].map(colors) if colors is not None else MUTED_TEXT_COLOR
    ax.scatter(positions + jitter, data["value"], color=point_colors, alpha=0.4,
               s=34, linewidths=0, label="individual", zorder=2)
    ax.errorbar(np.arange(len(order)), summary["mean"], yerr=summary["sem"],
                fmt="o", color=SUMMARY_COLOR, markersize=6, capsize=4,
                linewidth=2, label="mean ± SEM", zorder=3)
    ax.axhline(0, color=REFERENCE_COLOR, linestyle=(0, (3, 3)), linewidth=1.2, zorder=1)
    ax.set(xticks=np.arange(len(order)), xticklabels=labels, ylabel=ylabel,
           ylim=ylim, xlim=(-0.45, len(order) - 0.55))
    ax.set_title(title, loc="left", pad=16)
    ax.legend(loc=legend_loc)


def make_figure(data: dict[str, pd.DataFrame]) -> plt.Figure:
    configure_matplotlib()
    fig = plt.figure(figsize=(13.4, 11.8), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=0.095, right=0.985, bottom=0.14,
                            top=0.91, hspace=0.48, height_ratios=(1, 1.08))
    top = outer[0].subgridspec(1, 2, wspace=0.27)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(1.18, 1, 1), wspace=0.53)
    axes = [fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])]
    axes.extend(fig.add_subplot(bottom[0, i]) for i in range(3))
    a, b, c, d, e = axes

    for _, subject_data in data["A"].groupby("subject"):
        a.plot(subject_data["category"], subject_data["value"], color=SPINE_COLOR, alpha=0.4, linewidth=0.9)
    summary = _summary(data["A"], (1, 2, 3, 4))
    a.errorbar([1, 2, 3, 4], summary["mean"], yerr=summary["std"], fmt="o",
               color=SUMMARY_COLOR, markersize=6, capsize=4, linewidth=2, label="mean ± SD")
    a.axvline(3.5, color=REFERENCE_COLOR, linestyle=(0, (3, 3)), linewidth=1.2)
    a.set(xticks=[1, 2, 3, 4], xticklabels=["Day 1\nrun 1", "Day 1\nrun 2", "Day 1\nrun 3", "Day 2\nlast run"],
          ylim=(0, 1.05), xlim=(0.85, 4.15), ylabel="Accuracy")
    a.set_title("Preconditioning across runs", loc="left", pad=16)
    a.legend(loc="lower left")
    _bar_inset(a.inset_axes([0.47, 0.17, 0.45, 0.40]), data["A inset"],
               ("A1-B1", "A2-B2", "C-D"), "Day 2 last run", outcome_order=(1, 2, 0))

    summary = _summary(data["B"], (1, 2, 3))
    jitter = np.random.default_rng(0).normal(0, 0.04, size=len(data["B"]))
    b.scatter(data["B"]["category"] + jitter, data["B"]["value"], color=MUTED_TEXT_COLOR,
               alpha=0.4, s=32, linewidths=0)
    b.errorbar([1, 2, 3], summary["mean"], yerr=summary["sem"], fmt="o-",
               color=SUMMARY_COLOR, markersize=6, capsize=4, linewidth=2)
    b.set(xticks=[1, 2, 3], xticklabels=["Run 1", "Run 2", "Run 3"],
          ylim=(0, 1.05), xlim=(0.78, 3.22), ylabel="Accuracy")
    b.set_title("Conditioning across runs", loc="left", pad=16)
    _bar_inset(b.inset_axes([0.60, 0.17, 0.37, 0.40]), data["B inset"],
               ("B1", "B2", "D"), "Run 3", outcome_order=(1, 2, 0))

    _plot_changes(c, data["C"], (2, 1, 0), ("Sated\nodor", "Non-sated\nodor", "Clean\nair"),
                  "Selective decrease\nin pleasantness", "Change in pleasantness rating\n(post − pre)",
                  (-4.05, 2.15), legend_loc="lower right", colors=OUTCOME_COLORS)
    for ax, letter, order, title in (
        (d, "D", CONDITIONED_CHOICES, "Conditioned\ncue choices"),
        (e, "E", PRECONDITIONED_CHOICES, "Preconditioned\ncue choices"),
    ):
        _plot_changes(ax, data[letter], order, COMPARISON_LABELS, title,
                      "Change in choice proportion\n(post − pre)", (-1.1, 1.1))
        # Keep the original codes as a secondary label below the two-line
        # meaning, with identical contrast order across learning routes.
        for x, choice in enumerate(order):
            ax.annotate(
                f"({choice.replace('>', ' > ')})",
                xy=(x, 0), xycoords=ax.get_xaxis_transform(),
                xytext=(0, -49), textcoords="offset points",
                ha="center", va="top", fontsize=INSET_TICK_SIZE,
                color=MUTED_TEXT_COLOR, annotation_clip=False,
            )
    for ax, letter in zip(axes, "ABCDE"):
        style_axis(ax)
        add_panel_header(ax, letter, ax.get_title(loc="left"))
    fig.legend(
        handles=[
            Patch(facecolor=OUTCOME_COLORS[1], label="Non-sated odor: A1/B1"),
            Patch(facecolor=OUTCOME_COLORS[2], label="Sated odor: A2/B2"),
            Patch(facecolor=OUTCOME_COLORS[0], label="Clean air: C/D"),
        ],
        loc="upper center", bbox_to_anchor=(0.54, 0.998), ncol=3,
        frameon=False, fontsize=12, columnspacing=2,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-prefix", type=Path, help="Output path without extension (default: <project>/figures/fig_behavior).")
    parser.add_argument("--skip-subs", nargs="*", default=list(DEFAULT_SKIP_SUBS))
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png"), default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    project_root = args.project_root.expanduser().resolve()
    output_prefix = args.output_prefix.expanduser().resolve() if args.output_prefix else project_root / "figures" / "fig_behavior"
    data = build_panel_data(project_root, tuple(args.skip_subs))
    print(f"Excluded subjects: {', '.join(args.skip_subs) or '(none)'}")
    for panel, table in data.items():
        print(f"\nPanel {panel} (subject-level observations):")
        print(_summary(table, tuple(table["category"].unique())).to_string(float_format=lambda x: f"{x:.6f}"))
    fig = make_figure(data)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in dict.fromkeys(args.formats):
        output_path = output_prefix.with_suffix(f".{extension}")
        fig.savefig(output_path, dpi=args.dpi, facecolor="white",
                    bbox_inches="tight", pad_inches=0.06)
        print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
