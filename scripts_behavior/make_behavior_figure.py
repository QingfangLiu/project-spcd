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
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_font_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")
PAIR_GROUPS = {1: "A1-B1", 4: "A1-B1", 2: "A2-B2", 5: "A2-B2", 3: "C-D", 6: "C-D"}
CONDITIONED_CHOICES = ("B2>B1", "B2>D", "B1>D")
PRECONDITIONED_CHOICES = ("A2>A1", "A2>C", "A1>C")
POINT_COLOR = "#1f77b4"


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


def _bar_inset(ax: plt.Axes, data: pd.DataFrame, order: tuple, title: str) -> None:
    summary = _summary(data, order)
    x = np.arange(len(order))
    ax.bar(x, summary["mean"], yerr=summary["sem"], width=0.6, color="gray",
           capsize=2.5, error_kw={"elinewidth": 1})
    ax.set(xticks=x, xticklabels=order, ylim=(0, 1.07), yticks=[0, 0.5, 1], ylabel="Accuracy")
    ax.set_title(title, fontsize=11, pad=5)
    ax.tick_params(labelsize=9)
    ax.yaxis.label.set_size(10)


def _plot_changes(
    ax: plt.Axes, data: pd.DataFrame, order: tuple, labels: tuple,
    title: str, ylabel: str, ylim: tuple[float, float], *, legend_loc: str = "upper left",
) -> None:
    summary = _summary(data, order)
    positions = data["category"].map({label: i for i, label in enumerate(order)}).to_numpy(dtype=float)
    jitter = np.random.default_rng(0).normal(0, 0.08, size=len(data))
    ax.scatter(positions + jitter, data["value"], color=POINT_COLOR, alpha=0.3,
               s=27, linewidths=0, label="individual", zorder=2)
    ax.errorbar(np.arange(len(order)), summary["mean"], yerr=summary["sem"],
                fmt="o", color="black", markersize=4.5, capsize=3.5,
                linewidth=1.5, label="mean ± SEM", zorder=3)
    ax.axhline(0, color=POINT_COLOR, linestyle=":", linewidth=1, alpha=0.8, zorder=1)
    ax.set(xticks=np.arange(len(order)), xticklabels=labels, ylabel=ylabel,
           ylim=ylim, xlim=(-0.45, len(order) - 0.55))
    ax.set_title(title, pad=9)
    ax.legend(loc=legend_loc)


def make_figure(data: dict[str, pd.DataFrame]) -> plt.Figure:
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "axes.linewidth": 0.9, "legend.fontsize": 9,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.grid": False, "legend.framealpha": 0.95,
    })
    fig = plt.figure(figsize=(13.4, 10.6), facecolor="white")
    outer = fig.add_gridspec(2, 1, left=0.075, right=0.985, bottom=0.075,
                            top=0.95, hspace=0.31, height_ratios=(1, 1.02))
    top = outer[0].subgridspec(1, 2, wspace=0.27)
    bottom = outer[1].subgridspec(1, 3, width_ratios=(1.18, 1, 1), wspace=0.43)
    axes = [fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])]
    axes.extend(fig.add_subplot(bottom[0, i]) for i in range(3))
    a, b, c, d, e = axes

    for _, subject_data in data["A"].groupby("subject"):
        a.plot(subject_data["category"], subject_data["value"], color="gray", alpha=0.3, linewidth=0.8)
    summary = _summary(data["A"], (1, 2, 3, 4))
    a.errorbar([1, 2, 3, 4], summary["mean"], yerr=summary["std"], fmt="o",
               color="black", markersize=4.5, capsize=3.5, linewidth=1.5, label="mean ± SD")
    a.axvline(3.5, color="gray", linestyle="--", alpha=0.6)
    a.set(xticks=[1, 2, 3, 4], xticklabels=["Day 1\nrun 1", "Day 1\nrun 2", "Day 1\nrun 3", "Day 2\nlast run"],
          ylim=(0, 1.05), xlim=(0.85, 4.15), ylabel="Accuracy")
    a.set_title("Preconditioning across runs", pad=9)
    a.legend(loc="lower left")
    _bar_inset(a.inset_axes([0.48, 0.16, 0.34, 0.36]), data["A inset"],
               ("A1-B1", "A2-B2", "C-D"), "Day 2 last run")

    summary = _summary(data["B"], (1, 2, 3))
    jitter = np.random.default_rng(0).normal(0, 0.04, size=len(data["B"]))
    b.scatter(data["B"]["category"] + jitter, data["B"]["value"], color=POINT_COLOR,
               alpha=0.4, s=22, linewidths=0)
    b.errorbar([1, 2, 3], summary["mean"], yerr=summary["sem"], fmt="o-",
               color="black", markersize=4.5, capsize=3.5, linewidth=2)
    b.set(xticks=[1, 2, 3], xticklabels=["Run 1", "Run 2", "Run 3"],
          ylim=(0, 1.05), xlim=(0.78, 3.22), ylabel="Accuracy")
    b.set_title("Conditioning across runs", pad=9)
    _bar_inset(b.inset_axes([0.60, 0.13, 0.36, 0.38]), data["B inset"], ("B1", "B2", "D"), "Run 3")

    _plot_changes(c, data["C"], (2, 1, 0), ("Sated odor", "Non-sated odor", "Clean air"),
                  "Selective decrease in pleasantness", "Change in pleasantness rating (post − pre)",
                  (-4.05, 2.15), legend_loc="lower right")
    for ax, letter, order, title in (
        (d, "D", CONDITIONED_CHOICES, "Conditioned cue choices"),
        (e, "E", PRECONDITIONED_CHOICES, "Preconditioned cue choices"),
    ):
        _plot_changes(ax, data[letter], order, order, title,
                      "Change in choice proportion (post − pre)", (-1.1, 1.1))
    for ax, letter in zip(axes, "ABCDE"):
        box = ax.get_position()
        fig.text(box.x0 - 0.045, box.y1 + 0.018, letter, fontsize=16, fontweight="bold",
                 ha="left", va="bottom")
        ax.tick_params(direction="out", length=3.5, width=0.8)
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
        fig.savefig(output_path, dpi=args.dpi, facecolor="white")
        print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
