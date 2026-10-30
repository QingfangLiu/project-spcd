#!/usr/bin/env python3
"""Reproduce the directional paired change comparisons in the manuscript.

Run from any directory:
    python3 /path/to/project-spcd/scripts_behavior/analyze_devaluation_change_tests.py

Reuse the Figure 2 participant exclusions, trial filters, cue orientation, and
post-minus-pre changes. One-sided Wilcoxon signed-rank tests assess whether the devalued
condition has a more negative change than the non-devalued condition.
Contrasts are rounded to 12 decimal places to preserve numerical ties, as in
the Figure 2 change tests. Zero differences are excluded from ranking. W is
the positive-rank sum, and all p-values are uncorrected.

Write participant-level contrasts and full-precision statistics to
beh_processed_subs/ without changing existing data or figure outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from make_behavior_figure import DEFAULT_SKIP_SUBS, build_panel_data


CONTRASTS = (
    ("odor_pleasantness", "C", 2, 1, "devalued", "non-devalued"),
    ("conditioned_choices", "D", "B2>D", "B1>D", "B2>D", "B1>D"),
    ("preconditioned_choices", "E", "A2>C", "A1>C", "A2>C", "A1>C"),
)


def analyze(panels: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations, results = [], []
    for name, panel, first, second, first_label, second_label in CONTRASTS:
        wide = panels[panel].pivot(index="subject", columns="category", values="value")
        contrast = wide[[first, second]].copy()
        contrast.columns = ["first_change", "second_change"]
        contrast["difference_in_changes"] = contrast["first_change"] - contrast["second_change"]
        values = contrast["difference_in_changes"].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all() or np.std(values, ddof=1) == 0:
            raise ValueError(f"Invalid participant contrasts for {name}")
        ranked_values = np.round(values, decimals=12)
        test = stats.wilcoxon(
            ranked_values, alternative="less", zero_method="wilcox", method="auto",
        )
        results.append({
            "contrast": name, "first_condition": first_label, "second_condition": second_label,
            "n": len(values), "mean_first_change": contrast["first_change"].mean(),
            "mean_second_change": contrast["second_change"].mean(),
            "mean_difference_in_changes": values.mean(),
            "sd_difference_in_changes": values.std(ddof=1),
            "test": "Wilcoxon signed-rank", "W": float(test.statistic),
            "n_nonzero": int(np.count_nonzero(ranked_values)),
            "zero_method": "wilcox", "method": "auto", "rank_round_decimals": 12,
            "alternative": "first_change_less_than_second_change",
            "p_one_sided_uncorrected": float(test.pvalue),
        })
        observations.append(contrast.reset_index().assign(
            contrast=name, first_condition=first_label, second_condition=second_label,
        ))
    return pd.concat(observations, ignore_index=True), pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Default: <project>/beh_processed_subs")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else root / "beh_processed_subs"
    observations, results = analyze(build_panel_data(root, DEFAULT_SKIP_SUBS))
    output.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output / "behavioral_devaluation_change_test_source_data.csv", index=False)
    results.to_csv(output / "behavioral_devaluation_change_test_statistics.csv", index=False)
    print(results[["contrast", "n", "W", "p_one_sided_uncorrected"]].to_string(index=False))
    print(f"Saved participant contrasts and statistics to {output}")


if __name__ == "__main__":
    main()
