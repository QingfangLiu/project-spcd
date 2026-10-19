#!/usr/bin/env python3
"""Additive subject-level regression of A2>A1 choice change on value and RSA.

Uses the change scores from corr_probe_beh_odor_rating.ipynb and the premeal
preconditioned-cue OFC identity score used by the existing neural analysis.
The identity coefficient tests an adjusted association with choice change.
The model contains two main effects and an intercept, with HC3 standard
errors, t-based 95% confidence intervals, and two-sided tests.
Outputs default to a temporary preview folder.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DEFAULT_RSA = Path(
    "rsa_roi/probe_imagewise_odor_identity_no_image_first_image/"
    "premeal_ofc_k2_spearman/cluster2_central_anterior_lateral/"
    "roi-OFC_K2_cluster2_central_anterior_lateral_subject_spearman.csv"
)
EXCLUDED_SUBJECTS = {"sub-19", "sub-33"}
FORMULA = "choice_change ~ relative_pleasantness_change + identity_z"


def build_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce notebook changes; validate cached choices against raw trials."""
    behavior_root = root / "beh_processed_subs"
    choices = pd.read_csv(behavior_root / "probe_choice_behavior_all.csv")
    choices = choices.loc[choices["choice_class"].eq("A2>A1")].set_index("subject")
    if choices.index.has_duplicates:
        raise ValueError("Duplicate A2>A1 rows in cached choice data")
    if not np.allclose(choices["change"], choices["post"] - choices["pre"]):
        raise ValueError("Cached choice change disagrees with post minus pre")

    rsa = pd.read_csv(root / DEFAULT_RSA)
    rsa = rsa.loc[
        rsa["cue_subset"].eq("preconditioned_odor_only") & rsa["session"].eq("premeal")
    ].set_index("sub")["odor_identity_no_image_spearman_rho"]
    if rsa.index.has_duplicates:
        raise ValueError("Duplicate subject-level premeal RSA rows")

    rows, audit = [], []
    subjects = sorted(set(choices.index) | set(rsa.index))
    for subject in subjects:
        rating_path = behavior_root / subject / "pleasantness_summary.csv"
        reasons = []
        if subject in EXCLUDED_SUBJECTS:
            reasons.append("excluded by original correlation notebook")
        else:
            if subject not in choices.index:
                reasons.append("missing A2>A1 choice data")
            if not rating_path.is_file():
                reasons.append("missing pleasantness ratings")
            if subject not in rsa.index or not np.isfinite(rsa.loc[subject]):
                reasons.append("missing or nonfinite identity RSA")
        audit.append({"subject": subject, "included": not reasons, "reason": "; ".join(reasons)})
        if reasons:
            continue

        ratings = pd.read_csv(rating_path).set_index("odor_num")
        if ratings.index.has_duplicates or not {1, 2}.issubset(ratings.index):
            raise ValueError(f"Invalid odor rows in {rating_path}")
        change = ratings["post_mean"] - ratings["pre_mean"]
        if not np.allclose(change, ratings["change"], atol=1e-10, rtol=0):
            raise ValueError(f"Stored rating changes disagree with post minus pre: {subject}")
        choice = choices.loc[subject]
        trial_counts = {}
        for phase in ("pre", "post"):
            trials = pd.read_csv(behavior_root / subject / f"{phase}_meal_dat.csv")
            trials = trials.dropna(subset=["choice_class", "acc", "rt"])
            trials = trials.loc[trials["choice_class"].isin(["A2>A1", "A1>A2"])].copy()
            reverse = trials["choice_class"].eq("A1>A2")
            trials.loc[reverse, "acc"] = 1 - trials.loc[reverse, "acc"]
            if not len(trials) or not np.isclose(trials["acc"].mean(), choice[phase]):
                raise ValueError(f"Cached choices disagree with raw trials: {subject}/{phase}")
            trial_counts[f"n_{phase}_trials"] = len(trials)
        rows.append({
            "subject": subject,
            "choice_pre": float(choice["pre"]),
            "choice_post": float(choice["post"]),
            "choice_change": float(choice["change"]),
            "sated_pleasantness_change": float(change.loc[2]),
            "nonsated_pleasantness_change": float(change.loc[1]),
            "relative_pleasantness_change": float(change.loc[2] - change.loc[1]),
            "identity_rsa": float(rsa.loc[subject]),
            **trial_counts,
        })
    data = pd.DataFrame(rows)
    if len(data) < 5 or not np.isfinite(data.drop(columns="subject").to_numpy()).all():
        raise ValueError("Too few complete subjects or nonfinite input values")
    identity_sd = data["identity_rsa"].std(ddof=1)
    if identity_sd <= 0:
        raise ValueError("Identity RSA has no between-subject variation")
    data["identity_z"] = (data["identity_rsa"] - data["identity_rsa"].mean()) / identity_sd
    return data, pd.DataFrame(audit)


def make_plot(data: pd.DataFrame, model, output_dir: Path) -> None:
    """Show observed changes and model-adjusted changes at mean value change."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
    value_mean = data["relative_pleasantness_change"].mean()
    data["choice_change_adjusted_for_pleasantness"] = (
        data["choice_change"]
        - model.params["relative_pleasantness_change"]
        * (data["relative_pleasantness_change"] - value_mean)
    )
    panels = (
        ("relative_pleasantness_change", "choice_change",
         "Relative pleasantness change\nΔ(sated − non-sated rating)",
         "Observed choice and pleasantness changes", "#687887"),
        ("identity_rsa", "choice_change_adjusted_for_pleasantness",
         "Premeal preconditioned identity RSA (Spearman ρ)",
         "Identity association adjusted for pleasantness", "#2878B5"),
    )
    for i, (xcol, ycol, xlabel, title, color) in enumerate(panels):
        ax = axes[i]
        ax.axhline(0, color="#C3C8CD", lw=0.8, zorder=0)
        ax.scatter(data[xcol], data[ycol], s=55, c=color, alpha=0.8,
                   edgecolors="white", linewidths=0.6, zorder=3)
        x = np.linspace(data[xcol].min(), data[xcol].max(), 200)
        predict = pd.DataFrame({"relative_pleasantness_change": np.full(len(x), value_mean),
                                "identity_z": np.zeros(len(x))})
        if i == 0:
            predict["relative_pleasantness_change"] = x
        else:
            predict["identity_z"] = (x - data["identity_rsa"].mean()) / data["identity_rsa"].std(ddof=1)
        interval = model.get_prediction(predict).summary_frame(alpha=0.05)
        ax.plot(x, interval["mean"], color=color, lw=2)
        ax.fill_between(x, interval["mean_ci_lower"], interval["mean_ci_upper"],
                        color=color, alpha=0.14, linewidth=0)
        ax.set(xlabel=xlabel, title=title)
        ax.set_ylabel("A2>A1 choice change (post − pre)" if i == 0 else
                      "Choice change adjusted for pleasantness\n(at mean pleasantness change)")
        ax.grid(axis="y", color="#ECEFF1", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.07)
    ci = model.conf_int().loc["identity_z"]
    axes[1].text(0.97, 0.05,
                 f"Identity β per SD = {model.params['identity_z']:.3f}\n"
                 f"95% CI [{ci.iloc[0]:.3f}, {ci.iloc[1]:.3f}]\n"
                 f"Two-sided p = {model.pvalues['identity_z']:.3f}",
                 transform=axes[1].transAxes, va="bottom", ha="right", fontsize=10,
                 bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"})
    fig.suptitle(f"Additive model of preconditioned-cue devaluation · n = {len(data)}\n"
                 "Central/anterior-lateral OFC · identity and value main effects",
                 fontsize=14)
    fig.supxlabel("One point per subject. Lines: additive model at the other regressor's mean; bands: 95% HC3 confidence intervals.\n"
                  "More negative choice changes indicate greater avoidance of the sated cue.", fontsize=9)
    fig.savefig(output_dir / "additive_identity_devaluation.png", dpi=180)
    fig.savefig(output_dir / "additive_identity_devaluation.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "spcd_identity_additive_preview")
    args = parser.parse_args()
    data, audit = build_data(args.project_root.resolve())
    model = smf.ols(FORMULA, data=data).fit(cov_type="HC3", use_t=True)
    if np.linalg.matrix_rank(model.model.exog) != 3:
        raise ValueError("Additive model design is rank deficient")
    # Independent numerical check of coefficients and the adjusted plot slope.
    expected = np.linalg.lstsq(model.model.exog, data["choice_change"], rcond=None)[0]
    np.testing.assert_allclose(model.params, expected, atol=1e-12)
    residual_x = smf.ols("identity_z ~ relative_pleasantness_change", data=data).fit().resid
    residual_y = smf.ols("choice_change ~ relative_pleasantness_change", data=data).fit().resid
    np.testing.assert_allclose(np.dot(residual_x, residual_y) / np.dot(residual_x, residual_x),
                               model.params["identity_z"], atol=1e-12)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    make_plot(data, model, output_dir)
    coefficients = pd.DataFrame({"coefficient": model.params, "se_hc3": model.bse,
                                 "ci95_low": model.conf_int()[0], "ci95_high": model.conf_int()[1],
                                 "t": model.tvalues, "p_two_sided": model.pvalues})
    coefficients.index.name = "term"
    data["fitted_choice_change"] = model.fittedvalues
    data["residual"] = model.resid
    data["cooks_distance"] = model.get_influence().cooks_distance[0]
    data.to_csv(output_dir / "subject_analysis_data.csv", index=False)
    audit.to_csv(output_dir / "subject_inclusion.csv", index=False)
    coefficients.to_csv(output_dir / "coefficients.csv")
    (output_dir / "model_summary.txt").write_text(model.summary().as_text() + "\n")
    summary = {
        "formula": FORMULA, "choice_comparison": "A2>A1", "n_subjects": len(data),
        "rsa_source": str(DEFAULT_RSA), "rsa_subset": "preconditioned_odor_only",
        "identity_mean": float(data["identity_rsa"].mean()),
        "identity_sd": float(data["identity_rsa"].std(ddof=1)),
        "relative_pleasantness_definition": "(post_odor2 - pre_odor2) - (post_odor1 - pre_odor1)",
        "choice_change_definition": "post minus pre probability of choosing A2 over A1",
        "identity_prediction": "negative coefficient: greater avoidance of the sated cue at equal value change",
        "inference": "OLS; HC3 covariance; t-based 95% confidence intervals; two-sided p-values",
        "r_squared": float(model.rsquared), "adjusted_r_squared": float(model.rsquared_adj),
        "coefficients": coefficients.reset_index().to_dict(orient="records"),
        "excluded": audit.loc[~audit["included"]].to_dict(orient="records"),
        "notes": ["One equally weighted observation per subject; two regressors plus intercept.",
                  "Adjusted association; the model does not test moderation or establish causality.",
                  "Cached choice scores verified against raw trial means for every retained subject."],
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"n = {len(data)}; R² = {model.rsquared:.3f}")
    print(coefficients.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
