#!/usr/bin/env python3
"""Relate preconditioned-cue identity RSA to value-guided probe choices.

The primary analysis asks whether a subject-level premeal identity-RSA score
moderates the effect of the meal-induced, within-subject change in outcome
value on trial-level preconditioned-cue choices.  The script deliberately
defaults to a temporary preview directory and refuses to write into the
repository's ``figures`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_xdg_cache")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr


DEFAULT_RSA = Path(
    "rsa_roi/probe_imagewise_odor_identity_no_image_first_image/"
    "premeal_ofc_k2_spearman/cluster2_central_anterior_lateral/"
    "roi-OFC_K2_cluster2_central_anterior_lateral_subject_spearman.csv"
)
EXCLUDED_SUBJECTS = {"sub-19", "sub-33"}

# sub-14's behavioral cue indices follow the pre-correction mapping described
# in the premeal GLM documentation.  This mapping is needed only to validate
# response reconstruction; value assignment uses the physical odor columns.
STANDARD_A_CUES = {1: "A1", 3: "A2", 5: "A3", 7: "A1", 9: "A2", 11: "A3"}
SUB14_A_CUES = {1: "A2", 3: "A1", 5: "A3", 7: "A2", 9: "A1", 11: "A3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "spcd_neural_behavior_preview",
        help="Preview output directory; must not be inside <project>/figures.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=20_000,
        help="Subject bootstrap and permutation iterations.",
    )
    parser.add_argument("--seed", type=int, default=20261018)
    return parser.parse_args()


def _assert_safe_output(project_root: Path, output_dir: Path) -> None:
    figures_dir = (project_root / "figures").resolve()
    resolved = output_dir.resolve()
    if resolved == figures_dir or figures_dir in resolved.parents:
        raise ValueError(
            f"Preview output may not be written inside the figures directory: {resolved}"
        )


def _load_rsa(project_root: Path) -> pd.Series:
    path = project_root / DEFAULT_RSA
    table = pd.read_csv(path)
    selected = table.loc[
        table["cue_subset"].eq("preconditioned_odor_only"),
        ["sub", "odor_identity_no_image_spearman_rho"],
    ]
    if selected["sub"].duplicated().any():
        raise ValueError(f"Duplicate preconditioned RSA rows in {path}")
    return selected.set_index("sub")["odor_identity_no_image_spearman_rho"]


def _validate_choice_reconstruction(
    data: pd.DataFrame, subject: str, source: Path
) -> None:
    mapping = SUB14_A_CUES if subject == "sub-14" else STANDARD_A_CUES
    first_cue = data["cue_left_idx"].map(mapping)
    second_cue = data["cue_right_idx"].map(mapping)
    if first_cue.isna().any() or second_cue.isna().any():
        raise ValueError(f"Unrecognized preconditioned cue index in {source}")

    chosen_cue = np.where(data["resp"].eq(0), first_cue, second_cue)
    class_target = data["choice_class"].str.split(">").str[0]
    reconstructed_accuracy = (chosen_cue == class_target).astype(int)
    mismatch = reconstructed_accuracy.ne(data["acc"].astype(int))
    if mismatch.any():
        examples = data.loc[
            mismatch,
            ["Run", "trial", "choice_class", "cue_left_idx", "cue_right_idx", "resp", "acc"],
        ].head()
        raise ValueError(
            f"Choice reconstruction failed for {subject} in {source}:\n"
            f"{examples.to_string(index=False)}"
        )


def build_trial_table(project_root: Path, rsa: pd.Series) -> pd.DataFrame:
    behavior_root = project_root / "beh_processed_subs"
    ratings = pd.read_csv(behavior_root / "compiled_pleasantness_ratings.csv").set_index(
        "sub"
    )
    records: list[pd.DataFrame] = []

    for subject_dir in sorted(behavior_root.glob("sub-*")):
        subject = subject_dir.name
        if (
            subject in EXCLUDED_SUBJECTS
            or subject not in ratings.index
            or subject not in rsa.index
        ):
            continue

        for phase, filename, rating_prefix in (
            ("pre", "pre_meal_dat.csv", "pre"),
            ("post", "post_meal_dat.csv", "post"),
        ):
            source = subject_dir / filename
            data = pd.read_csv(source)
            data = data.loc[data["choice_class"].str.startswith("A", na=False)].copy()
            data = data.dropna(
                subset=[
                    "resp",
                    "rt",
                    "acc",
                    "odor_left",
                    "odor_right",
                    "cue_left_idx",
                    "cue_right_idx",
                ]
            )
            unexpected_responses = sorted(set(data["resp"]) - {0.0, 2.0})
            if unexpected_responses:
                raise ValueError(
                    f"Unexpected response codes in {source}: {unexpected_responses}"
                )
            _validate_choice_reconstruction(data, subject, source)

            value_lookup = {
                0: ratings.loc[subject, f"{rating_prefix}_clean"],
                1: ratings.loc[subject, f"{rating_prefix}_odor1"],
                2: ratings.loc[subject, f"{rating_prefix}_odor2"],
            }
            data["subject"] = subject
            data["phase"] = phase
            data["phase_post"] = int(phase == "post")
            data["odor_a"] = data[["odor_left", "odor_right"]].min(axis=1).astype(int)
            data["odor_b"] = data[["odor_left", "odor_right"]].max(axis=1).astype(int)
            data["odor_pair"] = (
                data["odor_a"].astype(str) + "-" + data["odor_b"].astype(str)
            )
            data["chosen_odor"] = np.where(
                data["resp"].eq(0), data["odor_left"], data["odor_right"]
            ).astype(int)
            data["choose_odor_a"] = data["chosen_odor"].eq(data["odor_a"]).astype(int)
            data["value_a"] = data["odor_a"].map(value_lookup)
            data["value_b"] = data["odor_b"].map(value_lookup)
            data["current_value_difference"] = data["value_a"] - data["value_b"]
            data["identity_rsa"] = float(rsa.loc[subject])

            records.append(
                data[
                    [
                        "subject",
                        "phase",
                        "phase_post",
                        "Run",
                        "trial",
                        "choice_class",
                        "odor_a",
                        "odor_b",
                        "odor_pair",
                        "chosen_odor",
                        "choose_odor_a",
                        "value_a",
                        "value_b",
                        "current_value_difference",
                        "identity_rsa",
                    ]
                ]
            )

    if not records:
        raise ValueError("No complete behavioral, rating, and neural records found")

    trials = pd.concat(records, ignore_index=True)
    phase_values = trials[
        ["subject", "phase", "odor_pair", "current_value_difference"]
    ].drop_duplicates()
    duplicate_cells = phase_values.duplicated(["subject", "phase", "odor_pair"])
    if duplicate_cells.any():
        raise ValueError("A subject/phase/odor-pair cell has inconsistent value differences")

    pair_means = phase_values.groupby(["subject", "odor_pair"])[
        "current_value_difference"
    ].mean()
    trials = trials.join(
        pair_means.rename("between_phase_value_difference"),
        on=["subject", "odor_pair"],
    )
    trials["within_subject_value_change"] = (
        trials["current_value_difference"]
        - trials["between_phase_value_difference"]
    )
    trials["between_phase_value_difference_c"] = (
        trials["between_phase_value_difference"]
        - trials["between_phase_value_difference"].mean()
    )

    subject_neural = trials[["subject", "identity_rsa"]].drop_duplicates()
    neural_mean = subject_neural["identity_rsa"].mean()
    neural_sd = subject_neural["identity_rsa"].std(ddof=1)
    trials["identity_rsa_z"] = (trials["identity_rsa"] - neural_mean) / neural_sd
    return trials


def fit_models(trials: pd.DataFrame):
    primary_formula = (
        "choose_odor_a ~ within_subject_value_change * identity_rsa_z "
        "+ between_phase_value_difference_c + phase_post + C(odor_pair)"
    )
    current_formula = (
        "choose_odor_a ~ current_value_difference * identity_rsa_z "
        "+ phase_post + C(odor_pair)"
    )
    primary = smf.gee(
        primary_formula,
        "subject",
        trials,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    current = smf.gee(
        current_formula,
        "subject",
        trials,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable(),
    ).fit()
    return primary, current


def _linear_probability_slope(value: pd.Series, choice: pd.Series) -> float:
    x = value.to_numpy(dtype=float)
    y = choice.to_numpy(dtype=float)
    centered = x - x.mean()
    denominator = np.sum(centered**2)
    if denominator == 0:
        return np.nan
    return float(np.sum(centered * (y - y.mean())) / denominator)


def build_subject_scores(trials: pd.DataFrame) -> pd.DataFrame:
    cells = (
        trials.groupby(["subject", "phase", "odor_pair"], as_index=False)
        .agg(
            choice_probability=("choose_odor_a", "mean"),
            value_difference=("current_value_difference", "first"),
            n_trials=("choose_odor_a", "size"),
            identity_rsa=("identity_rsa", "first"),
        )
    )

    rows: list[dict[str, float | str]] = []
    for subject, subject_trials in trials.groupby("subject"):
        subject_cells = cells.loc[cells["subject"].eq(subject)]
        choice_wide = subject_cells.pivot(
            index="odor_pair", columns="phase", values="choice_probability"
        )
        value_wide = subject_cells.pivot(
            index="odor_pair", columns="phase", values="value_difference"
        )
        common_pairs = choice_wide.dropna().index.intersection(value_wide.dropna().index)
        choice_change = (
            choice_wide.loc[common_pairs, "post"]
            - choice_wide.loc[common_pairs, "pre"]
        )
        value_change = (
            value_wide.loc[common_pairs, "post"] - value_wide.loc[common_pairs, "pre"]
        )

        rows.append(
            {
                "subject": subject,
                "identity_rsa": float(subject_trials["identity_rsa"].iloc[0]),
                "current_value_sensitivity": _linear_probability_slope(
                    subject_trials["current_value_difference"],
                    subject_trials["choose_odor_a"],
                ),
                "premeal_value_sensitivity": _linear_probability_slope(
                    subject_trials.loc[
                        subject_trials["phase"].eq("pre"),
                        "current_value_difference",
                    ],
                    subject_trials.loc[
                        subject_trials["phase"].eq("pre"), "choose_odor_a"
                    ],
                ),
                "postmeal_value_sensitivity": _linear_probability_slope(
                    subject_trials.loc[
                        subject_trials["phase"].eq("post"),
                        "current_value_difference",
                    ],
                    subject_trials.loc[
                        subject_trials["phase"].eq("post"), "choose_odor_a"
                    ],
                ),
                # Positive values mean that the pre-to-post change in choice
                # probability followed the direction of the value change.
                "value_congruent_choice_update": float(
                    np.mean(choice_change.to_numpy() * value_change.to_numpy())
                ),
                "n_trials": int(len(subject_trials)),
            }
        )
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def correlation_summary(
    scores: pd.DataFrame, metric: str, iterations: int, seed: int
) -> dict[str, float | int | str | list[float]]:
    complete = scores[["identity_rsa", metric]].dropna()
    x = complete["identity_rsa"].to_numpy()
    y = complete[metric].to_numpy()
    observed_spearman = float(spearmanr(x, y).statistic)
    pearson = pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = len(complete)

    boot = np.empty(iterations)
    for index in range(iterations):
        sample = rng.integers(0, n, n)
        boot[index] = spearmanr(x[sample], y[sample]).statistic
    boot = boot[np.isfinite(boot)]

    permuted = np.empty(iterations)
    for index in range(iterations):
        permuted[index] = spearmanr(x, rng.permutation(y)).statistic
    permutation_p_two_sided = float(
        (1 + np.sum(np.abs(permuted) >= abs(observed_spearman)))
        / (iterations + 1)
    )
    permutation_p_positive = float(
        (1 + np.sum(permuted >= observed_spearman)) / (iterations + 1)
    )

    return {
        "metric": metric,
        "n": n,
        "spearman_rho": observed_spearman,
        "spearman_p_asymptotic_two_sided": float(spearmanr(x, y).pvalue),
        "spearman_bootstrap_95_ci": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "spearman_permutation_p_two_sided": permutation_p_two_sided,
        "spearman_permutation_p_one_sided_positive": permutation_p_positive,
        "pearson_r": float(pearson.statistic),
        "pearson_p_two_sided": float(pearson.pvalue),
    }


def _model_term_summary(model, term: str) -> dict[str, float | str]:
    estimate = float(model.params[term])
    standard_error = float(model.bse[term])
    return {
        "term": term,
        "estimate_log_odds": estimate,
        "standard_error": standard_error,
        "z": float(estimate / standard_error),
        "p_two_sided": float(model.pvalues[term]),
        "odds_ratio": float(np.exp(estimate)),
        "ci_95_log_odds": [
            float(estimate - 1.96 * standard_error),
            float(estimate + 1.96 * standard_error),
        ],
    }


def plot_subject_scores(
    scores: pd.DataFrame, summaries: dict[str, dict], output_dir: Path
) -> None:
    sns.set_theme(style="ticks", context="talk")
    color = "#D95F4B"
    panels = [
        (
            "current_value_sensitivity",
            "Behavioral value sensitivity\n(probability per pleasantness unit)",
            "A",
        ),
        (
            "value_congruent_choice_update",
            "Value-congruent choice updating\n(mean Δvalue × Δchoice)",
            "B",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.7))
    for ax, (metric, ylabel, letter) in zip(axes, panels):
        sns.regplot(
            data=scores,
            x="identity_rsa",
            y=metric,
            ax=ax,
            ci=95,
            n_boot=5_000,
            seed=20261018,
            color=color,
            scatter_kws={"s": 55, "alpha": 0.82, "edgecolor": "white"},
            line_kws={"linewidth": 2.2},
        )
        summary = summaries[metric]
        ci_low, ci_high = summary["spearman_bootstrap_95_ci"]
        ax.text(
            0.04,
            0.96,
            (
                f"Spearman ρ = {summary['spearman_rho']:.2f}\n"
                f"95% CI [{ci_low:.2f}, {ci_high:.2f}]\n"
                f"permutation p = {summary['spearman_permutation_p_two_sided']:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
        )
        ax.set_xlabel("Preconditioned odor-identity RSA (Spearman ρ)")
        ax.set_ylabel(ylabel)
        ax.text(-0.14, 1.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=20)
        sns.despine(ax=ax)

    fig.suptitle(
        "Does stronger lateral-OFC identity coding predict greater value sensitivity?",
        fontsize=18,
        y=1.02,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"neural_behavior_subject_level_preview.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _moderation_effect_curve(model, trials: pd.DataFrame) -> pd.DataFrame:
    """Return the model-implied value effect across observed identity RSA."""
    subjects = trials[["subject", "identity_rsa", "identity_rsa_z"]].drop_duplicates()
    neural_z_grid = np.linspace(
        subjects["identity_rsa_z"].min(), subjects["identity_rsa_z"].max(), 200
    )
    neural_mean = subjects["identity_rsa"].mean()
    neural_sd = subjects["identity_rsa"].std(ddof=1)
    value_term = "within_subject_value_change"
    interaction_term = "within_subject_value_change:identity_rsa_z"
    covariance = model.cov_params()
    rows = []
    for neural_z in neural_z_grid:
        slope = float(
            model.params[value_term] + neural_z * model.params[interaction_term]
        )
        variance = float(
            covariance.loc[value_term, value_term]
            + neural_z**2 * covariance.loc[interaction_term, interaction_term]
            + 2
            * neural_z
            * covariance.loc[value_term, interaction_term]
        )
        standard_error = np.sqrt(max(variance, 0.0))
        rows.append(
            {
                "identity_rsa": neural_mean + neural_z * neural_sd,
                "identity_rsa_z": neural_z,
                "value_effect_log_odds": slope,
                "value_effect_odds_ratio": float(np.exp(slope)),
                "ci_low": float(np.exp(slope - 1.96 * standard_error)),
                "ci_high": float(np.exp(slope + 1.96 * standard_error)),
            }
        )
    return pd.DataFrame(rows)


def plot_moderation(model, trials: pd.DataFrame, output_dir: Path, seed: int) -> None:
    del seed  # Kept in the function signature for a stable command-line interface.
    prediction = _moderation_effect_curve(model, trials)
    sns.set_theme(style="ticks", context="talk")
    fig, ax = plt.subplots(figsize=(7.2, 5.9))
    color = "#D95F4B"
    ax.plot(
        prediction["identity_rsa"],
        prediction["value_effect_odds_ratio"],
        color=color,
        linewidth=2.6,
    )
    ax.fill_between(
        prediction["identity_rsa"],
        prediction["ci_low"],
        prediction["ci_high"],
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    subject_rsa = trials[["subject", "identity_rsa"]].drop_duplicates()["identity_rsa"]
    ax.plot(
        subject_rsa,
        np.full(len(subject_rsa), 0.985),
        "|",
        color="#555555",
        markersize=7,
        alpha=0.55,
        transform=ax.get_xaxis_transform(),
    )
    term = "within_subject_value_change:identity_rsa_z"
    ax.text(
        0.04,
        0.96,
        (
            "Value change × identity RSA\n"
            f"β = {model.params[term]:.2f}, SE = {model.bse[term]:.2f}\n"
            f"p = {model.pvalues[term]:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
    )
    ax.axhline(1, color="#999999", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Preconditioned odor-identity RSA (Spearman ρ)")
    ax.set_ylabel("Effect of value change on choice\n(odds ratio per pleasantness unit)")
    ax.set_title("Does identity coding alter behavioral value sensitivity?", fontsize=18)
    sns.despine(ax=ax)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            output_dir / f"neural_behavior_moderation_preview.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    prediction.to_csv(output_dir / "moderation_effect_curve.csv", index=False)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    _assert_safe_output(project_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rsa = _load_rsa(project_root)
    trials = build_trial_table(project_root, rsa)
    primary_model, current_model = fit_models(trials)
    scores = build_subject_scores(trials)

    metrics = ("current_value_sensitivity", "value_congruent_choice_update")
    correlations = {
        metric: correlation_summary(
            scores,
            metric,
            iterations=args.bootstrap_iterations,
            seed=args.seed + offset,
        )
        for offset, metric in enumerate(metrics)
    }
    primary_term = "within_subject_value_change:identity_rsa_z"
    current_term = "current_value_difference:identity_rsa_z"
    summary = {
        "analysis": (
            "Preconditioned-cue choice sensitivity to outcome value as a function "
            "of central/anterior-lateral OFC preconditioned odor-identity RSA"
        ),
        "n_subjects": int(trials["subject"].nunique()),
        "n_trials": int(len(trials)),
        "excluded_subjects": sorted(EXCLUDED_SUBJECTS),
        "subjects_missing_ratings_or_neural_data": sorted(
            set(rsa.index) - set(trials["subject"].unique()) - EXCLUDED_SUBJECTS
        ),
        "primary_gee_interaction": _model_term_summary(primary_model, primary_term),
        "current_value_gee_interaction": _model_term_summary(current_model, current_term),
        "subject_level_correlations": correlations,
        "notes": [
            "Positive within-person value differences indicate that odor A became more valuable relative to odor B.",
            "Positive value-congruent choice updating indicates that choice changed in the same direction as value.",
            "The subject-level slopes are descriptive; the clustered trial-level GEE interaction is the inferential test.",
        ],
    }

    trials.to_csv(output_dir / "preconditioned_trial_level_analysis.csv", index=False)
    scores.to_csv(output_dir / "subject_value_sensitivity_scores.csv", index=False)
    with (output_dir / "analysis_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    (output_dir / "primary_gee_summary.txt").write_text(primary_model.summary().as_text())
    (output_dir / "current_value_gee_summary.txt").write_text(current_model.summary().as_text())

    plot_subject_scores(scores, correlations, output_dir)
    plot_moderation(primary_model, trials, output_dir, args.seed)

    print(json.dumps(summary, indent=2))
    print(f"\nPreview outputs: {output_dir}")


if __name__ == "__main__":
    main()
