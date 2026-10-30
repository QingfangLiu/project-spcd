#!/usr/bin/env python3
"""Create the five-panel adaptive identity-allocation and fMRI figure.

The figure visualizes a normative model in which inferred/preconditioned (P)
and directly conditioned (C) cues share a finite identity-representation
budget. All numerical choices used for illustration are collected near the
top of this file so the figure can be revised without changing plotting logic.
Panel D uses noisy readouts of optimal allocations in simulated participants;
these readouts are illustrative proxies for representational strength, not fitted fMRI RSA values.
Panel E tests the prediction using existing participant-level OFC RSA results.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spcd_matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "spcd_font_cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy import stats


# ---------------------------------------------------------------------------
# Paths and illustrative parameters
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from figure_style import configure_matplotlib as configure_shared, style_axis as style_shared

FIGURES_DIR = REPO_ROOT / "fig"
PDF_PATH = FIGURES_DIR / "fig_model.pdf"
PNG_PATH = FIGURES_DIR / "fig_model.png"

EXAMPLE_BUDGET = 4.0
EXAMPLE_A_P = 4.0
EXAMPLE_A_C = 1.0

# Illustrative choices, specified independently of the empirical correlation.
PARTICIPANT_COUNT = 100
PARTICIPANT_SEED = 20261023
LOG_VALUE_RATIO_MEAN = float(np.log(EXAMPLE_A_P / EXAMPLE_A_C))
LOG_VALUE_RATIO_SD = 1.2
MEASUREMENT_NOISE_SD = 0.75


# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

TEXT_COLOR = "#20252B"
MUTED_TEXT_COLOR = "#5D6871"
GRID_COLOR = "#E5E7E9"
# Local model palette distinguishes allocation from behavioral outcomes and fMRI ROIs.
INFERRED_COLOR = "#B43C39"
DIRECT_COLOR = "#29343D"
TOTAL_COLOR = "#29343D"
REFERENCE_COLOR = "#7A858D"
COMPETITION_COLOR = "#29343D"
LOSS_COLORS = ("#EBC8C7", "#DDA19F", "#CA6B68", "#B43C39")


def configure_matplotlib() -> None:
    """Match manuscript figures at the shared 13.4-inch drawing width."""
    configure_shared()
    plt.rcParams.update({"legend.fontsize": 12})


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def single_cue_loss(a: float | np.ndarray, r: float | np.ndarray):
    """Decision loss for computational value ``a`` and resource ``r``."""

    return np.asarray(a) * np.exp(-np.asarray(r))


def optimal_allocation(a_p: float, a_c: float, budget: float) -> tuple[float, float]:
    """Return the constrained analytic optimum for the exponential model."""

    if a_p <= 0 or a_c <= 0:
        raise ValueError("a_p and a_c must be strictly positive")
    if budget < 0:
        raise ValueError("budget must be nonnegative")

    unconstrained_r_p = 0.5 * (budget + np.log(a_p / a_c))
    r_p = float(np.clip(unconstrained_r_p, 0.0, budget))
    return r_p, budget - r_p


def validate_example() -> None:
    """Guard the example optimum shown in panel B."""

    r_p, r_c = optimal_allocation(EXAMPLE_A_P, EXAMPLE_A_C, EXAMPLE_BUDGET)
    expected_r_p = 0.5 * (EXAMPLE_BUDGET + np.log(EXAMPLE_A_P / EXAMPLE_A_C))
    if not np.isclose(r_p, expected_r_p):
        raise AssertionError("Analytic optimum does not match the closed form")
    if not np.isclose(r_p + r_c, EXAMPLE_BUDGET):
        raise AssertionError("Optimal allocation does not exhaust the budget")

    marginal_p = single_cue_loss(EXAMPLE_A_P, r_p)
    marginal_c = single_cue_loss(EXAMPLE_A_C, r_c)
    if not np.isclose(marginal_p, marginal_c):
        raise AssertionError("Interior optimum does not equalize marginal values")


def simulate_participants(
    *,
    seed: int = PARTICIPANT_SEED,
    noise_sd: float = MEASUREMENT_NOISE_SD,
) -> dict[str, np.ndarray]:
    """Allocate a common budget, then add independent readout noise.

    Participants differ in log(a_P/a_C); a_C is held at one. The same
    constrained optimizer used in B determines their latent allocations.
    Gaussian readouts remain unclipped because they are noisy measurements,
    whereas latent resources must remain nonnegative and sum to R.
    """

    if not np.isfinite(noise_sd) or noise_sd < 0:
        raise ValueError("noise_sd must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    log_value_ratio = rng.normal(
        LOG_VALUE_RATIO_MEAN, LOG_VALUE_RATIO_SD, PARTICIPANT_COUNT,
    )
    allocation = np.array([
        optimal_allocation(float(np.exp(log_ratio)), 1.0, EXAMPLE_BUDGET)
        for log_ratio in log_value_ratio
    ])
    measured = allocation + rng.normal(0.0, noise_sd, allocation.shape)
    return {
        "log_value_ratio": log_value_ratio,
        "resource_p": allocation[:, 0],
        "resource_c": allocation[:, 1],
        "coding_p": measured[:, 0],
        "coding_c": measured[:, 1],
    }


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------


def style_axis(axis, *, grid_axis: str = "y") -> None:
    """Use the same axes, labels and faint guides as Figures 2 and 3."""
    style_shared(axis)
    axis.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.8)
    axis.tick_params(axis="both", labelsize=13)


def add_panel_header(axis, panel_label: str, title: str, *, pad: float = 18) -> None:
    """Use a fixed letter offset for both single plots and the paired E panel."""
    for text, offset, size in ((panel_label, -28, 20), (title, 0, 17)):
        # Keep the panel letter beside the first line of a wrapped heading.
        text_pad = pad + 20 * title.count("\n") if text == panel_label else pad
        axis.annotate(
            text, xy=(0, 1), xycoords="axes fraction", xytext=(offset, text_pad),
            textcoords="offset points", fontsize=size, fontweight="bold",
            color=TEXT_COLOR, ha="left", va="bottom", annotation_clip=False,
        )


# ---------------------------------------------------------------------------
# Empirical inputs: identical RSA metric, cue subsets and exclusions as Figure 3
# ---------------------------------------------------------------------------

EMPIRICAL_ROIS = (
    ("OFC_K2_cluster2_central_anterior_lateral", "cluster2_central_anterior_lateral", "Lateral OFC", "#238B8E"),
    ("OFC_K2_cluster1_medial_posterior_lateral", "cluster1_medial_posterior_lateral", "Medial OFC", "#8C6BB1"),
)
EMPIRICAL_SKIP_SUBS = ("sub-19", "sub-31", "sub-33")
EMPIRICAL_METRIC = "odor_identity_no_image_spearman_rho"
EMPIRICAL_CUES = ("conditioned_odor_only", "preconditioned_odor_only")


def load_empirical_correlations() -> list[dict]:
    """Retain complete paired coefficients and the original negative-direction test."""
    root = (REPO_ROOT / "rsa_roi" / "probe_imagewise_odor_identity_no_image_first_image"
            / "premeal_ofc_k2_spearman")
    results = []
    for roi, folder, label, color in EMPIRICAL_ROIS:
        path = root / folder / f"roi-{roi}_subject_spearman.csv"
        pairs = {}
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (row["sub"] in EMPIRICAL_SKIP_SUBS or row["roi"] != roi
                        or row["cue_subset"] not in EMPIRICAL_CUES):
                    continue
                subject = pairs.setdefault(row["sub"], {})
                cue = row["cue_subset"]
                if cue in subject:
                    raise ValueError(f"Duplicate coefficient: {roi}, {row['sub']}, {cue}")
                subject[cue] = float(row[EMPIRICAL_METRIC])
        subjects = sorted(pairs)
        if len(subjects) < 3 or any(set(pairs[sub]) != set(EMPIRICAL_CUES) for sub in subjects):
            raise ValueError(f"Incomplete conditioned/preconditioned pairs: {roi}")
        values = np.array([[pairs[sub][cue] for cue in EMPIRICAL_CUES] for sub in subjects])
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite empirical RSA coefficients: {roi}")
        x, y = values.T
        test = stats.spearmanr(x, y, alternative="less")
        results.append({"roi": roi, "label": label, "color": color,
                        "subjects": subjects, "x": x, "y": y,
                        "rho": float(test.statistic), "p": float(test.pvalue)})
    if results[0]["subjects"] != results[1]["subjects"]:
        raise ValueError("Both empirical ROIs must contain the same participants")
    return results


def draw_correlation(axis, x, y, color, *, point_size: float) -> None:
    """Draw the original descriptive OLS line and pointwise mean-response CI."""
    fit = stats.linregress(x, y)
    fit_x = np.linspace(x.min(), x.max(), 250)
    fit_y = fit.intercept + fit.slope * fit_x
    residual_sd = np.sqrt(np.sum((y - fit.intercept - fit.slope * x)**2) / (len(x) - 2))
    mean_se = residual_sd * np.sqrt(
        1 / len(x) + (fit_x - x.mean())**2 / np.sum((x - x.mean())**2)
    )
    half_width = stats.t.ppf(0.975, len(x) - 2) * mean_se
    axis.fill_between(fit_x, fit_y - half_width, fit_y + half_width,
                      color=color, alpha=0.18, linewidth=0, zorder=1)
    axis.plot(fit_x, fit_y, color=color, linewidth=2.0, zorder=2)
    axis.scatter(x, y, s=point_size, color=color, alpha=0.85,
                 edgecolor="white", linewidth=0.5, zorder=3)
    style_axis(axis)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def draw_panel_a(axis) -> None:
    """Plot unchanged loss functions, with their interpretation in the caption."""
    resource = np.linspace(0.0, 5.0, 500)
    for a_value, color in zip((0.5, 1.0, 2.0, 4.0), LOSS_COLORS):
        axis.plot(resource, single_cue_loss(a_value, resource), color=color,
                  linewidth=2.4, label=rf"$a_i={a_value:g}$")
    style_axis(axis)
    add_panel_header(axis, "A", "Benefits of identity\nrepresentation")
    axis.set_xlim(0.0, 5.0)
    axis.set_ylim(0.0, 4.25)
    axis.set_xlabel("Representational\nresource, $r_i$")
    axis.set_ylabel(r"Decision loss, $L_i$")
    axis.legend(loc="upper right", frameon=False, handlelength=1.8,
                borderaxespad=0.3, labelspacing=0.35)


def draw_panel_b(axis) -> None:
    """Show the same illustrative optimum without dense parameter annotations."""
    budget = EXAMPLE_BUDGET
    resource_p = np.linspace(0.0, budget, 600)
    total_loss = (single_cue_loss(EXAMPLE_A_P, resource_p)
                  + single_cue_loss(EXAMPLE_A_C, budget - resource_p))
    r_p_star, r_c_star = optimal_allocation(EXAMPLE_A_P, EXAMPLE_A_C, budget)
    minimum_loss = float(single_cue_loss(EXAMPLE_A_P, r_p_star)
                         + single_cue_loss(EXAMPLE_A_C, r_c_star))
    axis.plot(resource_p, total_loss, color=TOTAL_COLOR, linewidth=2.5)
    axis.axvline(r_p_star, color=REFERENCE_COLOR, linestyle=(0, (3, 3)), linewidth=1.2)
    axis.scatter([r_p_star], [minimum_loss], s=50, facecolor="white",
                 edgecolor=TOTAL_COLOR, linewidth=1.5, zorder=4)
    style_axis(axis)
    add_panel_header(axis, "B", "Optimal allocation", pad=38)
    axis.set_xlim(0.0, budget)
    axis.set_ylim(0.0, 4.25)
    axis.set_xlabel("Resource to preconditioned\nidentity, $r_P$")
    axis.set_ylabel("Total decision loss")
    axis.annotate("Optimum", xy=(r_p_star, minimum_loss), xytext=(1.9, 1.40),
                  fontsize=13, fontweight="bold", ha="center", va="center",
                  arrowprops={"arrowstyle": "-", "color": REFERENCE_COLOR, "linewidth": 1.0})


def draw_panel_c(axis) -> None:
    """Keep the full constrained allocation curve and simplify its annotations."""
    budget = EXAMPLE_BUDGET
    value_ratio = np.logspace(-2.2, 2.2, 800)
    difference = np.clip(np.log(value_ratio), -budget, budget)
    for mask, color in ((value_ratio <= 1, DIRECT_COLOR), (value_ratio >= 1, INFERRED_COLOR)):
        axis.plot(value_ratio[mask], difference[mask], color=color, linewidth=2.5)
        axis.fill_between(value_ratio[mask], 0, difference[mask], color=color, alpha=0.08)
    axis.axhline(0, color=REFERENCE_COLOR, linewidth=1.0)
    axis.axvline(1, color=REFERENCE_COLOR, linestyle=(0, (3, 3)), linewidth=1.0)
    for boundary in (-budget, budget):
        axis.axhline(boundary, color=GRID_COLOR, linestyle=(0, (2, 3)), linewidth=0.9)
    style_axis(axis)
    add_panel_header(axis, "C", "Allocation follows utility", pad=38)
    axis.set_xscale("log")
    axis.set_xlim(value_ratio.min(), value_ratio.max())
    axis.set_ylim(-4.4, 4.4)
    axis.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0, 100.0]))
    axis.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    axis.set_yticks([-4, -2, 0, 2, 4])
    axis.set_xlabel("Relative computational\nvalue, $a_P/a_C$")
    axis.set_ylabel("Allocation difference,\n$r_P^*-r_C^*$")
    axis.text(0.04, 0.40, "More to $C$", transform=axis.transAxes,
              color=DIRECT_COLOR, fontsize=13, fontweight="bold", ha="left", va="top")
    axis.text(0.96, 0.64, "More to $P$", transform=axis.transAxes,
              color=INFERRED_COLOR, fontsize=13, fontweight="bold", ha="right", va="top")
    axis.annotate("Equal utility", xy=(1, 0), xytext=(1.6, -1.25), fontsize=12,
                  color=MUTED_TEXT_COLOR, ha="left",
                  arrowprops={"arrowstyle": "-", "color": REFERENCE_COLOR, "linewidth": 0.9})


def draw_panel_d(axis) -> None:
    """Show the unchanged simulated shared-budget prediction."""
    participants = simulate_participants()
    x, y = participants["coding_c"], participants["coding_p"]
    draw_correlation(axis, x, y, COMPETITION_COLOR, point_size=25)
    add_panel_header(axis, "D", "Model prediction", pad=45)
    axis.text(0.5, 1.045, "Simulated participants", transform=axis.transAxes,
              ha="center", va="bottom", fontsize=13, fontweight="bold", color=COMPETITION_COLOR)
    axis.set_xlabel("Conditioned-cue identity\nrepresentation (a.u.)")
    axis.set_ylabel("Preconditioned-cue identity\nrepresentation (a.u.)")
    axis.margins(x=0.08)
    span_y = np.ptp(y)
    axis.set_ylim(y.min() - 0.18 * span_y, y.max() + 0.30 * span_y)
    rho = float(stats.spearmanr(x, y).statistic)
    axis.text(0.98, 0.97, rf"$\rho={rho:.3f}$", transform=axis.transAxes,
              ha="right", va="top", fontsize=13)


def draw_panel_e(axes, empirical: list[dict]) -> None:
    """Test the prediction in two OFC ROIs using their original RSA coefficients."""
    all_values = np.concatenate([np.concatenate([r["x"], r["y"]]) for r in empirical])
    padding = 0.12 * np.ptp(all_values)
    limits = (float(all_values.min() - padding), float(all_values.max() + padding))
    for index, (axis, result) in enumerate(zip(axes, empirical)):
        draw_correlation(axis, result["x"], result["y"], result["color"], point_size=36)
        axis.axhline(0, color=REFERENCE_COLOR, linewidth=0.9, linestyle="--")
        axis.axvline(0, color=REFERENCE_COLOR, linewidth=0.9, linestyle="--")
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.set_xlabel("Conditioned-cue identity\nrepresentation (RSA ρ)")
        if index == 0:
            axis.set_ylabel("Preconditioned-cue identity\nrepresentation (RSA ρ)")
        axis.text(0.5, 1.045, result["label"], transform=axis.transAxes,
                  ha="center", va="bottom", fontsize=13, fontweight="bold", color=result["color"])
        axis.text(0.98, 0.97, f"ρ = {result['rho']:.3f}\np = {result['p']:.3f}",
                  transform=axis.transAxes, ha="right", va="top", fontsize=12)


# ---------------------------------------------------------------------------
# Figure assembly and export
# ---------------------------------------------------------------------------


def create_figure():
    """Three model-mechanism plots above a prediction and its paired empirical test."""
    configure_matplotlib()
    empirical = load_empirical_correlations()
    figure = plt.figure(figsize=(13.4, 9.8), facecolor="white")
    grid = figure.add_gridspec(2, 3, left=0.075, right=0.985, top=0.90, bottom=0.15,
                              wspace=0.50, hspace=0.75)
    axes = [figure.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    draw_panel_a(axes[0])
    draw_panel_b(axes[1])
    draw_panel_c(axes[2])
    draw_panel_d(axes[3])
    draw_panel_e(axes[4:], empirical)
    header_e = figure.add_subplot(grid[1, 1:], frame_on=False)
    header_e.set_axis_off()
    add_panel_header(header_e, "E", "Test in fMRI data", pad=45)
    return figure


def save_figure(figure) -> None:
    """Write an editable vector PDF and a high-resolution PNG preview."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    figure.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(figure)
    print(f"Wrote: {PDF_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote: {PNG_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    """Validate calculations, export the figure and report simulation/data statistics."""
    validate_example()
    save_figure(create_figure())
    participants = simulate_participants()
    rho = stats.spearmanr(participants["coding_c"], participants["coding_p"]).statistic
    print(f"Panel D: n={PARTICIPANT_COUNT}, Spearman rho={rho:.6f}, "
          f"seed={PARTICIPANT_SEED}, measurement noise SD={MEASUREMENT_NOISE_SD}")
    for result in load_empirical_correlations():
        print(f"Panel E / {result['label']}: n={len(result['subjects'])}, "
              f"rho={result['rho']:.6f}, one-sided p={result['p']:.6f}")


if __name__ == "__main__":
    main()
