#!/usr/bin/env python3
"""Create the four-panel adaptive identity-allocation figure.

The figure visualizes a normative model in which inferred/preconditioned (P)
and directly conditioned (C) cues share a finite identity-representation
budget. All numerical choices used for illustration are collected near the
top of this file so the figure can be revised without changing plotting logic.
Panel D uses noisy readouts of optimal allocations in simulated participants;
these readouts are illustrative coding proxies, not fitted fMRI RSA values.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy import stats


# ---------------------------------------------------------------------------
# Paths and illustrative parameters
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"
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
SPINE_COLOR = "#9AA3AA"
# Local model palette distinguishes allocation from behavioral outcomes and fMRI ROIs.
INFERRED_COLOR = "#B43C39"
DIRECT_COLOR = "#29343D"
TOTAL_COLOR = "#29343D"
REFERENCE_COLOR = "#7A858D"
COMPETITION_COLOR = "#29343D"
LOSS_COLORS = ("#EBC8C7", "#DDA19F", "#CA6B68", "#B43C39")


def configure_matplotlib() -> None:
    """Set journal-friendly defaults and preserve editable PDF text."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9.6,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": SPINE_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "legend.fontsize": 7.8,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


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
    """Apply the same restrained axis treatment to every panel."""

    axis.set_facecolor("white")
    axis.set_axisbelow(True)
    axis.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8, length=3, width=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE_COLOR)
    axis.spines["bottom"].set_color(SPINE_COLOR)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)


def add_panel_header(axis, panel_label: str, title: str) -> None:
    """Place a bold panel letter and left-aligned title above an axis."""

    axis.text(
        -0.10,
        1.105,
        panel_label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    axis.set_title(title, loc="left", pad=15, color=TEXT_COLOR)


def add_example_note(axis) -> None:
    """Label the example parameter values in panel B."""

    axis.text(
        0.98,
        0.96,
        r"Example: $R=4$, $a_P=4$, $a_C=1$",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=MUTED_TEXT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def draw_panel_a(axis) -> None:
    """Plot single-cue loss functions for several computational values."""

    resource = np.linspace(0.0, 5.0, 500)
    a_values = (0.5, 1.0, 2.0, 4.0)
    for a_value, color in zip(a_values, LOSS_COLORS):
        axis.plot(
            resource,
            single_cue_loss(a_value, resource),
            color=color,
            linewidth=2.0,
            label=rf"$a_i={a_value:g}$",
        )

    style_axis(axis)
    add_panel_header(axis, "A", "Identity resources have diminishing returns")
    axis.set_xlim(0.0, 5.0)
    axis.set_ylim(0.0, 4.25)
    axis.set_xlabel(r"Identity representational resource, $r_i$")
    axis.set_ylabel(r"Decision loss, $L_i$")
    axis.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.2,
        borderaxespad=0.3,
        labelspacing=0.35,
    )
    axis.text(
        0.96,
        0.48,
        "Higher computational value\nraises the cost of weak\nidentity representation",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        color=MUTED_TEXT_COLOR,
        linespacing=1.3,
    )


def draw_panel_b(axis) -> None:
    """Plot total loss across all feasible allocations of a fixed budget."""

    budget = EXAMPLE_BUDGET
    resource_p = np.linspace(0.0, budget, 600)
    resource_c = budget - resource_p
    total_loss = single_cue_loss(EXAMPLE_A_P, resource_p) + single_cue_loss(
        EXAMPLE_A_C, resource_c
    )
    r_p_star, r_c_star = optimal_allocation(
        EXAMPLE_A_P, EXAMPLE_A_C, EXAMPLE_BUDGET
    )
    minimum_loss = float(
        single_cue_loss(EXAMPLE_A_P, r_p_star)
        + single_cue_loss(EXAMPLE_A_C, r_c_star)
    )

    axis.plot(resource_p, total_loss, color=TOTAL_COLOR, linewidth=2.4)
    axis.axvline(
        r_p_star,
        color=REFERENCE_COLOR,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
    )
    axis.scatter(
        [r_p_star],
        [minimum_loss],
        s=34,
        facecolor="white",
        edgecolor=TOTAL_COLOR,
        linewidth=1.3,
        zorder=4,
    )

    style_axis(axis)
    add_panel_header(axis, "B", "Optimal allocation under a fixed budget")
    add_example_note(axis)
    axis.set_xlim(0.0, budget)
    axis.set_ylim(0.0, 4.25)
    axis.set_xlabel(r"Resource to inferred identity, $r_P$")
    axis.set_ylabel("Total decision loss")
    axis.annotate(
        "Optimum\n"
        + rf"$r_P^*={r_p_star:.2f}$"
        + "\n"
        + rf"$r_C^*={r_c_star:.2f}$",
        xy=(r_p_star, minimum_loss),
        xytext=(1.85, 1.38),
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": REFERENCE_COLOR,
            "linewidth": 0.9,
        },
    )
    axis.text(
        0.04,
        0.68,
        "Too little resource\nfor inferred identity",
        transform=axis.transAxes,
        color=INFERRED_COLOR,
        fontsize=7.5,
        ha="left",
        va="center",
    )
    axis.text(
        0.96,
        0.55,
        "Too little resource\nfor directly learned identity",
        transform=axis.transAxes,
        color=DIRECT_COLOR,
        fontsize=7.5,
        ha="right",
        va="center",
    )


def draw_panel_c(axis) -> None:
    """Plot the constrained allocation difference across relative values."""

    budget = EXAMPLE_BUDGET
    value_ratio = np.logspace(-2.2, 2.2, 800)
    allocation_difference = np.clip(np.log(value_ratio), -budget, budget)
    direct_mask = value_ratio <= 1.0
    inferred_mask = value_ratio >= 1.0

    axis.plot(
        value_ratio[direct_mask],
        allocation_difference[direct_mask],
        color=DIRECT_COLOR,
        linewidth=2.4,
    )
    axis.plot(
        value_ratio[inferred_mask],
        allocation_difference[inferred_mask],
        color=INFERRED_COLOR,
        linewidth=2.4,
    )
    axis.fill_between(
        value_ratio[direct_mask],
        allocation_difference[direct_mask],
        0.0,
        color=DIRECT_COLOR,
        alpha=0.08,
    )
    axis.fill_between(
        value_ratio[inferred_mask],
        0.0,
        allocation_difference[inferred_mask],
        color=INFERRED_COLOR,
        alpha=0.08,
    )
    axis.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.9)
    axis.axvline(
        1.0,
        color=REFERENCE_COLOR,
        linestyle=(0, (3, 3)),
        linewidth=0.9,
    )
    axis.axhline(
        budget,
        color=GRID_COLOR,
        linestyle=(0, (2, 3)),
        linewidth=0.8,
    )
    axis.axhline(
        -budget,
        color=GRID_COLOR,
        linestyle=(0, (2, 3)),
        linewidth=0.8,
    )

    style_axis(axis, grid_axis="y")
    add_panel_header(axis, "C", "Allocation follows relative value")
    axis.set_xscale("log")
    axis.set_xlim(value_ratio.min(), value_ratio.max())
    axis.set_ylim(-4.4, 4.4)
    axis.xaxis.set_major_locator(FixedLocator([0.01, 0.1, 1.0, 10.0, 100.0]))
    axis.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    axis.set_yticks([-4, -2, 0, 2, 4])
    axis.set_xlabel(r"Relative computational value, $a_P/a_C$")
    axis.set_ylabel(r"Optimal allocation difference, $r_P^*-r_C^*$")
    axis.text(
        0.08,
        -2.0,
        "More resource to\ndirectly learned identity",
        color=DIRECT_COLOR,
        fontsize=7.7,
        fontweight="bold",
        ha="center",
        va="center",
    )
    axis.text(
        13.0,
        1.80,
        "More resource to\ninferred identity",
        color=INFERRED_COLOR,
        fontsize=7.7,
        fontweight="bold",
        ha="center",
        va="center",
    )
    axis.annotate(
        "Equal utility",
        xy=(1.0, 0.0),
        xytext=(1.45, -1.15),
        fontsize=7.5,
        color=MUTED_TEXT_COLOR,
        ha="left",
        arrowprops={
            "arrowstyle": "-",
            "color": REFERENCE_COLOR,
            "linewidth": 0.8,
        },
    )
    axis.text(
        0.985,
        0.09,
        rf"Boundary solutions at $\pm R$ ($R={budget:g}$)",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        color=MUTED_TEXT_COLOR,
    )


def draw_panel_d(axis) -> None:
    """Show the noisy between-participant prediction of a shared budget."""

    participants = simulate_participants()
    x = participants["coding_c"]
    y = participants["coding_p"]
    rho = float(stats.spearmanr(x, y).statistic)
    fit = stats.linregress(x, y)
    fit_x = np.linspace(x.min(), x.max(), 250)
    fit_y = fit.intercept + fit.slope * fit_x
    residual_sd = np.sqrt(np.sum((y - fit.intercept - fit.slope * x)**2) / (len(x) - 2))
    mean_se = residual_sd * np.sqrt(
        1 / len(x) + (fit_x - x.mean())**2 / np.sum((x - x.mean())**2)
    )
    half_width = stats.t.ppf(0.975, len(x) - 2) * mean_se

    axis.fill_between(
        fit_x, fit_y - half_width, fit_y + half_width,
        color=COMPETITION_COLOR, alpha=0.16, linewidth=0,
    )
    axis.plot(fit_x, fit_y, color=COMPETITION_COLOR, linewidth=2.0)
    axis.scatter(
        x, y, s=17, color=COMPETITION_COLOR, alpha=0.75,
        edgecolor="white", linewidth=0.35, zorder=3,
    )
    style_axis(axis)
    add_panel_header(axis, "D", "Direct and inferred coding compete")
    axis.set_xlabel("Direct identity coding (a.u.)")
    axis.set_ylabel("Inferred identity coding (a.u.)")
    axis.margins(x=0.08)
    span_y = np.ptp(y)
    axis.set_ylim(y.min() - 0.18 * span_y, y.max() + 0.30 * span_y)
    axis.text(
        0.97, 0.97,
        rf"Simulated Spearman $\rho={rho:.3f}$" + "\n"
        + rf"$n={len(x)}$ simulated participants",
        transform=axis.transAxes, ha="right", va="top", fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )
    axis.text(
        0.03, 0.04,
        "Shared budget + independent measurement noise\nIllustrative prediction; no fit to fMRI",
        transform=axis.transAxes, ha="left", va="bottom", fontsize=6.9,
        color=MUTED_TEXT_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )


# ---------------------------------------------------------------------------
# Figure assembly and export
# ---------------------------------------------------------------------------


def create_figure():
    """Assemble the four panels at two-column journal width."""

    configure_matplotlib()
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 6.35),
        facecolor="white",
    )
    draw_panel_a(axes[0, 0])
    draw_panel_b(axes[0, 1])
    draw_panel_c(axes[1, 0])
    draw_panel_d(axes[1, 1])
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.925,
        bottom=0.095,
        wspace=0.34,
        hspace=0.46,
    )
    return figure


def save_figure(figure) -> None:
    """Write an editable vector PDF and a high-resolution PNG preview."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        PDF_PATH,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor=figure.get_facecolor(),
    )
    figure.savefig(
        PNG_PATH,
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)
    print(f"Wrote: {PDF_PATH.relative_to(REPO_ROOT)}")
    print(f"Wrote: {PNG_PATH.relative_to(REPO_ROOT)}")


def main() -> None:
    """Validate displayed calculations, build the figure, and export it."""

    validate_example()
    save_figure(create_figure())
    participants = simulate_participants()
    rho = stats.spearmanr(participants["coding_c"], participants["coding_p"]).statistic
    print(
        f"Panel D: n={PARTICIPANT_COUNT}, Spearman rho={rho:.6f}, "
        f"seed={PARTICIPANT_SEED}, measurement noise SD={MEASUREMENT_NOISE_SD}"
    )


if __name__ == "__main__":
    main()
