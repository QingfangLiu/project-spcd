import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Single source of truth
# =========================================================

CONDITIONS = [
    "A11", "A21", "C1", "B11", "B21", "D1",
    "A12", "A22", "C2", "B12", "B22", "D2",
]


def format_condition_label(cond: str) -> str:
    """Convert condition labels to paper-style with subscript."""
    if cond and cond[0] in ("A", "B") and len(cond) == 3:
        return f"{cond[0]}{cond[1]}$_{cond[2]}$"
    if cond and cond[0] in ("C", "D") and len(cond) == 2:
        return f"{cond[0]}$_{cond[1]}$"
    return cond


# =========================================================
# Generic RDM utilities
# =========================================================

def build_group_map(group_dict: dict) -> dict:
    """group_dict: {group_name: [cond, ...]} -> {cond: group_name}"""
    cond_to_group = {}
    for g, conds in group_dict.items():
        for c in conds:
            cond_to_group[c] = g
    return cond_to_group


def build_rdm_from_groups(
    conditions: list,
    cond_to_group: dict,
    diagonal: float = 1.0,
    dtype=float,
):
    """
    Build a (n x n) RDM with:
      - diagonal = diagonal
      - off-diagonal = 1 if same group else 0
    """
    n = len(conditions)
    rdm = np.zeros((n, n), dtype=dtype)

    for i, ci in enumerate(conditions):
        gi = cond_to_group[ci]
        for j, cj in enumerate(conditions):
            if i == j:
                rdm[i, j] = diagonal
            else:
                rdm[i, j] = 1.0 if gi == cond_to_group[cj] else 0.0

    return rdm


def reorder_by_groups(conditions: list, group_dict: dict, group_order: list):
    """
    Returns:
      idx: indices into `conditions` in the desired order
      ordered_conditions: reordered condition labels
    """
    ordered_conditions = []
    for g in group_order:
        ordered_conditions.extend(group_dict[g])

    cond_to_idx = {c: i for i, c in enumerate(conditions)}
    idx = [cond_to_idx[c] for c in ordered_conditions]
    return idx, ordered_conditions


def plot_rdm_heatmap_pdf(
    rdm: np.ndarray,
    ordered_conditions: list,
    save_path: str,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    im = ax.imshow(
        rdm,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="Greys",
        zorder=2,
    )

    ax.set_title(title, fontsize=16, pad=6)

    labels = [format_condition_label(c) for c in ordered_conditions]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Subtle grid
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", linewidth=0.4, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["0", "1"])

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return save_path


# =========================================================
# Model configurations (only differences live here)
# =========================================================

IDENTITY_GROUPS = {
    "Odor 1": ["A11", "B11", "A12", "B12"],
    "Odor 2": ["A21", "B21", "A22", "B22"],
    "Clean":  ["C1",  "D1",  "C2",  "D2"],
}
IDENTITY_GROUP_ORDER = ["Odor 1", "Odor 2", "Clean"]

VALUE_GROUPS = {
    "Odor (merged)": ["A11", "B11", "A12", "B12", "A21", "B21", "A22", "B22"],
    "Clean":         ["C1", "D1", "C2", "D2"],
}
VALUE_GROUP_ORDER = ["Odor (merged)", "Clean"]

ASSOCIATION_GROUPS = {
    "A11~B11": ["A11", "B11"],
    "A21~B21": ["A21", "B21"],
    "A12~B12": ["A12", "B12"],
    "A22~B22": ["A22", "B22"],
    "C1~D1":   ["C1",  "D1"],
    "C2~D2":   ["C2",  "D2"],
}
ASSOCIATION_GROUP_ORDER = ["A11~B11", "A21~B21", "C1~D1", "A12~B12", "A22~B22", "C2~D2"]


# =========================================================
# Public functions matching old behavior
# =========================================================

def create_identity_model_rdm(diagonal: float = 1.0):
    cond_to_group = build_group_map(IDENTITY_GROUPS)
    rdm = build_rdm_from_groups(CONDITIONS, cond_to_group, diagonal=diagonal, dtype=float)
    return rdm, CONDITIONS, IDENTITY_GROUPS


def create_value_model_rdm(diagonal: float = 1.0):
    cond_to_group = build_group_map(VALUE_GROUPS)
    rdm = build_rdm_from_groups(CONDITIONS, cond_to_group, diagonal=diagonal, dtype=float)
    return rdm

def create_association_model_rdm(diagonal: float = 1.0):
    cond_to_group = build_group_map(ASSOCIATION_GROUPS)
    rdm = build_rdm_from_groups(CONDITIONS, cond_to_group, diagonal=diagonal, dtype=float)
    return rdm


# =========================================================
# Conditioned vs Preconditioned splits (A/C vs B/D)
# =========================================================

PRECONDITIONED_CONDS = ["A11", "A12", "A21", "A22", "C1", "C2"]  # A and C
CONDITIONED_CONDS    = ["B11", "B12", "B21", "B22", "D1", "D2"]  # B and D


def subset_rdm(
    rdm: np.ndarray,
    all_conditions: list,
    subset_conditions: list,
):
    """
    Return the sub-RDM restricted to subset_conditions, preserving the subset order.
    """
    cond_to_idx = {c: i for i, c in enumerate(all_conditions)}
    idx = [cond_to_idx[c] for c in subset_conditions]
    return rdm[np.ix_(idx, idx)], subset_conditions


def plot_rdm_split_conditioning(
    rdm: np.ndarray,
    all_conditions: list,
    save_path_prefix: str,
    title_prefix: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Save two PDFs:
      - <prefix>_preconditioned.pdf : only A/C cues
      - <prefix>_conditioned.pdf    : only B/D cues
    """
    rdm_pre, conds_pre = subset_rdm(rdm, all_conditions, PRECONDITIONED_CONDS)
    plot_rdm_heatmap_pdf(
        rdm_pre,
        ordered_conditions=conds_pre,
        save_path=f"{save_path_prefix}_preconditioned.pdf",
        title=f"{title_prefix} (preconditioned: A/C)",
        vmin=vmin,
        vmax=vmax,
    )

    rdm_con, conds_con = subset_rdm(rdm, all_conditions, CONDITIONED_CONDS)
    plot_rdm_heatmap_pdf(
        rdm_con,
        ordered_conditions=conds_con,
        save_path=f"{save_path_prefix}_conditioned.pdf",
        title=f"{title_prefix} (conditioned: B/D)",
        vmin=vmin,
        vmax=vmax,
    )




def plot_identity_rdm_heatmap_pdf(
    save_path: str = "identity_model_rdm.pdf",
    title: str = "Identity model RDM",
):
    rdm, conditions, groups = create_identity_model_rdm(diagonal=1.0)
    idx, ordered_conditions = reorder_by_groups(conditions, groups, IDENTITY_GROUP_ORDER)
    rdm_ord = rdm[np.ix_(idx, idx)]
    return plot_rdm_heatmap_pdf(rdm_ord, ordered_conditions, save_path=save_path, title=title)


def plot_value_rdm_heatmap_pdf(
    save_path: str = "value_model_rdm.pdf",
    title: str = "Value model RDM (odor vs clean)",
):
    rdm = create_value_model_rdm(diagonal=1.0)
    idx, ordered_conditions = reorder_by_groups(CONDITIONS, VALUE_GROUPS, VALUE_GROUP_ORDER)
    rdm_ord = rdm[np.ix_(idx, idx)]
    return plot_rdm_heatmap_pdf(rdm_ord, ordered_conditions, save_path=save_path, title=title)


def plot_association_rdm_heatmap_pdf(
    save_path: str = "association_model_rdm.pdf",
    title: str = "Association model RDM (paired cues)",
):
    rdm = create_association_model_rdm(diagonal=1.0)
    idx, ordered_conditions = reorder_by_groups(CONDITIONS, ASSOCIATION_GROUPS, ASSOCIATION_GROUP_ORDER)
    rdm_ord = rdm[np.ix_(idx, idx)]
    return plot_rdm_heatmap_pdf(rdm_ord, ordered_conditions, save_path=save_path, title=title)



# =========================================================
# Split-plot wrappers for each model
# =========================================================

def plot_identity_rdm_split_pdfs(save_path_prefix: str = "identity_model_rdm"):
    rdm, conditions, _groups = create_identity_model_rdm(diagonal=1.0)
    plot_rdm_split_conditioning(
        rdm=rdm,
        all_conditions=conditions,
        save_path_prefix=save_path_prefix,
        title_prefix="Identity model RDM",
    )


def plot_value_rdm_split_pdfs(save_path_prefix: str = "value_model_rdm"):
    rdm = create_value_model_rdm(diagonal=1.0)
    plot_rdm_split_conditioning(
        rdm=rdm,
        all_conditions=CONDITIONS,
        save_path_prefix=save_path_prefix,
        title_prefix="Value model RDM (odor vs clean)",
    )


def plot_association_rdm_split_pdfs(save_path_prefix: str = "association_model_rdm"):
    rdm = create_association_model_rdm(diagonal=1.0)
    plot_rdm_split_conditioning(
        rdm=rdm,
        all_conditions=CONDITIONS,
        save_path_prefix=save_path_prefix,
        title_prefix="Association model RDM (paired cues)",
    )




if __name__ == "__main__":
    plot_identity_rdm_heatmap_pdf()
    plot_value_rdm_heatmap_pdf()
    plot_association_rdm_heatmap_pdf()

    # Split (preconditioned vs conditioned)
    plot_identity_rdm_split_pdfs()
    plot_value_rdm_split_pdfs()
    plot_association_rdm_split_pdfs()




