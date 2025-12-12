import matplotlib.pyplot as plt
import seaborn as sns

# ----- Global rcParams -----
plt.rcParams.update({
    # Figure
    "figure.figsize": (6, 5),
    "figure.dpi": 300,

    # Fonts
    "font.family": "Arial",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 12,

    # Axes
    "axes.linewidth": 1.2,
    "axes.labelpad": 6,

    # Ticks
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 5,
    "ytick.major.size": 5,

    # Lines
    "lines.linewidth": 2.0,
    "lines.markersize": 6,

    # Grid
    "axes.grid": False,
})

# Seaborn base style
sns.set_style("white")

# Shared colors
session_colors = {
    "pre":  "#2C7BB6",  # Cerulean Blue
    "post": "#D7191C",  # Vermilion
}
