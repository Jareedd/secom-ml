"""One shared look for every figure, so they read as a set."""

import matplotlib

matplotlib.use("Agg")  # write image files; never open a window

import matplotlib.pyplot as plt

# Series colors, used in this fixed order whenever a chart has several series.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Quiet axes and grid so the data is the darkest thing on the chart."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
