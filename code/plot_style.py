#!/usr/bin/env python3
"""
Shared plotting style utilities for consistent color usage.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "gray": "#4D4D4D",
    "black": "#000000",
}

LINE_COLORS = [
    PALETTE["blue"],
    PALETTE["vermillion"],
    PALETTE["green"],
    PALETTE["purple"],
    PALETTE["sky"],
    PALETTE["orange"],
    PALETTE["gray"],
]

CORPUS_COLORS = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["purple"],
]

CATEGORY_COLORS = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["green"],
    PALETTE["purple"],
]

MODEL_COLORS = {
    "full": PALETTE["blue"],
    "no_parallelism": PALETTE["orange"],
    "no_licensing": PALETTE["green"],
}

RIDGELINE_CMAP = LinearSegmentedColormap.from_list(
    "hpc_blues",
    ["#DCE9F8", PALETTE["blue"]],
)


def set_plot_style() -> None:
    """Apply a consistent color cycle across matplotlib plots."""
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=LINE_COLORS)
