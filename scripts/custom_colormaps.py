"""Reusable custom Matplotlib colormap helpers."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, ListedColormap


def diverging_with_white_plateau(
    cmap: str | Colormap,
    vmin: float,
    vmax: float,
    white_half_width: float,
    *,
    center: float = 0.0,
    samples: int = 256,
    name: str | None = None,
) -> ListedColormap:
    """Return a diverging colormap with a flat white central value interval.

    Values from ``center - white_half_width`` through
    ``center + white_half_width`` map to pure white. Colors outside that
    interval retain the original colormap's mapping over ``vmin`` to ``vmax``.
    """
    if not vmin < vmax:
        raise ValueError("vmin must be less than vmax")
    if white_half_width < 0.0:
        raise ValueError("white_half_width must be non-negative")
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if center - white_half_width < vmin or center + white_half_width > vmax:
        raise ValueError("The central white interval must lie within vmin and vmax")

    source = mpl.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
    positions = np.linspace(0.0, 1.0, samples)
    values = vmin + positions * (vmax - vmin)
    colors = source(positions)
    in_plateau = np.abs(values - center) <= white_half_width
    colors[in_plateau] = mpl.colors.to_rgba("white")
    output_name = name or f"{source.name}_white_{white_half_width:g}"
    return ListedColormap(colors, name=output_name)
