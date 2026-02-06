"""Colormap utilities for PyNutil.

This module provides simple colormap implementations for mapping
intensity values to RGB colors, avoiding matplotlib dependency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _viridis(t):
    return 1.0 - t, t, 0.5 + 0.5 * t


def _plasma(t):
    return t, 1.0 - t, 1.0 - 0.5 * t


def _magma(t):
    return t, t**2, 1.0 - t


def _hot(t):
    return min(1.0, t * 3), min(1.0, max(0.0, t * 3 - 1)), min(1.0, max(0.0, t * 3 - 2))


_COLORMAPS = {
    "viridis": _viridis,
    "plasma": _plasma,
    "magma": _magma,
    "hot": _hot,
}


def get_colormap_color(value: int, name: str = "gray") -> Tuple[int, int, int]:
    """Map an intensity value (0-255) to RGB color based on colormap name.

    Parameters
    ----------
    value : int
        Intensity value (0-255).
    name : str
        Colormap name. Options: "gray", "viridis", "plasma", "magma", "hot".

    Returns
    -------
    tuple
        (r, g, b) color values (0-255).
    """
    t = np.clip(value, 0, 255) / 255.0

    fn = _COLORMAPS.get(name)
    if fn is None:
        v = int(t * 255)
        return v, v, v

    r, g, b = fn(t)
    return int(r * 255), int(g * 255), int(b * 255)
