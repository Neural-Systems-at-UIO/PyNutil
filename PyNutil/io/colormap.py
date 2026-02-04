"""Colormap utilities for PyNutil.

This module provides simple colormap implementations for mapping
intensity values to RGB colors, avoiding matplotlib dependency.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


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
    value = np.clip(value, 0, 255) / 255.0

    if name == "gray":
        v = int(value * 255)
        return v, v, v

    # Simple implementations of common colormaps
    if name == "viridis":
        # Simplified viridis approximation
        r = 1.0 - value
        g = value
        b = 0.5 + 0.5 * value
    elif name == "plasma":
        r = value
        g = 1.0 - value
        b = 1.0 - 0.5 * value
    elif name == "magma":
        r = value
        g = value**2
        b = 1.0 - value
    elif name == "hot":
        r = min(1.0, value * 3)
        g = min(1.0, max(0.0, value * 3 - 1))
        b = min(1.0, max(0.0, value * 3 - 2))
    else:
        # Default to gray
        v = int(value * 255)
        return v, v, v

    return int(r * 255), int(g * 255), int(b * 255)
