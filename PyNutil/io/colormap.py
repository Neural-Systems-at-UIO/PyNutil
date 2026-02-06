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
    return (
        np.minimum(1.0, t * 3),
        np.minimum(1.0, np.maximum(0.0, t * 3 - 1)),
        np.minimum(1.0, np.maximum(0.0, t * 3 - 2)),
    )


_COLORMAPS = {
    "viridis": _viridis,
    "plasma": _plasma,
    "magma": _magma,
    "hot": _hot,
}

# Pre-built 256-entry LUT cache:  {name: (N, 3) uint8 array}
_LUT_CACHE: dict[str, np.ndarray] = {}


def _build_lut(name: str) -> np.ndarray:
    """Build a (256, 3) uint8 lookup table for *name*."""
    if name in _LUT_CACHE:
        return _LUT_CACHE[name]
    t = np.arange(256, dtype=np.float64) / 255.0
    fn = _COLORMAPS.get(name)
    if fn is None:
        # gray
        v = np.arange(256, dtype=np.uint8)
        lut = np.column_stack([v, v, v])
    else:
        r, g, b = fn(t)
        lut = np.column_stack([
            np.clip(np.asarray(r) * 255, 0, 255).astype(np.uint8),
            np.clip(np.asarray(g) * 255, 0, 255).astype(np.uint8),
            np.clip(np.asarray(b) * 255, 0, 255).astype(np.uint8),
        ])
    _LUT_CACHE[name] = lut
    return lut


def get_colormap_colors(values: np.ndarray, name: str = "gray") -> np.ndarray:
    """Vectorised colormap lookup for an array of intensity values (0-255).

    Returns a (N, 3) uint8 array of RGB colours.
    """
    lut = _build_lut(name)
    idx = np.clip(values, 0, 255).astype(np.intp)
    return lut[idx]


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
    lut = _build_lut(name)
    idx = int(np.clip(value, 0, 255))
    row = lut[idx]
    return int(row[0]), int(row[1]), int(row[2])
