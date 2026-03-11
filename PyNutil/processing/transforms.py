"""Coordinate transformation utilities for PyNutil.

This module consolidates all coordinate transformation functions:
- Scaling between segmentation and registration spaces
- Linear transformation using QuickNII anchoring vectors
- Non-linear deformation using VisuAlign markers
- Region area computation from atlas maps
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .atlas_map import get_region_areas  # noqa: F401  — re-exported for backward compat


# -----------------------------------------------------------------------------
# Coordinate scaling
# -----------------------------------------------------------------------------


def transform_to_registration(
    seg_height: int,
    seg_width: int,
    reg_height: int,
    reg_width: int,
) -> Tuple[float, float]:
    """Compute scaling factors from segmentation to registration space.

    Parameters
    ----------
    seg_height : int
        Segmentation height.
    seg_width : int
        Segmentation width.
    reg_height : int
        Registration height.
    reg_width : int
        Registration width.

    Returns
    -------
    tuple
        (y_scale, x_scale) factors.
    """
    y_scale = reg_height / seg_height
    x_scale = reg_width / seg_width
    return y_scale, x_scale


# -----------------------------------------------------------------------------
# Atlas space transformation
# -----------------------------------------------------------------------------


def transform_to_atlas_space(
    anchoring: List[float],
    y: np.ndarray,
    x: np.ndarray,
    reg_height: int,
    reg_width: int,
) -> np.ndarray:
    """Transform coordinates to atlas space using QuickNII anchoring vector.

    The anchoring vector encodes a 3D affine transformation from 2D section
    coordinates to 3D atlas coordinates:
        atlas_coord = O + (x/width) * U + (y/height) * V

    Parameters
    ----------
    anchoring : list
        9-element anchoring vector [O[3], U[3], V[3]].
    y : ndarray
        Y coordinates in registration space.
    x : ndarray
        X coordinates in registration space.
    reg_height : int
        Registration height.
    reg_width : int
        Registration width.

    Returns
    -------
    ndarray
        (N, 3) array of transformed 3D coordinates.
    """
    # NOTE: This implementation intentionally avoids building intermediate arrays via
    # np.array([row0, row1, row2]).T, which has been observed to miscompute under
    # some Python/numpy builds for large inputs.
    o = np.asarray(anchoring[0:3], dtype=np.float64)
    u = np.asarray(anchoring[3:6], dtype=np.float64)
    v = np.asarray(anchoring[6:9], dtype=np.float64)

    y_arr = np.asarray(y, dtype=np.float64).ravel()
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    y_scale = y_arr / float(reg_height)
    x_scale = x_arr / float(reg_width)

    # Shape: (N, 3)
    return (
        o[None, :] + (x_scale[:, None] * u[None, :]) + (y_scale[:, None] * v[None, :])
    )


# -----------------------------------------------------------------------------
# Non-linear transformation
# -----------------------------------------------------------------------------

