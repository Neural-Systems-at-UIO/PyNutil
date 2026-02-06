"""Coordinate transformation utilities for PyNutil.

This module consolidates all coordinate transformation functions:
- Scaling between segmentation and registration spaces
- Linear transformation using QuickNII anchoring vectors
- Non-linear deformation using VisuAlign markers
- Region area computation from atlas maps
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

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
    return o[None, :] + (x_scale[:, None] * u[None, :]) + (y_scale[:, None] * v[None, :])


# -----------------------------------------------------------------------------
# Non-linear transformation
# -----------------------------------------------------------------------------


def get_transformed_coordinates(
    non_linear: bool,
    slice_dict: Dict[str, Any],
    scaled_x: Optional[np.ndarray],
    scaled_y: Optional[np.ndarray],
    scaled_centroidsX: Optional[np.ndarray],
    scaled_centroidsY: Optional[np.ndarray],
    deformation: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply non-linear deformation to scaled coordinates.

    If non_linear is True and a deformation function is provided, applies the
    deformation. Otherwise, passes coordinates through unchanged.

    Parameters
    ----------
    non_linear : bool
        Whether to apply non-linear transformation.
    slice_dict : dict
        Slice metadata (kept for compatibility, not used for markers).
    scaled_x : ndarray or None
        Scaled x coordinates for points.
    scaled_y : ndarray or None
        Scaled y coordinates for points.
    scaled_centroidsX : ndarray or None
        Scaled x coordinates for centroids.
    scaled_centroidsY : ndarray or None
        Scaled y coordinates for centroids.
    deformation : callable or None
        Deformation function that takes (x, y) and returns (new_x, new_y).

    Returns
    -------
    tuple
        (new_x, new_y, centroids_new_x, centroids_new_y) - deformed coordinates.
    """
    new_x, new_y, centroids_new_x, centroids_new_y = None, None, None, None

    if non_linear and deformation is not None:
        if scaled_x is not None:
            new_x, new_y = deformation(scaled_x, scaled_y)
        if scaled_centroidsX is not None:
            centroids_new_x, centroids_new_y = deformation(
                scaled_centroidsX, scaled_centroidsY
            )
    else:
        new_x, new_y = scaled_x, scaled_y
        centroids_new_x, centroids_new_y = scaled_centroidsX, scaled_centroidsY

    return new_x, new_y, centroids_new_x, centroids_new_y


def transform_points_to_atlas_space(
    anchoring: List[float],
    new_x: Optional[np.ndarray],
    new_y: Optional[np.ndarray],
    centroids_new_x: Optional[np.ndarray],
    centroids_new_y: Optional[np.ndarray],
    reg_height: int,
    reg_width: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Transform points and centroids to atlas space.

    Final step in the transformation pipeline: applies the anchoring vector
    to convert 2D registration coordinates to 3D atlas coordinates.

    Parameters
    ----------
    anchoring : list
        Anchoring vector (12 floats).
    new_x : ndarray or None
        Transformed X coordinates.
    new_y : ndarray or None
        Transformed Y coordinates.
    centroids_new_x : ndarray or None
        Transformed X coordinates of centroids.
    centroids_new_y : ndarray or None
        Transformed Y coordinates of centroids.
    reg_height : int
        Registration height.
    reg_width : int
        Registration width.

    Returns
    -------
    tuple
        (points, centroids) - (N, 3) arrays of atlas-space coordinates.
    """
    points, centroids = None, None

    if new_x is not None:
        points = transform_to_atlas_space(
            anchoring, new_y, new_x, reg_height, reg_width
        )
    if centroids_new_x is not None:
        centroids = transform_to_atlas_space(
            anchoring,
            centroids_new_y,
            centroids_new_x,
            reg_height,
            reg_width,
        )
    return points, centroids
