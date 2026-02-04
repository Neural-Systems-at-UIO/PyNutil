"""Coordinate transformation utilities for PyNutil.

This module consolidates all coordinate transformation functions:
- Scaling between segmentation and registration spaces
- Linear transformation using QuickNII anchoring vectors
- Non-linear deformation using VisuAlign markers
- Region area computation from atlas maps
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .visualign_deformations import transform_vec, triangulate
from . import counting_and_load


# -----------------------------------------------------------------------------
# Triangulation setup
# -----------------------------------------------------------------------------


def get_triangulation(
    slice_dict: Dict[str, Any],
    reg_width: int,
    reg_height: int,
    non_linear: bool,
) -> Optional[Any]:
    """Generate triangulation data if non-linear markers exist.

    Parameters
    ----------
    slice_dict : dict
        Slice metadata containing 'markers' key if non-linear transform is needed.
    reg_width : int
        Registration width in pixels.
    reg_height : int
        Registration height in pixels.
    non_linear : bool
        Whether to apply non-linear transformation.

    Returns
    -------
    triangulation or None
        Triangulation structure for deformation, or None if not applicable.
    """
    if non_linear and "markers" in slice_dict:
        return triangulate(reg_width, reg_height, slice_dict["markers"])
    return None


# -----------------------------------------------------------------------------
# Region area computation
# -----------------------------------------------------------------------------


def get_region_areas(
    use_flat: bool,
    atlas_labels,
    flat_file_atlas: Optional[str],
    seg_width: int,
    seg_height: int,
    slice_dict: Dict[str, Any],
    atlas_volume: np.ndarray,
    hemi_mask: Optional[np.ndarray],
    triangulation: Optional[Any],
    damage_mask: Optional[np.ndarray],
) -> Tuple[Any, np.ndarray]:
    """Build the atlas map for a slice and compute region areas.

    This performs the atlas slice extraction (from volume or flat file), applies
    non-linear warping if requested (via `triangulation`), and converts the
    resulting label map into a region-area dataframe.

    Parameters
    ----------
    use_flat : bool
        Whether to use flat file instead of atlas volume.
    atlas_labels : pd.DataFrame
        Atlas label definitions.
    flat_file_atlas : str or None
        Path to flat file atlas if use_flat is True.
    seg_width : int
        Segmentation image width.
    seg_height : int
        Segmentation image height.
    slice_dict : dict
        Slice metadata including 'anchoring' and dimensions.
    atlas_volume : np.ndarray
        3D atlas annotation volume.
    hemi_mask : np.ndarray or None
        Hemisphere mask volume.
    triangulation : optional
        Triangulation structure for non-linear deformation.
    damage_mask : np.ndarray or None
        Damage mask for the section.

    Returns
    -------
    region_areas : pd.DataFrame
        DataFrame with region area statistics.
    atlas_map : np.ndarray
        2D atlas map for the section.
    """
    reg_width, reg_height = slice_dict["width"], slice_dict["height"]
    atlas_map = counting_and_load.load_image(
        flat_file_atlas,
        slice_dict["anchoring"],
        atlas_volume,
        triangulation,
        (reg_width, reg_height),
        atlas_labels,
    )
    region_areas = counting_and_load.flat_to_dataframe(
        atlas_map, damage_mask, hemi_mask, (seg_width, seg_height)
    )
    return region_areas, atlas_map


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


def image_to_atlas_space(image: np.ndarray, anchoring: List[float]) -> np.ndarray:
    """Transform an entire image to atlas space.

    Creates atlas-space coordinates for every pixel in the image.

    Parameters
    ----------
    image : ndarray
        Input image whose pixels will be transformed.
    anchoring : list
        9-element anchoring vector.

    Returns
    -------
    ndarray
        (height*width, 3) array of transformed coordinates.
    """
    width = image.shape[1]
    height = image.shape[0]
    x = np.arange(width)
    y = np.arange(height)
    x_coords, y_coords = np.meshgrid(x, y)
    coordinates = transform_to_atlas_space(
        anchoring, y_coords.flatten(), x_coords.flatten(), height, width
    )
    return coordinates


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
    triangulation: Optional[Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply non-linear deformation to scaled coordinates.

    If non_linear is True and markers exist in slice_dict, applies the
    triangulation-based deformation. Otherwise, passes coordinates through unchanged.

    Parameters
    ----------
    non_linear : bool
        Whether to apply non-linear transformation.
    slice_dict : dict
        Slice metadata including 'markers' key.
    scaled_x : ndarray or None
        Scaled x coordinates for points.
    scaled_y : ndarray or None
        Scaled y coordinates for points.
    scaled_centroidsX : ndarray or None
        Scaled x coordinates for centroids.
    scaled_centroidsY : ndarray or None
        Scaled y coordinates for centroids.
    triangulation : optional
        Triangulation structure from get_triangulation().

    Returns
    -------
    tuple
        (new_x, new_y, centroids_new_x, centroids_new_y) - deformed coordinates.
    """
    new_x, new_y, centroids_new_x, centroids_new_y = None, None, None, None

    if non_linear and "markers" in slice_dict:
        if scaled_x is not None:
            new_x, new_y = transform_vec(triangulation, scaled_x, scaled_y)
        if scaled_centroidsX is not None:
            centroids_new_x, centroids_new_y = transform_vec(
                triangulation, scaled_centroidsX, scaled_centroidsY
            )
    else:
        new_x, new_y = scaled_x, scaled_y
        centroids_new_x, centroids_new_y = scaled_centroidsX, scaled_centroidsY

    return new_x, new_y, centroids_new_x, centroids_new_y


def transform_points_to_atlas_space(
    slice_dict: Dict[str, Any],
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
    slice_dict : dict
        Slice metadata with 'anchoring' vector.
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
            slice_dict["anchoring"], new_y, new_x, reg_height, reg_width
        )
    if centroids_new_x is not None:
        centroids = transform_to_atlas_space(
            slice_dict["anchoring"],
            centroids_new_y,
            centroids_new_x,
            reg_height,
            reg_width,
        )
    return points, centroids
