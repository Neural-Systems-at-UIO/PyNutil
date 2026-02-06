"""MeshView JSON writer for PyNutil.

This module handles writing point data to MeshView-compatible JSON format,
supporting both atlas-region-based and intensity-based point clouds.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .atlas_loader import load_atlas_labels
from .colormap import get_colormap_color


def create_region_dict(
    points: np.ndarray,
    regions: np.ndarray,
) -> Dict[int, List[float]]:
    """Group point coordinates by their region labels.

    Parameters
    ----------
    points : np.ndarray
        A (N, 3) array of 3D coordinates for all points.
    regions : np.ndarray
        A 1D array of integer region labels for each point.

    Returns
    -------
    dict
        Keys are unique region labels, values are flattened [x, y, z, ...] coordinates.
    """
    region_dict = {
        region: points[regions == region].flatten().tolist()
        for region in np.unique(regions)
    }
    return region_dict


def _meshview_entry(idx, name, triplets, r, g, b):
    """Build a single MeshView JSON entry dict."""
    return {
        "idx": idx,
        "count": len(triplets) // 3,
        "name": name,
        "triplets": triplets,
        "r": int(r),
        "g": int(g),
        "b": int(b),
    }


def _write_points(
    points_dict: Dict[int, List[float]],
    filename: str,
    info_file: pd.DataFrame,
    colors_dict: Optional[Dict[int, tuple]] = None,
) -> None:
    """Save a region-based point dictionary to MeshView JSON.

    Each region is recorded with: index (idx), name, color components (r, g, b),
    and a count of how many points belong to that region.

    Parameters
    ----------
    points_dict : dict
        Keys are region IDs, values are flattened 3D coordinates.
    filename : str
        Destination JSON file to be written.
    info_file : pd.DataFrame
        A table with region IDs, names, and color data (r, g, b).
    colors_dict : dict, optional
        Keys are region IDs, values are (r, g, b) tuples to override atlas colors.
    """
    meshview = []
    for name, idx in zip(points_dict.keys(), range(len(points_dict.keys()))):
        region_info = info_file[info_file["idx"] == name]
        if len(region_info) == 0:
            continue

        r = int(region_info["r"].values[0])
        g = int(region_info["g"].values[0])
        b = int(region_info["b"].values[0])

        if colors_dict is not None and name in colors_dict:
            r, g, b = colors_dict[name]

        meshview.append(_meshview_entry(
            idx, str(region_info["name"].values[0]), points_dict[name], r, g, b,
        ))

    with open(filename, "w") as f:
        json.dump(meshview, f)


def write_hemi_points_to_meshview(
    points: Optional[np.ndarray],
    point_names: Optional[np.ndarray],
    hemi_label: Optional[np.ndarray],
    filename: str,
    info_file: Union[pd.DataFrame, str],
    intensities: Optional[np.ndarray] = None,
    colormap: str = "gray",
) -> None:
    """Write point data to MeshView JSON, optionally split by hemisphere.

    If hemisphere labels are provided (1, 2), separate outputs are saved for
    left and right hemispheres, as well as one file containing all points.

    Parameters
    ----------
    points : np.ndarray or None
        2D array containing [N, 3] point coordinates.
    point_names : np.ndarray or None
        1D array of region labels corresponding to each point.
    hemi_label : np.ndarray or None
        1D array with hemisphere labels (1 for left, 2 for right).
    filename : str
        Base path for output JSON. Separate hemispheres use prefixed filenames.
    info_file : pd.DataFrame or str
        Atlas labels DataFrame, or atlas name string.
    intensities : np.ndarray, optional
        1D array of intensity values for each point.
    colormap : str, optional
        Colormap to use for intensity mode (default is "gray").
    """
    if points is None or point_names is None:
        return

    if hemi_label is not None and not (hemi_label == None).all():
        for hval, prefix in ((1, "left_hemisphere_"), (2, "right_hemisphere_")):
            parts = filename.split("/")
            parts[-1] = prefix + parts[-1]
            hemi_path = os.sep.join(parts)
            mask = hemi_label == hval
            write_points_to_meshview(
                points[mask],
                point_names[mask],
                hemi_path,
                info_file,
                intensities[mask] if intensities is not None else None,
                colormap,
            )

    write_points_to_meshview(
        points, point_names, filename, info_file, intensities, colormap
    )


def write_points_to_meshview(
    points: np.ndarray,
    point_ids: np.ndarray,
    filename: str,
    info_file: Union[pd.DataFrame, str],
    intensities: Optional[np.ndarray] = None,
    colormap: str = "gray",
) -> None:
    """Write point data to MeshView JSON format.

    Parameters
    ----------
    points : np.ndarray
        2D array containing [N, 3] point coordinates.
    point_ids : np.ndarray
        1D array of region labels corresponding to each point.
    filename : str
        Output JSON file path.
    info_file : pd.DataFrame or str
        Atlas labels DataFrame, or atlas name string for brainglobe.
    intensities : np.ndarray, optional
        1D array of intensity values for each point.
    colormap : str, optional
        Colormap to use for intensity mode (default is "gray").
    """
    if isinstance(info_file, str):
        info_file = load_atlas_labels(info_file)

    if intensities is not None:
        _write_intensity_meshview(points, intensities, filename, colormap)
        return

    region_dict = create_region_dict(points, point_ids)
    _write_points(region_dict, filename, info_file)


def _write_intensity_meshview(
    points: np.ndarray,
    intensities: np.ndarray,
    filename: str,
    colormap: str,
) -> None:
    """Write intensity-based point cloud to MeshView JSON.

    Groups points by intensity bins for efficient visualization.
    """
    if colormap == "original_colours" and intensities.ndim == 2 and intensities.shape[1] == 3:
        _write_rgb_meshview(points, intensities, filename)
        return

    _write_scalar_meshview(points, intensities, filename, colormap)


def _write_rgb_meshview(
    points: np.ndarray,
    intensities: np.ndarray,
    filename: str,
) -> None:
    """Write RGB-coloured point cloud to MeshView JSON."""
    rgb_data = intensities.astype(np.uint8)
    unique_colors, inverse_indices = np.unique(rgb_data, axis=0, return_inverse=True)

    # If there are too many unique colors, MeshView UI becomes slow.
    # Rounding to nearest 8 (32 levels per channel) keeps it responsive.
    if len(unique_colors) > 1024:
        rgb_data = (np.round(rgb_data / 8) * 8)
        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
        unique_colors, inverse_indices = np.unique(rgb_data, axis=0, return_inverse=True)

    meshview = []
    for i, color in enumerate(unique_colors):
        mask = inverse_indices == i
        bin_points = points[mask]
        if len(bin_points) > 0:
            r, g, b = color
            triplets = bin_points.flatten().tolist()
            meshview.append(_meshview_entry(
                i, f"Color {r},{g},{b}", triplets, r, g, b,
            ))

    with open(filename, "w") as f:
        json.dump(meshview, f)


def _write_scalar_meshview(
    points: np.ndarray,
    intensities: np.ndarray,
    filename: str,
    colormap: str,
) -> None:
    """Write grayscale/colormap point cloud to MeshView JSON."""
    if intensities.ndim == 2 and intensities.shape[1] == 3:
        # Convert RGB to grayscale for binning
        intensities = (
            0.2989 * intensities[:, 0] +
            0.5870 * intensities[:, 1] +
            0.1140 * intensities[:, 2]
        ).astype(int)
    else:
        intensities = intensities.astype(int)

    unique_intensities = np.unique(intensities)

    meshview = []
    for val in unique_intensities:
        mask = intensities == val
        bin_points = points[mask]

        if len(bin_points) > 0:
            r, g, b = get_colormap_color(val, colormap)
            triplets = bin_points.flatten().tolist()
            meshview.append(_meshview_entry(
                int(val), f"Intensity {val}", triplets, r, g, b,
            ))

    with open(filename, "w") as f:
        json.dump(meshview, f)
