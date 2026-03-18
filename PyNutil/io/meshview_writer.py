"""MeshView JSON writer for PyNutil.

This module handles writing point data to MeshView-compatible JSON format,
supporting both atlas-region-based and intensity-based point clouds.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import orjson

import numpy as np
import pandas as pd

from .atlas_loader import load_atlas_labels
from .colormap import get_colormap_colors


def _group_triplets(
    points: np.ndarray,
    key_columns: dict[str, np.ndarray],
):
    """Yield grouped triplets from points using NumPy sort/grouping."""
    if points is None or len(points) == 0:
        return
    if not key_columns:
        yield None, points.ravel(), len(points)
        return

    keys = [np.asarray(arr) for arr in key_columns.values()]
    order = np.lexsort(tuple(keys[::-1]))
    points_sorted = points[order]
    keys_sorted = [arr[order] for arr in keys]

    n = points_sorted.shape[0]
    changed = np.zeros(n, dtype=bool)
    changed[0] = True
    for key_arr in keys_sorted:
        left = key_arr[:-1]
        right = key_arr[1:]
        if np.issubdtype(key_arr.dtype, np.floating):
            neq = ~(right == left) & ~(np.isnan(right) & np.isnan(left))
        else:
            neq = right != left
        changed[1:] |= neq

    starts = np.flatnonzero(changed)
    ends = np.r_[starts[1:], n]

    for start, end in zip(starts, ends):
        key_vals = tuple(key_arr[start] for key_arr in keys_sorted)
        key = key_vals[0] if len(key_vals) == 1 else key_vals
        chunk = points_sorted[start:end]
        yield key, chunk.ravel(), (end - start)



def _meshview_entry(idx, name, triplets, r, g, b, count=None):
    """Build a single MeshView JSON entry dict."""
    return {
        "idx": idx,
        "count": len(triplets) // 3 if count is None else count,
        "name": name,
        "triplets": triplets,
        "r": int(r),
        "g": int(g),
        "b": int(b),
    }


def _write_meshview_json(filename: str, payload) -> None:
    """Write MeshView payload to disk with NumPy serialization enabled."""
    with open(filename, "wb") as f:
        f.write(orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY))


def _write_grouped_meshview(
    *,
    points: np.ndarray,
    key_columns: dict[str, np.ndarray],
    filename: str,
    entry_builder,
) -> None:
    """Group points and write MeshView entries via a supplied entry builder."""
    meshview = []
    for entry_idx, (key, triplets, count) in enumerate(
        _group_triplets(points, key_columns)
    ):
        entry = entry_builder(entry_idx, key, triplets, count)
        if entry is not None:
            meshview.append(entry)
    _write_meshview_json(filename, meshview)


def _write_region_meshview(
    points: np.ndarray,
    point_ids: np.ndarray,
    filename: str,
    info_file: pd.DataFrame,
) -> None:
    """Write atlas-region grouped points to MeshView JSON."""
    info_index = info_file.set_index("idx", drop=False)

    def _build_entry(entry_idx, region_id, triplets, count):
        region_id = int(region_id)
        if region_id not in info_index.index:
            return None
        region_row = info_index.loc[region_id]
        return _meshview_entry(
            entry_idx,
            str(region_row["name"]),
            triplets,
            int(region_row["r"]),
            int(region_row["g"]),
            int(region_row["b"]),
            count=count,
        )

    _write_grouped_meshview(
        points=points,
        key_columns={"region": point_ids},
        filename=filename,
        entry_builder=_build_entry,
    )


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
            hemi_path = os.path.join(
                os.path.dirname(filename),
                f"{prefix}{os.path.basename(filename)}",
            )
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

    _write_region_meshview(points, point_ids, filename, info_file)


def _write_intensity_meshview(
    points: np.ndarray,
    intensities: np.ndarray,
    filename: str,
    colormap: str,
) -> None:
    """Write intensity-based point cloud to MeshView JSON.

    Groups points by intensity bins for efficient visualization.
    """
    if (
        colormap == "original_colours"
        and intensities.ndim == 2
        and intensities.shape[1] == 3
    ):
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
        rgb_data = np.round(rgb_data / 8) * 8
        rgb_data = np.clip(rgb_data, 0, 255).astype(np.uint8)
        unique_colors, inverse_indices = np.unique(
            rgb_data, axis=0, return_inverse=True
        )

    rgb_by_gid = {idx: unique_colors[idx] for idx in range(len(unique_colors))}

    def _build_entry(_entry_idx, gid, triplets, count):
        gid = int(gid)
        r, g, b = rgb_by_gid[gid]
        return _meshview_entry(
            gid,
            f"Color {r},{g},{b}",
            triplets,
            r,
            g,
            b,
            count=count,
        )

    _write_grouped_meshview(
        points=points,
        key_columns={"gid": inverse_indices},
        filename=filename,
        entry_builder=_build_entry,
    )


def _write_scalar_meshview(
    points: np.ndarray,
    intensities: np.ndarray,
    filename: str,
    colormap: str,
) -> None:
    """Write grayscale/colormap point cloud to MeshView JSON."""
    if len(intensities) == 0:
        _write_meshview_json(filename, [])
        return

    if intensities.ndim == 2 and intensities.shape[1] == 3:
        # Convert RGB to grayscale for binning
        intensities = (
            0.2989 * intensities[:, 0]
            + 0.5870 * intensities[:, 1]
            + 0.1140 * intensities[:, 2]
        ).astype(int)
    else:
        intensities = intensities.astype(int)

    unique_intensities = np.unique(intensities)
    colors = get_colormap_colors(unique_intensities, colormap)
    color_by_intensity = {
        int(val): (int(r), int(g), int(b))
        for val, (r, g, b) in zip(unique_intensities, colors)
    }

    def _build_entry(_entry_idx, val, triplets, count):
        val = int(val)
        r, g, b = color_by_intensity[val]
        return _meshview_entry(
            val,
            f"Intensity {val}",
            triplets,
            r,
            g,
            b,
            count=count,
        )

    _write_grouped_meshview(
        points=points,
        key_columns={"intensity": intensities},
        filename=filename,
        entry_builder=_build_entry,
    )
