import numpy as np
import pandas as pd

import os
import cv2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def safe_area_fraction(df, numerator_col, denominator_col, result_col):
    """Compute *numerator / denominator* with zero-division protection.

    If either column is missing from *df* the function silently does nothing,
    making it safe to call unconditionally for optional column pairs.

    Args:
        df: pandas DataFrame (modified **in-place**).
        numerator_col: Column name for the numerator.
        denominator_col: Column name for the denominator.
        result_col: Column name to write the result into.
    """
    if numerator_col not in df.columns or denominator_col not in df.columns:
        return
    mask = df[denominator_col] != 0
    df[result_col] = 0.0
    df.loc[mask, result_col] = (
        df.loc[mask, numerator_col] / df.loc[mask, denominator_col]
    )


def safe_mean_ratio(df, numerator_col, denominator_col, result_col):
    """Compute *numerator / denominator* where denominator > 0.

    Same contract as :func:`safe_area_fraction` but uses ``> 0`` rather than
    ``!= 0`` as the guard — appropriate for count-based denominators (e.g.
    mean intensity = sum / pixel_count).
    """
    if numerator_col not in df.columns or denominator_col not in df.columns:
        return
    mask = df[denominator_col] > 0
    df[result_col] = 0.0
    df.loc[mask, result_col] = (
        df.loc[mask, numerator_col] / df.loc[mask, denominator_col]
    )


def assign_labels_at_coordinates(coords_y, coords_x, source_map, reg_height, reg_width):
    """Look up values in *source_map* for coordinates scaled to atlas resolution.

    Coordinates are assumed to be in **registration space** (i.e. within
    ``[0, reg_height)`` × ``[0, reg_width)``).  They are scaled down to the
    (potentially smaller) *source_map* resolution, rounded, bounds-checked,
    and used to index *source_map*.

    Args:
        coords_y: 1-D array of Y coordinates in registration space.
        coords_x: 1-D array of X coordinates in registration space.
        source_map: 2-D array to sample from (e.g. atlas_map or hemi_mask).
        reg_height: Registration image height.
        reg_width: Registration image width.

    Returns:
        labels: 1-D array of looked-up values (same length as *coords_y*).
               Out-of-bounds coordinates receive 0.
    """
    map_h, map_w = source_map.shape
    scaled_y = coords_y * (map_h / reg_height)
    scaled_x = coords_x * (map_w / reg_width)
    iy = np.round(scaled_y).astype(int)
    ix = np.round(scaled_x).astype(int)
    valid = (iy >= 0) & (iy < map_h) & (ix >= 0) & (ix < map_w)
    labels = np.zeros(len(coords_y), dtype=source_map.dtype)
    if np.any(valid):
        labels[valid] = source_map[iy[valid], ix[valid]]
    return labels


def resize_mask_nearest(mask, width, height):
    """Resize a mask to (*height*, *width*) using nearest-neighbour interpolation.

    Args:
        mask: 2-D numpy array.
        width: Target width (columns).
        height: Target height (rows).

    Returns:
        Resized array with the same dtype as *mask* (bool masks are preserved).
    """
    is_bool = mask.dtype == bool
    resized = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool) if is_bool else resized


def reindex_to_atlas(df, atlas_labels):
    """Reindex a DataFrame so it contains one row per atlas region.

    Rows not present in *df* are filled with NaN (typically followed by
    ``fillna(0)`` by the caller).

    Args:
        df: DataFrame with an ``"idx"`` column.
        atlas_labels: Atlas labels DataFrame with an ``"idx"`` column.

    Returns:
        DataFrame reindexed to match *atlas_labels["idx"]*.
    """
    df = df.set_index("idx")
    df = df.reindex(index=atlas_labels["idx"])
    df = df.reset_index()
    return df


# Standard area-fraction column pairs used across the pipeline.
AREA_FRACTION_PAIRS = [
    ("pixel_count", "region_area", "area_fraction"),
    ("left_hemi_pixel_count", "left_hemi_region_area", "left_hemi_area_fraction"),
    ("right_hemi_pixel_count", "right_hemi_region_area", "right_hemi_area_fraction"),
    ("undamaged_pixel_count", "undamaged_region_area", "undamaged_area_fraction"),
    (
        "left_hemi_undamaged_pixel_count",
        "left_hemi_undamaged_region_area",
        "left_hemi_undamaged_area_fraction",
    ),
    (
        "right_hemi_undamaged_pixel_count",
        "right_hemi_undamaged_region_area",
        "right_hemi_undamaged_area_fraction",
    ),
]

# Standard mean-intensity column pairs used in the intensity pipeline.
MEAN_INTENSITY_PAIRS = [
    ("sum_intensity", "pixel_count", "mean_intensity"),
    ("left_hemi_sum_intensity", "left_hemi_pixel_count", "left_hemi_mean_intensity"),
    ("right_hemi_sum_intensity", "right_hemi_pixel_count", "right_hemi_mean_intensity"),
]


def apply_area_fractions(df):
    """Add all standard area-fraction columns to *df* (in-place)."""
    for num, den, res in AREA_FRACTION_PAIRS:
        safe_area_fraction(df, num, den, res)


def apply_mean_intensities(df):
    """Add all standard mean-intensity columns to *df* (in-place)."""
    for num, den, res in MEAN_INTENSITY_PAIRS:
        safe_mean_ratio(df, num, den, res)


_CHANNEL_INDEX = {"R": 2, "G": 1, "B": 0}


def convert_to_intensity(image, channel):
    """
    Converts an image to an intensity map based on the specified channel.

    Args:
        image (ndarray): Input image (BGR or grayscale).
        channel (str): Channel to extract ('R', 'G', 'B', 'grayscale', 'auto').

    Returns:
        ndarray: Intensity map as float32.
    """
    if image.ndim == 2:
        return image.astype(np.float32)

    idx = _CHANNEL_INDEX.get(channel)
    if idx is not None:
        return image[:, :, idx].astype(np.float32)

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def scale_positions(id_y, id_x, y_scale, x_scale):
    """
    Scales the Y and X coordinates to the registration space.

    Args:
        id_y (ndarray): Y coordinates.
        id_x (ndarray): X coordinates.
        y_scale (float): Y scaling factor.
        x_scale (float): X scaling factor.

    Returns:
        tuple: Scaled Y and X coordinates.
    """
    id_y = id_y * y_scale
    id_x = id_x * x_scale
    return id_y, id_x


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dzip"}


def discover_image_files(folder):
    """Discover image files in *folder* (case-insensitive, sorted, no dirs).

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        list: Sorted list of image file paths.

    Raises:
        ValueError: If no image files are found.
    """
    paths = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, name))
        and os.path.splitext(name)[1].lower() in _IMAGE_EXTS
    ]
    paths.sort()
    if not paths:
        raise ValueError(
            f"No image files found in folder {folder}. "
            "Make sure the folder contains images."
        )
    print(f"Found {len(paths)} segmentations in folder {folder}")
    return paths


from ..io.loaders import (  # noqa: E402, F401
    number_sections,
    get_flat_files,
    get_current_flat_file,
)
