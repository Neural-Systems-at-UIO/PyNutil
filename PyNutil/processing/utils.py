import numpy as np
import pandas as pd

import os
import cv2


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
    y_scale = (map_h - 1) / (reg_height - 1) if reg_height > 1 else 0.0
    x_scale = (map_w - 1) / (reg_width - 1) if reg_width > 1 else 0.0
    scaled_y = coords_y * y_scale
    scaled_x = coords_x * x_scale
    iy = np.floor(scaled_y).astype(int)
    ix = np.floor(scaled_x).astype(int)
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
    # Preserve non-atlas rows (e.g. out_of_atlas) before reindexing.
    extra = df[~df["idx"].isin(atlas_labels["idx"])]
    df = df.set_index("idx")
    df = df.reindex(index=atlas_labels["idx"])
    df = df.reset_index()
    if not extra.empty:
        df = pd.concat([df, extra], ignore_index=True)
    return df


# Standard area-fraction column pairs used across the pipeline.
AREA_FRACTION_PAIRS = [
    ("pixel_count", "region_area", "area_fraction"),
    ("left_hemi_pixel_count", "left_hemi_region_area", "left_hemi_area_fraction"),
    ("right_hemi_pixel_count", "right_hemi_region_area", "right_hemi_area_fraction"),
    ("undamaged_pixel_counts", "undamaged_region_area", "undamaged_area_fraction"),
    (
        "left_hemi_undamaged_pixel_counts",
        "left_hemi_undamaged_region_area",
        "left_hemi_undamaged_area_fraction",
    ),
    (
        "right_hemi_undamaged_pixel_counts",
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
        if num not in df.columns or den not in df.columns:
            continue
        mask = df[den] != 0
        df[res] = 0.0
        df.loc[mask, res] = df.loc[mask, num] / df.loc[mask, den]


def apply_mean_intensities(df):
    """Add all standard mean-intensity columns to *df* (in-place)."""
    for num, den, res in MEAN_INTENSITY_PAIRS:
        if num not in df.columns or den not in df.columns:
            continue
        mask = df[den] > 0
        df[res] = 0.0
        df.loc[mask, res] = df.loc[mask, num] / df.loc[mask, den]


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
