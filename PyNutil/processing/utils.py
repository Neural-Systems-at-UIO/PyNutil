import numpy as np
import pandas as pd

import re
import os
from glob import glob
import cv2


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

#: Standard regex for extracting section numbers from filenames (e.g. "_s001").
SECTION_NUMBER_PATTERN = re.compile(r"\_s(\d+)")
#: Fallback regex when the standard pattern is absent (e.g. "_001").
SECTION_NUMBER_FALLBACK = re.compile(r"\_(\d+)")


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

    if channel == "R":
        return image[:, :, 2].astype(np.float32)
    elif channel == "G":
        return image[:, :, 1].astype(np.float32)
    elif channel == "B":
        return image[:, :, 0].astype(np.float32)
    elif channel == "grayscale" or channel == "auto":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        # Default to grayscale if channel is unknown
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def number_sections(filenames, legacy=False):
    """
    Extract section numbers from a list of filenames.

    Args:
        filenames (list): List of file paths.
        legacy (bool, optional): Use a legacy extraction mode if True. Defaults to False.

    Returns:
        list: List of section numbers as integers.
    """
    filenames = [os.path.basename(filename) for filename in filenames]
    section_numbers = []
    for filename in filenames:
        if not legacy:
            # Try _s### first (standard PyNutil/QUINT format)
            match = re.findall(r"\_s(\d+)", filename)
            if len(match) == 0:
                # Try _### (common alternative)
                match = re.findall(r"\_(\d+)", filename)

            if len(match) == 0:
                raise ValueError(
                    f"No section number found in filename: {filename}. Expected format like '_s001' or '_001'."
                )

            section_numbers.append(int(match[-1]))
        else:
            match = re.sub("[^0-9]", "", filename)
            section_numbers.append(int(match[-3:]))
    if len(section_numbers) == 0:
        raise ValueError("No section numbers found in filenames")
    return section_numbers


def find_matching_pixels(segmentation, id):
    """
    Returns the Y and X coordinates of all the pixels in the segmentation that match the id provided.

    Args:
        segmentation (ndarray): Segmentation array.
        id (int): ID to match.

    Returns:
        tuple: Y and X coordinates of matching pixels.
    """
    mask = segmentation == id
    mask = np.all(mask, axis=2)
    id_positions = np.where(mask)
    id_y, id_x = id_positions[0], id_positions[1]
    return id_y, id_x


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


def update_spacing(anchoring, width, height, grid_spacing):
    """
    Calculates spacing along width and height from slice anchoring.

    Args:
        anchoring (list): Anchoring transformation parameters.
        width (int): Image width.
        height (int): Image height.
        grid_spacing (int): Grid spacing in image units.

    Returns:
        tuple: (xspacing, yspacing)
    """
    if len(anchoring) != 9:
        print("Anchoring does not have 9 elements.")
    ow = np.sqrt(sum([anchoring[i + 3] ** 2 for i in range(3)]))
    oh = np.sqrt(sum([anchoring[i + 6] ** 2 for i in range(3)]))
    xspacing = int(width * grid_spacing / ow)
    yspacing = int(height * grid_spacing / oh)
    return xspacing, yspacing


def create_damage_mask(section, grid_spacing):
    """
    Creates a binary damage mask from grid information in the given section.

    Args:
        section (dict): Dictionary with slice and grid data.
        grid_spacing (int): Space between grid marks.

    Returns:
        ndarray: Binary mask with damaged areas marked as 0.
    """
    width = section["width"]
    height = section["height"]
    anchoring = section["anchoring"]
    grid_values = section["grid"]
    gridx = section["gridx"]
    gridy = section["gridy"]

    xspacing, yspacing = update_spacing(anchoring, width, height, grid_spacing)
    x_coords = np.arange(gridx, width, xspacing)
    y_coords = np.arange(gridy, height, yspacing)

    num_markers = len(grid_values)
    markers = [
        (x_coords[i % len(x_coords)], y_coords[i // len(x_coords)])
        for i in range(num_markers)
    ]

    binary_image = np.ones((len(y_coords), len(x_coords)), dtype=int)

    for i, (x, y) in enumerate(markers):
        if grid_values[i] == 4:
            binary_image[y // yspacing, x // xspacing] = 0

    return binary_image


def get_segmentations(folder):
    """
    Collects segmentation file paths from the specified folder.

    Args:
        folder (str): Path to the folder containing segmentations.

    Returns:
        list: List of segmentation file paths.
    """
    segmentation_file_types = [".png", ".tif", ".tiff", ".jpg", ".jpeg", ".dzip"]
    segmentations = [
        file
        for file in glob(folder + "/*")
        if any([file.endswith(type) for type in segmentation_file_types])
    ]
    if len(segmentations) == 0:
        raise ValueError(
            f"No image files found in folder {folder}. Make sure the folder contains images."
        )
    print(f"Found {len(segmentations)} segmentations in folder {folder}")
    return segmentations


def get_flat_files(folder, use_flat=False):
    """
    Retrieves flat file paths from the given folder.

    Args:
        folder (str): Path to the folder containing flat files.
        use_flat (bool, optional): If True, filter only flat files.

    Returns:
        tuple: A list of flat file paths and their numeric indices.
    """
    if use_flat:
        flat_files = [
            file
            for file in glob(folder + "/flat_files/*")
            if any([file.endswith(".flat"), file.endswith(".seg")])
        ]
        print(f"Found {len(flat_files)} flat files in folder {folder}")
        flat_file_nrs = [int(number_sections([ff])[0]) for ff in flat_files]
        return flat_files, flat_file_nrs
    return [], []


def get_current_flat_file(seg_nr, flat_files, flat_file_nrs, use_flat):
    """
    Determines the correct flat file for a given section number.

    Args:
        seg_nr (int): Numeric index of the segmentation.
        flat_files (list): List of flat file paths.
        flat_file_nrs (list): Numeric indices for each flat file.
        use_flat (bool): If True, attempts to match flat files to segments.

    Returns:
        str or None: The matched flat file path, or None if not found or unused.
    """
    if use_flat:
        current_flat_file_index = np.where([f == seg_nr for f in flat_file_nrs])
        return flat_files[current_flat_file_index[0][0]]
    return None
