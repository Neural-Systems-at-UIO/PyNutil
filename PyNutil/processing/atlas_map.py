"""Atlas map construction and region-area computation.

This module groups every function that turns an anchoring vector (or
flat-file path) into a 2-D atlas map and derives per-region statistics
from it.  By concentrating this logic in one place, the
``transforms`` module no longer needs to depend on ``analysis``,
breaking the previous cyclic coupling.

Public API
----------
- :func:`generate_target_slice` — extract a 2-D slice from a 3-D atlas.
- :func:`warp_image` — apply non-linear deformation to an image.
- :func:`load_atlas_image` — load / generate an atlas map for a section.
- :func:`assign_labels_to_image` — replace flat-file pixel values with atlas IDs.
- :func:`calculate_scale_factor` — compute a resize scale factor.
- :func:`flat_to_dataframe` — count region pixels with optional damage/hemi masks.
- :func:`get_region_areas` — build atlas map and compute region areas for a section.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from ..io.loaders import read_flat_file, read_seg_file


# ---------------------------------------------------------------------------
# Slice extraction from 3-D atlas
# ---------------------------------------------------------------------------

def generate_target_slice(ouv, atlas):
    """Generate a 2D slice from a 3D atlas based on orientation vectors.

    Args:
        ouv (list): Orientation vector [ox, oy, oz, ux, uy, uz, vx, vy, vz].
        atlas (ndarray): 3D atlas volume.

    Returns:
        ndarray: 2D slice extracted from the atlas.
    """
    ox, oy, oz, ux, uy, uz, vx, vy, vz = ouv
    width = np.floor(math.hypot(ux, uy, uz)).astype(int) + 1
    height = np.floor(math.hypot(vx, vy, vz)).astype(int) + 1
    data = np.zeros((width, height), dtype=np.uint32).flatten()
    xdim, ydim, zdim = atlas.shape

    y_values = np.arange(height)
    x_values = np.arange(width)

    hx = ox + vx * (y_values / height)
    hy = oy + vy * (y_values / height)
    hz = oz + vz * (y_values / height)

    wx = ux * (x_values / width)
    wy = uy * (x_values / width)
    wz = uz * (x_values / width)

    lx = np.floor(hx[:, None] + wx).astype(int)
    ly = np.floor(hy[:, None] + wy).astype(int)
    lz = np.floor(hz[:, None] + wz).astype(int)

    valid_indices = (
        (0 <= lx) & (lx < xdim) & (0 <= ly) & (ly < ydim) & (0 <= lz) & (lz < zdim)
    ).flatten()

    lxf = lx.flatten()
    lyf = ly.flatten()
    lzf = lz.flatten()

    valid_lx = lxf[valid_indices]
    valid_ly = lyf[valid_indices]
    valid_lz = lzf[valid_indices]

    atlas_slice = atlas[valid_lx, valid_ly, valid_lz]
    data[valid_indices] = atlas_slice

    data_im = data.reshape((height, width))
    return data_im


# ---------------------------------------------------------------------------
# Image warping
# ---------------------------------------------------------------------------

def warp_image(image, deformation, rescaleXY):
    """Warp an image using a deformation function, applying optional resizing.

    Args:
        image (ndarray): Image array to be warped.
        deformation (callable): Deformation function that takes (x, y) arrays
            and returns (new_x, new_y) arrays.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        ndarray: The warped image array.
    """
    if rescaleXY is not None:
        w, h = rescaleXY
    else:
        h, w = image.shape
    reg_h, reg_w = image.shape
    oldX, oldY = np.meshgrid(np.arange(reg_w), np.arange(reg_h))
    oldX = oldX.flatten()
    oldY = oldY.flatten()
    h_scale = h / reg_h
    w_scale = w / reg_w
    oldX = oldX * w_scale
    oldY = oldY * h_scale
    newX, newY = deformation(oldX, oldY)
    newX = newX / w_scale
    newY = newY / h_scale
    newX = newX.reshape(reg_h, reg_w)
    newY = newY.reshape(reg_h, reg_w)
    newX = newX.astype(int)
    newY = newY.astype(int)
    tempX = newX.copy()
    tempY = newY.copy()
    tempX[tempX >= reg_w] = reg_w - 1
    tempY[tempY >= reg_h] = reg_h - 1
    tempX[tempX < 0] = 0
    tempY[tempY < 0] = 0
    new_image = image[tempY, tempX]
    new_image[newX >= reg_w] = 0
    new_image[newY >= reg_h] = 0
    new_image[newX < 0] = 0
    new_image[newY < 0] = 0
    return new_image


# ---------------------------------------------------------------------------
# Label assignment and loading
# ---------------------------------------------------------------------------

def assign_labels_to_image(image, labelfile):
    """Assign atlas or region labels to an image array.

    Args:
        image (ndarray): Image array to label.
        labelfile (DataFrame): Contains label IDs in the 'idx' column.

    Returns:
        ndarray: Image with assigned labels.
    """
    w, h = image.shape
    allen_id_image = np.zeros((h, w))
    coordsy, coordsx = np.meshgrid(list(range(w)), list(range(h)))

    values = image[coordsy, coordsx]
    lbidx = labelfile["idx"].values

    allen_id_image = lbidx[values.astype(int)]
    return allen_id_image


def load_atlas_image(file, image_vector, volume, deformation, rescaleXY, labelfile=None):
    """Load an image from file or generate from atlas volume, optionally warping.

    Args:
        file (str): File path for the source image.
        image_vector (ndarray): Preloaded image data array (anchoring vector).
        volume (ndarray): Atlas volume or similar data.
        deformation (callable or None): Deformation function for warping.
        rescaleXY (tuple): (width, height) for resizing.
        labelfile (DataFrame, optional): Label definitions.

    Returns:
        ndarray: The loaded or transformed image.
    """
    if image_vector is not None and volume is not None:
        image = generate_target_slice(image_vector, volume)
        image = np.float64(image)
        if deformation is not None:
            image = warp_image(image, deformation, rescaleXY)
    else:
        if file.endswith(".flat"):
            image = read_flat_file(file)
        if file.endswith(".seg"):
            image = read_seg_file(file)
        image = assign_labels_to_image(image, labelfile)

    return image


def calculate_scale_factor(image, rescaleXY):
    """Compute a resize scale factor.

    Args:
        image (ndarray): Original image array.
        rescaleXY (tuple): (width, height) for potential resizing.

    Returns:
        float or bool: Scale factor or False if not applicable.
    """
    if rescaleXY:
        image_shapeY, image_shapeX = image.shape[0], image.shape[1]
        image_pixels = image_shapeY * image_shapeX
        seg_pixels = rescaleXY[0] * rescaleXY[1]
        return seg_pixels / image_pixels
    return False


# ---------------------------------------------------------------------------
# Region counting from atlas map
# ---------------------------------------------------------------------------

def count_pixels_per_label(image, scale_factor=False):
    """Count the pixels associated with each label in an image.

    Args:
        image (ndarray): Image array containing labels.
        scale_factor (bool, optional): Apply scaling if True.

    Returns:
        DataFrame: Table of label IDs and pixel counts.
    """
    unique_ids, counts = np.unique(image, return_counts=True)
    if scale_factor:
        counts = counts * scale_factor
    area_per_label = list(zip(unique_ids, counts))
    df_area_per_label = pd.DataFrame(area_per_label, columns=["idx", "region_area"])
    return df_area_per_label


def flat_to_dataframe(image, damage_mask, hemi_mask, rescaleXY=None):
    """Build a DataFrame from an atlas map, with optional damage/hemisphere masks.

    Args:
        image (ndarray): Source image with label IDs.
        damage_mask (ndarray): Binary mask indicating damaged areas.
        hemi_mask (ndarray): Binary mask for hemisphere assignment.
        rescaleXY (tuple, optional): (width, height) for resizing.

    Returns:
        DataFrame: Pixel counts grouped by label.
    """
    scale_factor = calculate_scale_factor(image, rescaleXY)
    df_area_per_label = pd.DataFrame(columns=["idx"])
    if hemi_mask is not None:
        hemi_mask = cv2.resize(
            hemi_mask.astype(np.uint8),
            (image.shape[::-1]),
            interpolation=cv2.INTER_NEAREST,
        )

    if damage_mask is not None:
        damage_mask = cv2.resize(
            damage_mask.astype(np.uint8),
            (image.shape[::-1]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Build combinations for each scenario
    if (hemi_mask is not None) and (damage_mask is not None):
        combos = [
            (1, 0, "left_hemi_undamaged_region_area"),
            (1, 1, "left_hemi_damaged_region_area"),
            (2, 0, "right_hemi_undamaged_region_area"),
            (2, 1, "right_hemi_damaged_region_area"),
        ]
    elif (hemi_mask is not None) and (damage_mask is None):
        combos = [
            (1, 0, "left_hemi_region_area"),
            (2, 0, "right_hemi_region_area"),
        ]
    elif (hemi_mask is None) and (damage_mask is not None):
        combos = [
            (0, 0, "undamaged_region_area"),
            (0, 1, "damaged_region_area"),
        ]
    else:
        combos = [
            (None, None, "region_area")
        ]

    # Count pixels for each combo
    for hemi_val, damage_val, col_name in combos:
        mask = np.ones_like(image, dtype=bool)
        if hemi_mask is not None:
            mask &= hemi_mask == hemi_val
        if damage_mask is not None:
            mask &= damage_mask == damage_val
        combo_df = count_pixels_per_label(image[mask], scale_factor)
        combo_df = combo_df.rename(columns={"region_area": col_name})
        df_area_per_label = pd.merge(
            df_area_per_label, combo_df, on="idx", how="outer"
        ).fillna(0)

    # If both masks exist, compute additional columns
    if (hemi_mask is not None) and (damage_mask is not None):
        df_area_per_label["undamaged_region_area"] = (
            df_area_per_label["left_hemi_undamaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["damaged_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_damaged_region_area"]
        )
        df_area_per_label["left_hemi_region_area"] = (
            df_area_per_label["left_hemi_damaged_region_area"]
            + df_area_per_label["left_hemi_undamaged_region_area"]
        )
        df_area_per_label["right_hemi_region_area"] = (
            df_area_per_label["right_hemi_damaged_region_area"]
            + df_area_per_label["right_hemi_undamaged_region_area"]
        )
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )
    if (hemi_mask is not None) and (damage_mask is None):
        df_area_per_label["region_area"] = (
            df_area_per_label["left_hemi_region_area"]
            + df_area_per_label["right_hemi_region_area"]
        )
    if (hemi_mask is None) and (damage_mask is not None):
        df_area_per_label["region_area"] = (
            df_area_per_label["undamaged_region_area"]
            + df_area_per_label["damaged_region_area"]
        )
    return df_area_per_label


# ---------------------------------------------------------------------------
# High-level: region area computation for a section
# ---------------------------------------------------------------------------

def get_region_areas(
    use_flat: bool,
    atlas_labels,
    flat_file_atlas: Optional[str],
    seg_width: int,
    seg_height: int,
    anchoring: List[float],
    reg_width: int,
    reg_height: int,
    atlas_volume: np.ndarray,
    hemi_mask: Optional[np.ndarray],
    deformation: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]],
    damage_mask: Optional[np.ndarray],
) -> Tuple[Any, np.ndarray]:
    """Build the atlas map for a slice and compute region areas.

    This performs the atlas slice extraction (from volume or flat file), applies
    non-linear warping if requested (via ``deformation``), and converts the
    resulting label map into a region-area dataframe.

    Parameters
    ----------
    use_flat : bool
        Whether to use flat file instead of atlas volume.
    atlas_labels : pd.DataFrame
        Atlas label definitions.
    flat_file_atlas : str or None
        Path to flat file atlas if use_flat is True.
    seg_width, seg_height : int
        Segmentation image dimensions.
    anchoring : list
        Anchoring vector (9 floats).
    reg_width, reg_height : int
        Registration dimensions.
    atlas_volume : np.ndarray
        3D atlas annotation volume.
    hemi_mask : np.ndarray or None
        Hemisphere mask volume.
    deformation : callable or None
        Deformation function for non-linear transformation.
    damage_mask : np.ndarray or None
        Damage mask for the section.

    Returns
    -------
    region_areas : pd.DataFrame
        DataFrame with region area statistics.
    atlas_map : np.ndarray
        2D atlas map for the section.
    """
    atlas_map = load_atlas_image(
        flat_file_atlas,
        anchoring,
        atlas_volume,
        deformation,
        (reg_width, reg_height),
        atlas_labels,
    )
    region_areas = flat_to_dataframe(
        atlas_map, damage_mask, hemi_mask, (seg_width, seg_height)
    )
    return region_areas, atlas_map
