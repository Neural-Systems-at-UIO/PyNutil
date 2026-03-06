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
import os
from typing import Any, Callable, List, Optional, Tuple
from functools import lru_cache

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
    width = int(np.floor(math.hypot(ux, uy, uz))) + 1
    height = int(np.floor(math.hypot(vx, vy, vz))) + 1
    xdim, ydim, zdim = atlas.shape

    # Row/col normalised fractions (float32 saves bandwidth)
    yf = np.arange(height, dtype=np.float32) / height
    xf = np.arange(width, dtype=np.float32) / width

    lx = np.floor(ox + (vx * yf)[:, None] + ux * xf).astype(np.int32)
    ly = np.floor(oy + (vy * yf)[:, None] + uy * xf).astype(np.int32)
    lz = np.floor(oz + (vz * yf)[:, None] + uz * xf).astype(np.int32)

    # Clip to atlas bounds; out-of-bounds pixels will index corner voxels
    # but are zeroed out below via the valid mask.
    valid = (
        (lx >= 0) & (lx < xdim)
        & (ly >= 0) & (ly < ydim)
        & (lz >= 0) & (lz < zdim)
    )

    data_im = np.zeros((height, width), dtype=np.uint32)
    data_im[valid] = atlas[lx[valid], ly[valid], lz[valid]]
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
    w_scale = w / reg_w
    h_scale = h / reg_h
    newX, newY = deformation(oldX.ravel() * w_scale, oldY.ravel() * h_scale)
    newX = (newX / w_scale).reshape(reg_h, reg_w).astype(int)
    newY = (newY / h_scale).reshape(reg_h, reg_w).astype(int)
    oob = (newX < 0) | (newX >= reg_w) | (newY < 0) | (newY >= reg_h)
    new_image = image[np.clip(newY, 0, reg_h - 1), np.clip(newX, 0, reg_w - 1)]
    new_image[oob] = 0
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

    values = image[coordsy, coordsx].astype(int)
    lbidx = labelfile["idx"].values

    valid = (values >= 0) & (values < len(lbidx))
    allen_id_image[valid] = lbidx[values[valid]]
    return allen_id_image


@lru_cache(maxsize=8)
def _read_itksnap_label_lookup(path):
    """Read ITK-SNAP .label and return ordered atlas IDs by label index."""
    ids = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t") if "\t" in s else s.split()
            try:
                ids.append(int(parts[0]))
            except (ValueError, IndexError):
                continue
    return np.asarray(ids, dtype=np.int64)


def _find_nearby_label_file(flat_path):
    """Find a .label file in the flat folder or its parents."""
    current = os.path.dirname(flat_path)
    search_dirs = [current]
    parent = os.path.dirname(current)
    if parent and parent != current:
        search_dirs.append(parent)
    grandparent = os.path.dirname(parent)
    if grandparent and grandparent != parent:
        search_dirs.append(grandparent)

    for directory in search_dirs:
        try:
            for name in os.listdir(directory):
                if name.endswith(".label"):
                    return os.path.join(directory, name)
        except OSError:
            continue
    return None


def load_atlas_image(
    file, image_vector, volume, deformation, rescaleXY, labelfile=None
):
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
            max_value = int(np.max(image)) if image.size else 0
            if max_value >= len(labelfile["idx"].values):
                label_file = _find_nearby_label_file(file)
                if label_file is not None:
                    lookup = _read_itksnap_label_lookup(label_file)
                    if max_value < len(lookup):
                        return lookup[image.astype(int)]
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


def _build_area_combos(hemi_mask, damage_mask):
    """Return list of (hemi_val, damage_val, column_name) for area counting."""
    if (hemi_mask is not None) and (damage_mask is not None):
        return [
            (1, 0, "left_hemi_undamaged_region_area"),
            (1, 1, "left_hemi_damaged_region_area"),
            (2, 0, "right_hemi_undamaged_region_area"),
            (2, 1, "right_hemi_damaged_region_area"),
        ]
    if (hemi_mask is not None) and (damage_mask is None):
        return [
            (1, 0, "left_hemi_region_area"),
            (2, 0, "right_hemi_region_area"),
        ]
    if (hemi_mask is None) and (damage_mask is not None):
        return [
            (0, 0, "undamaged_region_area"),
            (0, 1, "damaged_region_area"),
        ]
    return [(None, None, "region_area")]


def _derive_area_aggregates(df, hemi_mask, damage_mask):
    """Derive aggregate area columns from leaf-level columns in *df*."""
    if (hemi_mask is not None) and (damage_mask is not None):
        df["undamaged_region_area"] = (
            df["left_hemi_undamaged_region_area"]
            + df["right_hemi_undamaged_region_area"]
        )
        df["damaged_region_area"] = (
            df["left_hemi_damaged_region_area"] + df["right_hemi_damaged_region_area"]
        )
        df["left_hemi_region_area"] = (
            df["left_hemi_damaged_region_area"] + df["left_hemi_undamaged_region_area"]
        )
        df["right_hemi_region_area"] = (
            df["right_hemi_damaged_region_area"]
            + df["right_hemi_undamaged_region_area"]
        )
        df["region_area"] = df["undamaged_region_area"] + df["damaged_region_area"]
    if (hemi_mask is not None) and (damage_mask is None):
        df["region_area"] = df["left_hemi_region_area"] + df["right_hemi_region_area"]
    if (hemi_mask is None) and (damage_mask is not None):
        df["region_area"] = df["undamaged_region_area"] + df["damaged_region_area"]


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

    combos = _build_area_combos(hemi_mask, damage_mask)
    total_area_df = count_pixels_per_label(image, scale_factor).rename(
        columns={"region_area": "_region_area_total"}
    )

    # Count pixels for each combo and join all at once (avoids repeated pd.merge)
    combo_dfs = []
    for hemi_val, damage_val, col_name in combos:
        mask = np.ones_like(image, dtype=bool)
        if hemi_mask is not None:
            mask &= hemi_mask == hemi_val
        if damage_mask is not None:
            mask &= damage_mask == damage_val
        combo_df = count_pixels_per_label(image[mask], scale_factor)
        combo_df = combo_df.rename(columns={"region_area": col_name}).set_index("idx")
        combo_dfs.append(combo_df)

    if combo_dfs:
        # Use outer join to preserve all region IDs from any combo
        df_area_per_label = combo_dfs[0]
        for cdf in combo_dfs[1:]:
            df_area_per_label = df_area_per_label.join(cdf, how="outer")
        for col in df_area_per_label.columns:
            if col != "idx":
                df_area_per_label[col] = pd.to_numeric(
                    df_area_per_label[col], errors="coerce"
                )
        df_area_per_label = df_area_per_label.fillna(0).reset_index()
    else:
        df_area_per_label = pd.DataFrame(columns=["idx"])

    _derive_area_aggregates(df_area_per_label, hemi_mask, damage_mask)

    # Nutil computes region area from hemi/damage leaf aggregates when masks
    # are present; only use full-image totals when no masks constrain area.
    if (hemi_mask is None) and (damage_mask is None) and not total_area_df.empty:
        df_area_per_label = df_area_per_label.merge(
            total_area_df,
            on="idx",
            how="outer",
        )
        df_area_per_label["region_area"] = pd.to_numeric(
            df_area_per_label["_region_area_total"], errors="coerce"
        ).fillna(df_area_per_label.get("region_area", 0))
        df_area_per_label.drop(columns=["_region_area_total"], inplace=True)
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
    deformation: Optional[
        Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    ],
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
    image_vector = None if use_flat else anchoring
    volume = None if use_flat else atlas_volume
    atlas_map = load_atlas_image(
        flat_file_atlas,
        image_vector,
        volume,
        deformation,
        (reg_width, reg_height),
        atlas_labels,
    )
    region_areas = flat_to_dataframe(
        atlas_map, damage_mask, hemi_mask, (seg_width, seg_height)
    )
    return region_areas, atlas_map
